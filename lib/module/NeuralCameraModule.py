import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

from pytorch3d.transforms import so3_exponential_map


class NeuralCameraModule(nn.Module):
    def __init__(self, avatarmodule, opt):
        super(NeuralCameraModule, self).__init__()

        self.avatarmodule = avatarmodule
        self.model_bbox = opt.model_bbox
        self.image_size = opt.image_size
        self.N_samples = opt.N_samples
        self.near_far = opt.near_far

    @staticmethod
    def gen_part_rays(extrinsic, intrinsic, resolution, image_size):
         # resolution (width, height)
        rays_o_list = []
        rays_d_list = []
        rot = extrinsic[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, extrinsic[:, :3, 3:])
        c2w = torch.cat((rot, trans.reshape(-1, 3, 1)), dim=2)
        for b in range(intrinsic.shape[0]):
            fx, fy, cx, cy = intrinsic[b, 0, 0], intrinsic[b, 1, 1], intrinsic[b, 0, 2], intrinsic[b, 1, 2]
            res_w = resolution[b, 0].int().item()
            res_h = resolution[b, 1].int().item()
            W = image_size[b, 0].int().item()
            H = image_size[b, 1].int().item()
            i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, res_w, device=c2w.device), torch.linspace(0.5, H-0.5, res_h, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[b, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[b, :3,-1].expand(rays_d.shape)
            rays_o_list.append(rays_o.unsqueeze(0))
            rays_d_list.append(rays_d.unsqueeze(0))
        rays_o_list = torch.cat(rays_o_list, dim=0)
        rays_d_list = torch.cat(rays_d_list, dim=0)
        # rays [B, C, H, W]
        return rearrange(rays_o_list, 'b h w c -> b c h w'), rearrange(rays_d_list, 'b h w c -> b c h w')

    @staticmethod
    def coords_select(image, coords):
        select_rays = []
        for i in range(image.shape[0]):
            select_rays.append(image[i, :, coords[i, :, 1], coords[i, :, 0]].unsqueeze(0))
        select_rays = torch.cat(select_rays, dim=0)
        return select_rays

    @staticmethod
    def gen_near_far_fixed(near, far, samples, batch_size, device):
        nf = torch.zeros((batch_size, 2, samples), device=device)
        nf[:, 0, :] = near
        nf[:, 1, :] = far
        return nf

    def gen_near_far(self, rays_o, rays_d, R, T, S):
        """calculate intersections with 3d bounding box for batch"""
        B = rays_o.shape[0]
        rays_o_can = torch.bmm(R.permute(0,2,1), (rays_o - T)) / S
        rays_d_can = torch.bmm(R.permute(0,2,1), rays_d) / S
        bbox = torch.tensor(self.model_bbox, dtype=rays_o.dtype, device=rays_o.device)
        mask_in_box_batch = []
        near_batch = []
        far_batch = []
        for b in range(B):
            norm_d = torch.linalg.norm(rays_d_can[b], axis=0, keepdims=True)
            viewdir = rays_d_can[b] / norm_d
            viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
            viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
            tmin = (bbox[:, :1] - rays_o_can[b, :, :1]) / viewdir
            tmax = (bbox[:, 1:2] - rays_o_can[b, :, :1]) / viewdir
            t1 = torch.minimum(tmin, tmax)
            t2 = torch.maximum(tmin, tmax)
            near = torch.max(t1, 0)[0]
            far = torch.min(t2, 0)[0]
            mask_in_box = near < far
            mask_in_box_batch.append(mask_in_box)
            near_batch.append((near / norm_d[0]))
            far_batch.append((far / norm_d[0]))
        mask_in_box_batch = torch.stack(mask_in_box_batch)
        near_batch = torch.stack(near_batch)
        far_batch = torch.stack(far_batch)
        return near_batch, far_batch, mask_in_box_batch

    @staticmethod
    def sample_pdf(density, z_vals, rays_d, N_importance):
        r"""sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """
        bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        _, _, _, weights = NeuralCameraModule.integrate(density, z_vals, rays_d)
        weights = weights[..., 1:-1] + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], dtype=weights.dtype, device=weights.device)

        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        sample_z = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        sample_z, _ = torch.sort(sample_z, dim=-1)

        return sample_z

    @staticmethod
    def integrate(density, z_vals, rays_d, color=None, method='nerf'):
        '''Transforms module's predictions to semantically meaningful values.
        Args:
            density: [num_rays, num_samples along ray, 4]. Prediction from module.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            acc_map: [num_rays]. Sum of weights along each ray.
            depth_map: [num_rays]. Estimated distance to object.
        '''

        dists = (z_vals[...,1:] - z_vals[...,:-1]) * 1e2
        dists = torch.cat([dists, torch.ones(1, device=density.device).expand(dists[..., :1].shape) * 1e10], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        if method == 'nerf':
            alpha = 1 - torch.exp(-F.relu(density[...,0])*dists)
        elif method == 'unisurf':
            alpha = density[...,0]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=density.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        acc_map = torch.sum(weights, -1)
        depth_map = torch.sum(weights * z_vals, -1)

        if color == None:
            return None, acc_map, depth_map, weights
        rgb_map = torch.sum(weights[..., None] * color, -2)
        return rgb_map, acc_map, depth_map, weights

    def render_rays(self, data, N_samples=64):
        B, C, N = data['rays_o'].shape

        rays_o = rearrange(data['rays_o'], 'b c n -> (b n) c')
        rays_d = rearrange(data['rays_d'], 'b c n -> (b n) c')
        N_rays = rays_o.shape[0]
        rays_nf = rearrange(data['rays_nf'], 'b c n -> (b n) c')

        near, far = rays_nf[...,:1], rays_nf[...,1:] # [-1,1]
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device).unsqueeze(0)
        z_vals = near*(1-t_vals) + far*t_vals
        z_vals = z_vals.expand([N_rays, N_samples])
        
        # 采样点 coarse
        query_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B)
        query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_samples, 1), '(b n) s c -> b c (n s)', b=B)
        data['query_pts'] = query_pts
        data['query_viewdirs'] = query_viewdirs
        data = self.avatarmodule('head', data)
        density = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
        color = rearrange(data['color'], 'b c (n s) -> (b n) s c', n=N)
        density = torch.cat([density, torch.ones([density.shape[0], 1, density.shape[2]], device=density.device) * 1e8], 1)
        color = torch.cat([color, torch.ones([color.shape[0], 1, color.shape[2]], device=color.device)], 1)
        z_vals = torch.cat([z_vals, torch.ones([z_vals.shape[0], 1], device=z_vals.device) * 1e8], 1)
        
        render_image, render_mask, _, _ = NeuralCameraModule.integrate(density, z_vals, rays_d, color=color, method='nerf')
        render_image = rearrange(render_image, '(b n) c -> b c n', b=B)
        render_mask = rearrange(render_mask, '(b n c) -> b c n', b=B, c=1)
        
        data.update({'render_image': render_image, 'render_mask': render_mask})
        return data
    

    def forward(self, data, resolution):
        B = data['exp_code_3d'].shape[0]
        H = W = resolution // 4
        device = data['exp_code_3d'].device

        rays_o_grid, rays_d_grid = self.gen_part_rays(data['extrinsic'], 
                                                      data['intrinsic'], 
                                                      torch.FloatTensor([[H, W]]).repeat(B, 1), 
                                                      torch.FloatTensor([[self.image_size, self.image_size]]).repeat(B, 1))

        rays_o = rearrange(rays_o_grid, 'b c h w -> b c (h w)')
        rays_d = rearrange(rays_d_grid, 'b c h w -> b c (h w)')

        rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
        R = so3_exponential_map(data['pose'][:, :3])
        T = data['pose'][:, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
        S = data['scale'][:, :, None]
        rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
        for b in range(B):
            rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
            rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]

        render_data = {
            'exp_code_3d': data['exp_code_3d'],
            'pose': data['pose'],
            'scale': data['scale'],
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rays_nf': rays_nf
        }
        render_data = self.render_rays(render_data, N_samples=self.N_samples)

        render_feature = rearrange(render_data['render_image'], 'b c (h w) -> b c h w', h=H)
        render_mask = rearrange(render_data['render_mask'], 'b c (h w) -> b c h w', h=H)
        render_image = self.avatarmodule('upsample', render_feature)
        data['render_feature'] = render_feature
        data['render_image'] = render_image
        data['render_mask'] = render_mask
        return data
