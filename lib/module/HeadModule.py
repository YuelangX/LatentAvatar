import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map

from lib.network.MLP import MLP
from lib.network.Generator import Generator
from lib.network.PositionalEmbedding import get_embedder

class HeadModule(nn.Module):
    def __init__(self, opt):
        super(HeadModule, self).__init__()
        
        self.generator = Generator(opt.triplane_res, opt.exp_dim_3d, opt.triplane_dim * 3 // 2)
        self.density_mlp = MLP(opt.density_mlp, last_op=None)
        self.color_mlp = MLP(opt.color_mlp, last_op=None)
        self.pos_embedding, _ = get_embedder(opt.pos_freq)
        self.view_embedding, _ = get_embedder(opt.view_freq)
        self.noise = opt.noise
        self.bbox = opt.bbox

    def forward(self, data):
        B, C, N = data['query_pts'].shape
        query_pts = data['query_pts']
        query_viewdirs = data['query_viewdirs']
        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, 3:, None]
            S = data['scale'][:, :, None]
            query_pts = torch.bmm(R.permute(0,2,1), (query_pts - T)) / S
            query_viewdirs = torch.bmm(R.permute(0,2,1), query_viewdirs)
        query_viewdirs_embedding = self.view_embedding(rearrange(query_viewdirs, 'b c n -> (b n) c'))

        triplanes = self.generate(data)

        plane_dim = triplanes.shape[1] // 3
        plane_x = triplanes[:, plane_dim*0:plane_dim*1, :, :]
        plane_y = triplanes[:, plane_dim*1:plane_dim*2, :, :]
        plane_z = triplanes[:, plane_dim*2:plane_dim*3, :, :]

        u = (query_pts[:, 0:1] - 0.5 * (self.bbox[0][0] + self.bbox[0][1])) / (0.5 * (self.bbox[0][1] - self.bbox[0][0]))
        v = (query_pts[:, 1:2] - 0.5 * (self.bbox[1][0] + self.bbox[1][1])) / (0.5 * (self.bbox[1][1] - self.bbox[1][0]))
        w = (query_pts[:, 2:3] - 0.5 * (self.bbox[2][0] + self.bbox[2][1])) / (0.5 * (self.bbox[2][1] - self.bbox[2][0]))
        vw = rearrange(torch.cat([v, w], dim=1), 'b (t c) n -> b n t c', t=1)
        uw = rearrange(torch.cat([u, w], dim=1), 'b (t c) n -> b n t c', t=1)
        uv = rearrange(torch.cat([u, v], dim=1), 'b (t c) n -> b n t c', t=1)
        feature_x = torch.nn.functional.grid_sample(plane_x, vw, align_corners=True, mode='bilinear')
        feature_y = torch.nn.functional.grid_sample(plane_y, uw, align_corners=True, mode='bilinear')
        feature_z = torch.nn.functional.grid_sample(plane_z, uv, align_corners=True, mode='bilinear')
        feature_x = rearrange(feature_x, 'b c n t -> b c (n t)')
        feature_y = rearrange(feature_y, 'b c n t -> b c (n t)')
        feature_z = rearrange(feature_z, 'b c n t -> b c (n t)')
        feature = feature_x + feature_y + feature_z
        feature = rearrange(feature, 'b c n -> (b n) c')

        density = rearrange(self.density_mlp(feature), '(b n) c -> b c n', b=B)
        if self.training:
            density = density + torch.randn_like(density) * self.noise
        
        color_input = torch.cat([feature, query_viewdirs_embedding], 1)
        color = rearrange(self.color_mlp(color_input), '(b n) c -> b c n', b=B)

        data['density'] = density
        data['color'] = color
        return data

    def generate(self, data):
        code = data['exp_code_3d']
        triplanes = self.generator(code)
        return triplanes
