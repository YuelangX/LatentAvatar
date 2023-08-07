import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from lib.preprocess.FaceAlignment import FaceAlignment


class AvatarTrainer():
    def __init__(self, dataloader, avatarmodule, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.facealignment = FaceAlignment(device='cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
    
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):

                to_cuda = ['image', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                condition_image = data['image']
                condition_image = torch.clamp(condition_image + torch.randn_like(condition_image) * 0.05, 0.0, 1.0)
                condition_image = self.facealignment.get_face(condition_image)
                if condition_image is None:
                    continue
                
                image = data['image']
                if idx < 100 and epoch == 0:
                    image = image - torch.randn_like(image).abs() * 0.1
                image_coarse = F.interpolate(image, scale_factor=0.25)
                mask = data['mask']
                mask_coarse = F.interpolate(mask, scale_factor=0.25)

                exp_code_2d = self.avatarmodule('encode', condition_image)
                exp_code_3d = self.avatarmodule('mapping', exp_code_2d)
                data['exp_code_3d'] = exp_code_3d

                data = self.camera(data, image.shape[2])
                render_image = data['render_image']
                render_image_coarse = data['render_feature'][:,0:3,:,:]
                render_mask_coarse = data['render_mask']

                loss_rgb = F.l1_loss(render_image, image)
                loss_rgb_coarse = F.l1_loss(render_image_coarse, image_coarse)
                loss_mask = F.mse_loss(render_mask_coarse, mask_coarse)
                loss_vgg = self.fn_lpips(render_image - 0.5, image - 0.5).mean()
                
                loss = loss_rgb + loss_rgb_coarse + loss_mask * 1e-1 + loss_vgg * 1e-1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                data['condition_image'] = condition_image
                data['render_image'] = render_image
                data['render_image_coarse'] = render_image_coarse
                log = {
                    'data' : data,
                    'avatarmodule' : self.avatarmodule,
                    'camera' : self.camera,
                    'loss_rgb': loss_rgb,
                    'loss_rgb_coarse': loss_rgb_coarse,
                    'loss_vgg': loss_vgg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)
