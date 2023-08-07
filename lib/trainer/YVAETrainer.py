import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from lib.preprocess.FaceAlignment import FaceAlignment


class YVAETrainer():
    def __init__(self, dataloader, yvaemodule, avatarmodule, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.yvaemodule = yvaemodule
        self.avatarmodule = avatarmodule
        self.optimizer = optimizer
        self.recorder = recorder
        self.domains = yvaemodule.module.domains
        self.device = torch.device('cuda:%d' % gpu_id)
        self.facealignment = FaceAlignment(device='cuda:%d' % gpu_id)
    
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):

                to_cuda = ['image_avatar', 'image_actor']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                image_avatar = data['image_avatar']
                image_avatar = self.facealignment.get_face(image_avatar)
                if image_avatar is not None:
                    input_image = torch.clamp(image_avatar + torch.randn_like(image_avatar) * 0.05, 0.0, 1.0)

                    with torch.no_grad():
                        exp_code_2d_avatar_gt = self.avatarmodule('encode', input_image)
                        exp_code_3d_avatar_gt = self.avatarmodule('mapping', exp_code_2d_avatar_gt)

                    exp_code_2d_avatar = self.yvaemodule('encode', input_image)
                    recon_image_avatar = self.yvaemodule('decode', (exp_code_2d_avatar, self.domains[0]))
                    exp_code_3d_avatar = self.yvaemodule('mapping', exp_code_2d_avatar)
                    
                    ssim_loss = 1 - ssim(recon_image_avatar, image_avatar, data_range=1.0, size_average=True)
                    l1_loss = F.l1_loss(recon_image_avatar, image_avatar)
                    loss_image = ssim_loss + l1_loss
                    loss_code = F.mse_loss(exp_code_3d_avatar, exp_code_3d_avatar_gt)

                    with torch.no_grad():
                        recon_image_avatar_star = self.yvaemodule('decode', (exp_code_2d_avatar, self.domains[1]))
                    exp_code_2d_avatar_star = self.yvaemodule('encode', recon_image_avatar_star)
                    loss_cycle = F.mse_loss(exp_code_2d_avatar, exp_code_2d_avatar_star)

                    loss_avatar = loss_image * 1e-1 + loss_code + loss_cycle * 1e-5
                    self.optimizer.zero_grad()
                    loss_avatar.backward()
                    self.optimizer.step()

                image_actor = data['image_actor']
                image_actor = self.facealignment.get_face(image_actor)
                if image_actor is not None:
                    input_image = torch.clamp(image_actor + torch.randn_like(image_actor) * 0.05, 0.0, 1.0)

                    exp_code_2d_actor = self.yvaemodule('encode', input_image)
                    recon_image_actor = self.yvaemodule('decode', (exp_code_2d_actor, self.domains[1]))
                    
                    ssim_loss = 1 - ssim(recon_image_actor, image_actor, data_range=1.0, size_average=True)
                    l1_loss = F.l1_loss(recon_image_actor, image_actor)
                    loss_image = ssim_loss + l1_loss

                    with torch.no_grad():
                        recon_image_actor_star = self.yvaemodule('decode', (exp_code_2d_actor, self.domains[0]))
                    exp_code_2d_actor_star = self.yvaemodule('encode', recon_image_actor_star)
                    loss_cycle = F.mse_loss(exp_code_2d_actor, exp_code_2d_actor_star)

                    loss_actor = loss_image + loss_cycle * 1e-4
                    self.optimizer.zero_grad()
                    loss_actor.backward()
                    self.optimizer.step()

                if image_avatar is not None and image_actor is not None:
                    data['image_avatar'] = image_avatar
                    data['image_actor'] = image_actor
                    data['recon_image_avatar'] = recon_image_avatar
                    data['recon_image_actor'] = recon_image_actor
                    data['recon_image_avatar_star'] = recon_image_avatar_star
                    data['recon_image_actor_star'] = recon_image_actor_star
                    log = {
                        'data' : data,
                        'yvaemodule' : self.yvaemodule,
                        'loss_avatar': loss_avatar,
                        'loss_actor': loss_actor,
                        'loss_code': loss_code,
                        'epoch' : epoch,
                        'iter' : idx + epoch * len(self.dataloader)
                    }
                    self.recorder.log(log)
