from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2

class TrainRecorder():
    def __init__(self, opt):
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = opt.name
        self.checkpoint_path = opt.checkpoint_path
        self.result_path = opt.result_path
        
        self.save_freq = opt.save_freq
        self.show_freq = opt.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)
    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])
        self.logger.add_scalar('loss_rgb_coarse', log_data['loss_rgb_coarse'], log_data['iter'])
        self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['avatarmodule'].module.state_dict(), '%s/%s/latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['avatarmodule'].module.state_dict(), '%s/%s/epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            
        if log_data['iter'] % self.show_freq == 0:
            print('saving recon results.')
            image = log_data['data']['image'][0].detach().permute(1,2,0).cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            render_image = log_data['data']['render_image'][0].detach().permute(1,2,0).cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            condition_image = torch.clamp(log_data['data']['condition_image'], 0.0, 1.0)[0].permute(1,2,0).detach().cpu().numpy()
            condition_image = (condition_image * 255).astype(np.uint8)[:,:,::-1]
            condition_image = cv2.resize(condition_image, (render_image.shape[0], render_image.shape[1]))

            render_image_coarse = torch.clamp(log_data['data']['render_image_coarse'], 0.0, 1.0)[0].detach().permute(1,2,0).cpu().numpy()
            render_image_coarse = (render_image_coarse * 255).astype(np.uint8)[:,:,::-1]
            render_image_coarse = cv2.resize(render_image_coarse, (render_image.shape[0], render_image.shape[1]))

            result = np.hstack((condition_image, render_image_coarse, render_image, image))
            cv2.imwrite('%s/%s/result_%05d.jpg' % (self.result_path, self.name, log_data['iter']), result)
        

class TrainYVAERecorder():
    def __init__(self, opt):
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = opt.name
        self.checkpoint_path = opt.checkpoint_path
        self.result_path = opt.result_path
        
        self.save_freq = opt.save_freq
        self.show_freq = opt.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)
    
    def log(self, log_data):
        self.logger.add_scalar('loss_avatar', log_data['loss_avatar'], log_data['iter'])
        self.logger.add_scalar('loss_actor', log_data['loss_actor'], log_data['iter'])
        self.logger.add_scalar('loss_code', log_data['loss_code'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['yvaemodule'].module.state_dict(), '%s/%s/latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['yvaemodule'].module.state_dict(), '%s/%s/epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            
        if log_data['iter'] % self.show_freq == 0:
            print('saving recon results.')
            image_avatar = log_data['data']['image_avatar'][0].detach().permute(1,2,0).cpu().numpy()
            image_avatar = (image_avatar * 255).astype(np.uint8)[:,:,::-1]

            recon_image_avatar = log_data['data']['recon_image_avatar'][0].detach().permute(1,2,0).cpu().numpy()
            recon_image_avatar = (recon_image_avatar * 255).astype(np.uint8)[:,:,::-1]

            recon_image_avatar_star = log_data['data']['recon_image_avatar_star'][0].detach().permute(1,2,0).cpu().numpy()
            recon_image_avatar_star = (recon_image_avatar_star * 255).astype(np.uint8)[:,:,::-1]

            image_actor = log_data['data']['image_actor'][0].detach().permute(1,2,0).cpu().numpy()
            image_actor = (image_actor * 255).astype(np.uint8)[:,:,::-1]

            recon_image_actor = log_data['data']['recon_image_actor'][0].detach().permute(1,2,0).cpu().numpy()
            recon_image_actor = (recon_image_actor * 255).astype(np.uint8)[:,:,::-1]

            recon_image_actor_star = log_data['data']['recon_image_actor_star'][0].detach().permute(1,2,0).cpu().numpy()
            recon_image_actor_star = (recon_image_actor_star * 255).astype(np.uint8)[:,:,::-1]

            result = np.vstack([np.hstack((image_avatar, recon_image_avatar, recon_image_avatar_star)), 
                                np.hstack((image_actor, recon_image_actor, recon_image_actor_star))])
            cv2.imwrite('%s/%s/result_%05d.jpg' % (self.result_path, self.name, log_data['iter']), result)


class InferRecorder():
    def __init__(self, opt):
        self.name = opt.name
        self.result_path = opt.result_path

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    def log(self, log_data):
        data = log_data['data']

        image = data['condition_image'][0].permute(1,2,0).cpu().numpy()
        image = cv2.resize((image * 255).astype(np.uint8)[:,:,::-1], (512,512))
        render_image = data['render_image'][0].permute(1,2,0).cpu().numpy()
        render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]
        result = np.hstack([image, render_image])

        cv2.imwrite('%s/%s/image_%04d.jpg' % (self.result_path, self.name, log_data['fid']), result)

