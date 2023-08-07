import torch
import torchvision as tv
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class TrainAvatarDataset(Dataset):

    def __init__(self, opt):
        super(TrainAvatarDataset, self).__init__()

        self.dataroot = opt.dataroot
        self.resolution = opt.resolution
        self.loader = tv.datasets.folder.default_loader
        self.transform = tv.transforms.Compose([tv.transforms.Resize(self.resolution), tv.transforms.ToTensor()])

        self.intrinsic = torch.tensor([[5.0000e+03, 0.0000e+00, 2.5600e+02],
                                       [0.0000e+00, 5.0000e+03, 2.5600e+02],
                                       [0.0000e+00, 0.0000e+00, 1.0000e+00]]).float()
        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  4.0000]]).float()
        
        self.samples = []
        video_folder = os.path.join(self.dataroot, opt.video_name)
        image_paths = sorted(glob.glob(os.path.join(video_folder, 'img_*')))

        for i, image_path in enumerate(image_paths):
            mask_path = image_path.replace('img', 'mask')
            param_path = image_path.replace('img', 'params').replace('jpg', 'npz')
            if not os.path.exists(param_path):
                continue
            param = np.load(param_path)
            pose = torch.from_numpy(param['pose'])
            scale = torch.from_numpy(param['scale'])
            sample = (image_path, mask_path, pose, scale)
            self.samples.append(sample)
        

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        image_path = sample[0]
        image = self.transform(self.loader(image_path))
        mask_path = sample[1]
        mask = self.transform(self.loader(mask_path))
        image = image * mask + torch.ones_like(image) * (1 - mask)

        pose = sample[2][0]
        scale = sample[3]

        intrinsic = self.intrinsic
        extrinsic = self.extrinsic

        index = torch.tensor(index).long()

        return {'image': image,
                'mask': mask,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'pose': pose,
                'scale': scale,
                'index': index}

    def __len__(self):
        return len(self.samples)


class TrainYVAEDataset(Dataset):

    def __init__(self, opt):
        super(TrainYVAEDataset, self).__init__()

        self.dataroot = opt.dataroot
        self.resolution = opt.resolution
        self.loader = tv.datasets.folder.default_loader
        self.transform = tv.transforms.Compose([tv.transforms.Resize(self.resolution), tv.transforms.ToTensor()])
        
        video_folder = os.path.join(self.dataroot, opt.video_name_avatar)
        self.samples_avatar = sorted(glob.glob(os.path.join(video_folder, 'img_*')))

        video_folder = os.path.join(self.dataroot, opt.video_name_actor)
        self.samples_actor = sorted(glob.glob(os.path.join(video_folder, 'img_*')))

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        index = random.randint(0, len(self.samples_avatar)-1)
        image_path_avatar = self.samples_avatar[index]
        image_avatar = self.transform(self.loader(image_path_avatar))
        
        index = random.randint(0, len(self.samples_actor)-1)
        image_path_actor = self.samples_actor[index]
        image_actor = self.transform(self.loader(image_path_actor))

        index = torch.tensor(index).long()

        return {'image_avatar': image_avatar,
                'image_actor': image_actor}

    def __len__(self):
        return len(self.samples_avatar) + len(self.samples_actor)
