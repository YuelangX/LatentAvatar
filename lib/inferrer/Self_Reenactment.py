import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.preprocess.FaceAlignment import FaceAlignment


class Self_Reenactment():
    def __init__(self, dataloader, avatarmodule, camera, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.recorder = recorder
        if gpu_id is None:
            self.device = torch.device('cpu')
            self.facealignment = FaceAlignment(device='cpu')
        else:
            self.device = torch.device('cuda:%d' % gpu_id)
            self.facealignment = FaceAlignment(device='cuda:%d' % gpu_id)

        self.end_frame = len(dataloader)
    
    def infer(self, end_frame=None):
        if not end_frame:
            end_frame = self.end_frame
        for idx, data in tqdm(enumerate(self.dataloader)):
            if idx > end_frame:
                break

            to_cuda = ['image', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale']
            for data_item in to_cuda:
                data[data_item] = data[data_item].to(device=self.device)
            
            condition_image = data['image']
            condition_image = self.facealignment.get_face(condition_image)
            if condition_image is None:
                continue

            with torch.no_grad():
                exp_code_2d = self.avatarmodule('encode', condition_image - 0.5)
                exp_code_3d = self.avatarmodule('mapping', exp_code_2d)
                data['exp_code_3d'] = exp_code_3d
                data = self.camera(data, 512)
                render_image = data['render_image']

            data['condition_image'] = condition_image
            data['render_image'] = render_image
            log = {
                'data': data,
                'fid' : idx
            }
            self.recorder.log(log)
