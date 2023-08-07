from config.config import config_avatar
import argparse
import torch
from torch.nn import DataParallel
import os

from lib.dataset.Dataset import TrainAvatarDataset
from lib.module.YVAEModule import YVAEModule
from lib.module.AvatarModule import AvatarModule
from lib.module.NeuralCameraModule import NeuralCameraModule
from lib.recorder.Recorder import InferRecorder
from lib.inferrer.Cross_Reenactment import Cross_Reenactment
from lib.utils.util_seed import seed_everything

if __name__ == '__main__':
    seed_everything(11111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/cross_reenactment.yaml')
    arg = parser.parse_args()

    cfg = config_avatar()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = TrainAvatarDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device('cuda:%d' % cfg.gpu_ids[0])
    yvaemodule = YVAEModule(cfg.yvaemodule).to(device)
    yvaemodule.load_state_dict(torch.load(cfg.load_yvaemodule_checkpoint, map_location=lambda storage, loc: storage))
    yvaemodule = DataParallel(yvaemodule, cfg.gpu_ids)
    avatarmodule = AvatarModule(cfg.avatarmodule).to(device)
    avatarmodule.load_state_dict(torch.load(cfg.load_avatarmodule_checkpoint, map_location=lambda storage, loc: storage))
    
    neural_camera = NeuralCameraModule(avatarmodule, cfg.neuralcamera)
    optimizer = torch.optim.Adam(avatarmodule.parameters(), lr=cfg.lr)
    recorder = InferRecorder(cfg.recorder)

    inferrer = Cross_Reenactment(dataloader, yvaemodule, avatarmodule, neural_camera, recorder, cfg.gpu_ids[0])
    inferrer.infer()