from config.config import config_avatar
import argparse
import torch
from torch.nn import DataParallel
import os

from lib.dataset.Dataset import TrainAvatarDataset
from lib.module.AvatarModule import AvatarModule
from lib.module.NeuralCameraModule import NeuralCameraModule
from lib.recorder.Recorder import TrainRecorder
from lib.trainer.AvatarTrainer import AvatarTrainer
from lib.utils.util_seed import seed_everything

if __name__ == '__main__':
    seed_everything(11111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_avatar.yaml')
    arg = parser.parse_args()

    cfg = config_avatar()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = TrainAvatarDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    device = torch.device('cuda:%d' % cfg.gpu_ids[0])
    avatarmodule = AvatarModule(cfg.avatarmodule).to(device)
    if os.path.exists(cfg.load_avatarmodule_checkpoint):
        avatarmodule.load_state_dict(torch.load(cfg.load_avatarmodule_checkpoint, map_location=lambda storage, loc: storage))
    avatarmodule = DataParallel(avatarmodule, cfg.gpu_ids)
    
    neural_camera = NeuralCameraModule(avatarmodule, cfg.neuralcamera)
    optimizer = torch.optim.Adam(avatarmodule.parameters(), lr=cfg.lr)
    recorder = TrainRecorder(cfg.recorder)

    trainer = AvatarTrainer(dataloader, avatarmodule, neural_camera, optimizer, recorder, cfg.gpu_ids[0])
    trainer.train(cfg.start_epoch, 1000)
