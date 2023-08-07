from config.config import config_yvae
import argparse
import torch
from torch.nn import DataParallel
import os

from lib.dataset.Dataset import TrainYVAEDataset
from lib.module.YVAEModule import YVAEModule
from lib.module.AvatarModule import AvatarModule
from lib.module.NeuralCameraModule import NeuralCameraModule
from lib.recorder.Recorder import TrainYVAERecorder
from lib.trainer.YVAETrainer import YVAETrainer
from lib.utils.util_seed import seed_everything

if __name__ == '__main__':
    seed_everything(11111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_yvae.yaml')
    arg = parser.parse_args()

    cfg = config_yvae()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = TrainYVAEDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    device = torch.device('cuda:%d' % cfg.gpu_ids[0])
    yvaemodule = YVAEModule(cfg.yvaemodule).to(device)
    if os.path.exists(cfg.load_yvaemodule_checkpoint):
        yvaemodule.load_state_dict(torch.load(cfg.load_yvaemodule_checkpoint, map_location=lambda storage, loc: storage))
    yvaemodule = DataParallel(yvaemodule, cfg.gpu_ids)

    avatarmodule = AvatarModule(cfg.avatarmodule).to(device)
    avatarmodule.load_state_dict(torch.load(cfg.load_avatarmodule_checkpoint, map_location=lambda storage, loc: storage))
    avatarmodule = DataParallel(avatarmodule, cfg.gpu_ids)
    
    optimizer = torch.optim.Adam(yvaemodule.parameters(), lr=cfg.lr)
    recorder = TrainYVAERecorder(cfg.recorder)

    trainer = YVAETrainer(dataloader, yvaemodule, avatarmodule, optimizer, recorder, cfg.gpu_ids[0])
    trainer.train(cfg.start_epoch, 1000)
