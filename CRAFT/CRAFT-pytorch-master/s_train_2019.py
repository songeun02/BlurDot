# 12/13 최종 

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from craft import CRAFT
from S_loss import Loss
import os
import time
import numpy as np
from S_config import cfg
from S_dataset import SynthTextDataset
from S_sync_batchnorm import convert_model

# 저장 경로
SAVE_PATH = "./models/"

# 저장 모델구조 및 파라미터 모두 저장
SAVE_MODEL = "top_model_all.pth"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def train(cfg):
    device = "cpu"
	# model = CRAFT()
	# model = convert_model(model)
    model = CRAFT(pretrained=True)
    state_dict = torch.load('/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/weights/craft_mlt_25k.pth', map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict)
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)

    trainset = SynthTextDataset(cfg)
    train_loader = data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
        num_workers=cfg.num_workers, drop_last=cfg.drop_last)
    file_num = len(trainset)
    batch_num = int(file_num / cfg.batch_size)
    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, verbose=True
    )

    loss_history = []
    for epoch in range(cfg.epoch_iter):
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_region, gt_affinity, conf_map) in enumerate(train_loader):
            model.train()
            start_time = time.time()
            img, gt_region, gt_affinity, conf_map = list(
                map(lambda x: x.to(device), [img, gt_region, gt_affinity, conf_map]))
            pred_region, pred_affinity = model(img)
            loss = criterion(gt_region, pred_region, gt_affinity, pred_affinity, conf_map)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                'Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                    epoch + 1, cfg.epoch_iter, i + 1, batch_num, time.time() - start_time, loss.item()))
            loss_history.append(loss.item())

            torch.save(model.state_dict(), f'{SAVE_PATH}top_model_all_{np.round(loss.item())}_wbs.pth')

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
            epoch_loss / batch_num, time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)


if __name__ == '__main__':
    train(cfg.train)
