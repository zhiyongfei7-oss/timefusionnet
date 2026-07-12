import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4,
                                           betas=(0.9, 0.999))  # 新增betas=(0.9, 0.999)
        # 使用 SmoothL1Loss
        self.criterion = nn.SmoothL1Loss()
        # 余弦退火
        self.cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=100,
                                                  eta_min=0.00005)  # 新增eta_min=0.001， T——max：10-100

        self.grad_clip = 1.0  # 梯度裁剪阈值
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # 添加梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
                norm_type=2
            )

            self.optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        # 更新学习率
        self.cosine_scheduler.step()

        # self.scheduler.step()
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, target in dataloader:
                output = self.model(inputs)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(targets)

    def save_model(self, path):
        """保存模型的 state_dict"""
        torch.save(self.model.state_dict(), path)