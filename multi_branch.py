import numpy as np
import torch
from torch import nn
from models.layers import (
    StatsFeatureExtractor, FrequencyFeatureExtractor, InterpolationLayer
)
import torch.nn.functional as F
from scipy.interpolate import CubicSpline


class CubicSplineUpsample(nn.Module):
    """三次样条插值上采样层"""

    def __init__(self, output_size=24):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # x: (batch, channels, input_size)
        batch, channels, input_size = x.size()
        upsampled = torch.zeros(batch, channels, self.output_size, device=x.device)

        # 原始时间点（输入）
        x_orig = np.linspace(0, 1, input_size)
        # 新时间点（输出）
        x_new = np.linspace(0, 1, self.output_size)

        for b in range(batch):
            for c in range(channels):
                # 获取当前通道的信号
                signal = x[b, c].detach().cpu().numpy()

                # 使用三次样条插值
                cs = CubicSpline(x_orig, signal)
                interpolated = cs(x_new)

                # 转换为张量
                upsampled[b, c] = torch.tensor(interpolated, device=x.device)

        return upsampled


class SimpleUpsample(nn.Module):
    """PyTorch内置插值方法（备选方案）"""

    def __init__(self, output_size=24):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # 使用PyTorch内置的插值方法
        return F.interpolate(
            x,
            size=self.output_size,
            mode='bicubic',
            align_corners=True
        )


class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class MultiBranchModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 统计特征分支
        self.stats_branch = nn.Sequential(
            StatsFeatureExtractor(),
            nn.Linear(60, 720),
            nn.ReLU(),
            InterpolationLayer(12, 24)
        )

        # 频域特征分支 - 使用优化的上采样
        self.freq_branch = nn.Sequential(
            FrequencyFeatureExtractor(),  # [B, 120, 10] => [B, 10, 10]
            # 使用传感器感知的智能上采样
            CubicSplineUpsample(output_size=24),  # [B, 10, 10] => [B, 10, 24]
            nn.Conv1d(10, 480, kernel_size=3, padding=1),  # [B, 480, 24]
            nn.ReLU(),
            PermuteLayer(0, 2, 1)  # [B, 24, 480]
        )

    def forward(self, x):
        # 输入x形状: (batch, 120, 10)

        # 统计特征分支
        stats = self.stats_branch(x)

        # 频域特征分支
        freq = self.freq_branch(x)

        # 特征拼接
        combined = torch.cat([stats, freq], dim=2)

        return combined
