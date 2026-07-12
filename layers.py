import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAttentionLayer(nn.Module):
    """时间注意力层，带时滞效应建模"""

    def __init__(self, dim, num_heads=4, lag_window=15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lag_window = lag_window

        # 时滞权重矩阵
        # 可学习的时滞参数（调整为动态长度）
        self.lag_weights = nn.Parameter(torch.linspace(0.5, 1.0, lag_window))
        self.lag_proj = nn.Linear(lag_window, 24)

        # 注意力参数
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # 残差连接
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """输入形状：(batch_size, seq_len, feature_dim)"""
        B, T, C = x.shape

        # 生成QKV
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # 生成相对位置矩阵
        indices = torch.arange(T, device=x.device)
        relative_pos = indices.view(1, T) - indices.view(T, 1)  # (24,24)

        # 限制时滞窗口并创建掩码
        relative_pos = torch.clamp(relative_pos, 0, self.lag_window - 1)
        valid_mask = (relative_pos >= 0) & (relative_pos < self.lag_window)

        # 构建时滞效应矩阵 (24,24)
        lag_matrix = torch.zeros(T, T, device=x.device)
        lag_matrix[valid_mask] = self.lag_weights[relative_pos[valid_mask]]

        # 扩展维度匹配注意力头 (32,4,24,24)
        lag_effect = lag_matrix.unsqueeze(0).unsqueeze(0)  # (1,1,24,24)
        lag_effect = lag_effect.repeat(B, self.num_heads, 1, 1)  # (32,4,24,24)

        # 添加时滞效应
        attn_scores += lag_effect  # 直接相加确保维度一致

        # 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # 残差连接
        return x + self.res_scale * out


class ChannelAttentionLayer(nn.Module):
    """通道注意力层"""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """输入形状：(batch_size, seq_len, channels)"""
        B, T, C = x.shape

        # 通道统计
        avg_out = self.mlp(self.avg_pool(x.permute(0, 2, 1)).view(B, C))
        max_out = self.mlp(self.max_pool(x.permute(0, 2, 1)).view(B, C))
        channel_weights = avg_out + max_out

        # 特征重标定
        return x * channel_weights.unsqueeze(1)  # 广播到时间维度


class HierarchicalPooling(nn.Module):
    """全局分层池化"""

    def __init__(self, pool_levels=[1, 3, 5]):
        super().__init__()
        self.pool_levels = pool_levels
        self.pool_ops = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(pl),
                nn.Flatten()
            ) for pl in pool_levels
        ])

    def forward(self, x):
        """输入形状：(batch_size, seq_len, channels)"""
        x = x.permute(0, 2, 1)  # (B,C,T)
        pooled = [op(x) for op in self.pool_ops]
        return torch.cat(pooled, dim=1)


class SimplifiedFinalRegression(nn.Module):
    """最终回归网络"""

    def __init__(self, output_dim, input_channels=1200, pool_levels=[1, 3, 5], hidden_dim=128):
        super().__init__()
        self.pooling = HierarchicalPooling()
        total_pool_size = input_channels * sum(pool_levels)  # 计算总池化维度
        self.compress = nn.Sequential(
            nn.Linear(total_pool_size, hidden_dim),  # 根据pool_levels计算
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        self.regressor = nn.Linear(32, output_dim)

    def forward(self, x):
        """输入形状：(batch_size, seq_len, channels)"""
        pooled = self.pooling(x)
        compressed = self.compress(pooled)
        return self.regressor(compressed)



class FinalRegression(nn.Module):
    """最终回归网络"""

    def __init__(self, output_dim, input_channels=480, pool_levels=[1, 3, 5]):
        super().__init__()
        self.pooling = HierarchicalPooling()
        total_pool_size = input_channels * sum(pool_levels)  # 计算总池化维度
        self.compress = nn.Sequential(
            nn.Linear(total_pool_size, 128),  # 根据pool_levels计算
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """输入形状：(batch_size, seq_len, channels)"""
        pooled = self.pooling(x)
        return self.compress(pooled)



class StatsFeatureExtractor(nn.Module):
    def __init__(self, window=5):
        super().__init__()
        self.window = window

    def forward(self, x):
        # x形状: (B, T, C)
        B, T, C = x.shape
        x = x.unfold(1, self.window, self.window)  # (B, num_windows, C, window)

        # 计算统计特征1,2,3
        means = x.mean(dim=-1)
        vars = x.var(dim=-1)
        maxs = x.max(dim=-1)[0]

        # 计算斜率特征4
        t = torch.arange(self.window, device=x.device).float()
        t_mean = t.mean()
        x_centered = x - means.unsqueeze(-1)
        t_centered = t - t_mean
        numerator = (x_centered * t_centered).sum(dim=-1)
        denominator = (t_centered ** 2).sum()
        slopes = numerator / denominator

        # 新特征5
        x_lag0 = x[..., :-1]  # 当前值
        x_lag1 = x[..., 1:]  # 滞后一步值
        cov = ((x_lag0 - x_lag0.mean(dim=-1, keepdim=True)) *
               (x_lag1 - x_lag1.mean(dim=-1, keepdim=True))).mean(dim=-1)
        std_product = x_lag0.std(dim=-1, unbiased=False) * x_lag1.std(dim=-1, unbiased=False)
        autocorr = cov / (std_product + 1e-6)  # 防止除零

        # 新特征6：累积绝对变化量
        diffs = x[..., 1:] - x[..., :-1]
        cumulative_change = torch.abs(diffs).sum(dim=-1)

        # 拼接六个特征
        features = torch.stack([means, vars, maxs, slopes,
                                autocorr, cumulative_change], dim=-1)
        return features.view(B, -1, 60)  # (B,12,60)


class FrequencyFeatureExtractor(nn.Module):
    def forward(self, x):
        # x形状: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        fft = torch.fft.rfft(x, dim=-1).abs()
        top10 = fft[:, :, :20]  # 取前10个频率分量
        return top10


class InterpolationLayer(nn.Module):
    def __init__(self, in_len, out_len):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len

    def forward(self, x):
        return F.interpolate(x.permute(0, 2, 1),
                             size=self.out_len,
                             mode='linear').permute(0, 2, 1)


class TimeAttentionLayer_NoLag(nn.Module):
    """消融实验：移除时滞效应矩阵，其余结构完全不变"""
    def __init__(self, dim, num_heads=4, lag_window=15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lag_window = lag_window  # 保留参数定义，保证初始化一致

        # 注意力参数（不变）
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # 残差连接（不变）
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """输入形状：(batch_size, seq_len, feature_dim)"""
        B, T, C = x.shape

        # QKV 计算（不变）
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 原始注意力分数（不变）
        attn_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # ===================== 核心消融 =====================
        # 完全移除：相对位置、时延矩阵、时延加权
        # 直接做 softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和（不变）
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # 残差连接（不变）
        return x + self.res_scale * out


