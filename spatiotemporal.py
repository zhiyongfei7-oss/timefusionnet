from torch import nn

from .layers import (
    TimeAttentionLayer_NoLag, ChannelAttentionLayer,
    FinalRegression, SimplifiedFinalRegression
)


class SpatioTemporalNetwork(nn.Module):
    """时空分离网络总成"""

    def __init__(self, output_dim, input_dim=1200):
        super().__init__()
        self.time_attn = TimeAttentionLayer_NoLag(input_dim)
        self.channel_attn = ChannelAttentionLayer(input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.regression = SimplifiedFinalRegression(output_dim)

    def forward(self, x):
        """输入形状：(batch_size, 24, 特征维度)"""
        # 时间注意力
        x = self.norm1(x + self.time_attn(x))

        # 通道注意力
        x = self.norm2(self.channel_attn(x))
        x = self.norm2(x + self.channel_attn(x))


        # 回归输出
        return self.regression(x)


