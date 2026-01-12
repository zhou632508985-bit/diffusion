import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, time_emb_dim):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), # 改通道数 保持尺寸
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.time_emb_linear = nn.Linear(time_emb_dim, out_channel) # 时间步嵌入 线性层

        self.relu = nn.ReLU()

        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1), # 保持通道数和尺寸
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, t_emb):
        """
        :param x: 输入图片张量，形状为 (B, in_channel, H, W)
        :param t_emb: 时间步嵌入张量，形状为 (B, time_emb_dim)
        """

        # 1. 改通道数， 不改尺寸
        x = self.seq1(x) # [B, out_channel, H, W]

        # 2. 时间步embedding 转 out_channel宽， 
        t_emb = self.time_emb_linear(t_emb) # [B, out_channel]
        t_emb = self.relu(t_emb.view(x.size(0), x.size(1), 1, 1)) # [B, out_channel, 1, 1]

        # 3. 时间步embedding 加到每个像素点上
        return self.seq2(x + t_emb)  # 保持通道数和尺寸


if __name__ == "__main__":
    pass