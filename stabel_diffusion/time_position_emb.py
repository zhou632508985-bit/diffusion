import torch
import torch.nn as nn
from config import *
import math

class TimePositionEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        position = torch.arange(0, T).unsqueeze(1)  # [T, 1]
        self.div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(T, dim) # [T, dim]

        pe[:, 0::2] = torch.sin(position * self.div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * self.div_term)  # 奇数维度

        self.register_buffer("pe", pe) # 将pe注册为buffer，不作为模型参数更新

    def forward(self, t):
        """
        :param t: 时间步张量，形状为 (B,)
        :return: 时间步位置编码，形状为 (B, dim)
        """
        return self.pe[t]  # 根据时间步t获取对应的时间步位置编码


        

if __name__ == "__main__":
    # 测试
    time_emb = TimePositionEmbedding(dim=32).to(DEVICE)

    batch_t = torch.tensor([0, 10, 100, 999]).to(DEVICE)  # (B,)
    batch_time_emb = time_emb(batch_t)  # (B, dim)

    print(batch_time_emb)