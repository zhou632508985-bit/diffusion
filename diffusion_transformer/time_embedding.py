import math
import torch
import torch.nn as nn
from config import *

class TimePositionEmbedding(nn.Module):

    def __init__(self, emb_size):
        super().__init__()

        position = torch.arange(0, T).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, emb_size, 2) * -math.log(10000.0) / emb_size)

        pe = torch.zeros(T, emb_size)

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    
    def forward(self, t):
        return self.pe[t]
    

if __name__ == "__main__":
    time_emb = TimePositionEmbedding(16)
    t = torch.randint(0, T, (2,))
    embs = time_emb(t)
    print(embs)