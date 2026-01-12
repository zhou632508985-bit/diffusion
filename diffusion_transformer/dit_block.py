import torch.nn as nn
import math
import torch

class DiTBlock(nn.Module):

    def __init__(self, emb_size, n_head):
        super().__init__()

        self.emb_size = emb_size
        self.n_head = n_head
        self.head_dim = emb_size // n_head


        # conditioning
        self.gamma1=nn.Linear(emb_size,emb_size)
        self.beta1=nn.Linear(emb_size,emb_size)        
        self.alpha1=nn.Linear(emb_size,emb_size)

        self.gamma2=nn.Linear(emb_size,emb_size)
        self.beta2=nn.Linear(emb_size,emb_size)
        self.alpha2=nn.Linear(emb_size,emb_size)



        # layer norm
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

        # MHA
        self.wq = nn.Linear(emb_size, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(emb_size, self.n_head * self.head_dim, bias=False)
        self.wv = nn.Linear(emb_size, self.n_head * self.head_dim, bias=False)

        self.wo = nn.Linear(self.n_head * self.head_dim, emb_size, bias=False)


        # FFN
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size)
        )


    
    def forward(self, x, cond):
        """
            x: (bsz, seq_len, emb_size)  图片
            cond: (bsz, emb_size)        文字 + 时间序列
        """

        # conditioning (batch,emb_size)
        gamma1_val=self.gamma1(cond)
        beta1_val=self.beta1(cond)
        alpha1_val=self.alpha1(cond)

        gamma2_val=self.gamma2(cond)
        beta2_val=self.beta2(cond)
        alpha2_val=self.alpha2(cond)


        # layer norm
        y = self.layer_norm1(x) 

        # scale & shift
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)


        # attention
        q = self.wq(y)
        k = self.wk(y)
        v = self.wv(y)
        q = q.view(q.size(0), q.size(1), self.n_head, self.head_dim).permute(0, 2, 1, 3) # [bsz, n_head, seq_len, head_dim]
        k = k.view(q.size(0), k.size(1), self.n_head, self.head_dim).permute(0, 2, 3, 1) # [bsz, n_head, seq_len, head_dim]
        v = v.view(v.size(0), v.size(1), self.n_head, self.head_dim).permute(0, 2, 1, 3) # [bsz, n_head, seq_len, head_dim]


        attn = q @ k / math.sqrt(q.size(3))  #  [bsz, n_head, seq_len, seq_len]
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.permute(0, 2, 1, 3) # [bsz, seq_len, n_head, emb_size]
        y = y.reshape(y.size(0), y.size(1), -1)  # [bsz, seq_len, n_head * emb_size]
        y = self.wo(y) # [bsz, seq_len, emb_size]


        # scale
        y=y*alpha1_val.unsqueeze(1)
        # redisual
        y=x+y  
        
        # layer norm
        z=self.layer_norm2(y)
        # scale&shift
        z=z*(1+gamma2_val.unsqueeze(1))+beta2_val.unsqueeze(1)
        # feef-forward
        z=self.ff(z)
        # scale 
        z=z*alpha2_val.unsqueeze(1)
        # residual
        return y+z
    
    


        