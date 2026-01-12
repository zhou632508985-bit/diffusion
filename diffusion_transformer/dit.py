import torch
import torch.nn as nn
from time_embedding import TimePositionEmbedding
from dit_block import DiTBlock

class DiT(nn.Module):

    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super().__init__()


        self.patch_size = patch_size  
        self.patch_count = img_size // self.patch_size
        self.channel = channel

        # patchify
        self.conv = nn.Conv2d(in_channels=channel, 
                              out_channels=channel*patch_size**2, 
                              kernel_size=patch_size, 
                              padding=0,
                              stride=patch_size
                              )   
        
        self.patch_emb = nn.Linear(in_features=channel*patch_size**2, out_features=emb_size)
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count**2, emb_size))

        
        # time_embedding
        self.time_emb = nn.Sequential(
            TimePositionEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # label_emb
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)


        # dit blocks
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))


        # layer norm
        self.ln = nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear = nn.Linear(emb_size, channel*patch_size**2) 

    
    def forward(self, x, t, y):
        """
            x: [bsz, channel, H, W]
            t: [bsz, 1]
            y: [bsz, 1]
        """

        # ==== tranformer 输入向量构造 ====

        # 1. label emb
        y_emb = self.label_emb(y)  # [bsz, emb_size]

        # 2. time emb
        t_emb = self.time_emb(t)  # [bsz, emb_size]


        # 3. condition emb
        cond = y_emb + t_emb

        # 4. patch emb
        x = self.conv(x)  # [bsz, out_channel, patch_count, patch_count]
        x = x.permute(0,2,3,1)  # [bsz, patch_count, patch_count, out_channel]
        x = x.view(x.size(0), self.patch_count**2 , x.size(3))

        x = self.patch_emb(x) # [bsz, patch_count**2, emb_size]
        x = x + self.patch_pos_emb # [bsz, patch_count**2, emb_size]


        for dit in self.dits:
            x = dit(x, cond)

        # layer norm
        x = self.ln(x)

        # linear back to patch
        x = self.linear(x)   # (batch,patch_count**2,channel*patch_size*patch_size)

        # reshape  还原回原来图片的形状
        x = x.view(x.size(0), self.patch_count, self.patch_count, self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)  # [bsz, channel, patch_count(H), patch_count(W), patch_size(H), patch_size(W)]
        x = x.permute(0, 1, 2, 4, 3, 5)  # [bsz, channel, patch_count(H), patch_size(H), patch_count(W),patch_size(W)]
        x = x.reshape(x.size(0),self.channel,self.patch_count*self.patch_size,self.patch_count*self.patch_size)   # (batch,channel,img_size,img_size)
        return x


        
