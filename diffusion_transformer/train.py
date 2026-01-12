import os
import sys

from diffusion import diffusion_forward
from dit import DiT
from config import *
from dataset import MNIST

from torch.utils.data import DataLoader
import torch 
from torch import nn 




DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备

dataset=MNIST() # 数据集

model=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE) # 模型

try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

'''
    训练模型
'''

EPOCH=500
BATCH_SIZE=100

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)    # 数据加载器

model.train()

iter_count=0
for epoch in range(EPOCH):
    for imgs,labels in dataloader:
        x=imgs*2-1 # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
        t=torch.randint(0,T,(imgs.size(0),)).to(DEVICE)  # 为每张图片生成随机t时刻
        y=labels
        x = x.to(DEVICE)
        x,noise=diffusion_forward(x,t) # x:加噪图 noise:噪音
        pred_noise=model(x,t,y.to(DEVICE))

        loss=loss_fn(pred_noise,noise)
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        
        if iter_count%1000==0:
            print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
        iter_count+=1