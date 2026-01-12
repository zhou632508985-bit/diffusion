import torch
from config import * 
from diffusion import *
import matplotlib.pyplot as plt

from dataset import tensor_to_pil
import torch.nn as nn


def inference(model, batch_x_t):

    """
        batch_x_t: 纯噪声图
    """
    global alpha 
    global beta


    steps = [batch_x_t, ]

    alpha = alpha.to(DEVICE)
    beta = beta.to(DEVICE)
    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)


    with torch.no_grad():
        for t in range(T-1, -1, -1):
            # [bsz,1] 
            batch_t = torch.full((batch_x_t.size(0),), t).to(DEVICE)

            batch_predict_noise_t = model(batch_x_t, batch_t)

            # 生成 t-1 时间步的图像
            shape = (batch_x_t.size(0), 1, 1, 1)
            alpha_t = alpha[batch_t].view(*shape)
            alpha_t_hat = alpha_hat[batch_t].view(*shape)
            beta_t = beta[batch_t].view(*shape)

            batch_x_t_pre =  1 / torch.sqrt(alpha_t) * (batch_x_t - ((1-alpha_t)/torch.sqrt(1-alpha_t_hat)) * batch_predict_noise_t)

            if t != 0:
                batch_x_t = batch_x_t_pre + torch.randn_like(batch_x_t) * beta_t

            else:
                batch_x_t = batch_x_t_pre

            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)

    return steps
            
    
if __name__ == "__main__":

    model = torch.load("model.pt",weights_only=False)

    # 生成噪音图
    batch_size = 10
    batch_x_t = torch.randn(size=(batch_size, 1, IMG_SIZE, IMG_SIZE)) # (5,1,48,48)

    steps = inference(model, batch_x_t)

    num_imgs = 20

    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(0, num_imgs):
            idx = int(T/num_imgs) * (i+1)

            final_img = (steps[idx][b].to('cpu') + 1) / 2

            final_img = tensor_to_pil(final_img)
            plt.subplot(batch_size, num_imgs, b*num_imgs+i+1)
            plt.imshow(final_img)
    plt.show()