import torch
from config import *


beta = torch.linspace(0.0001,0.02, T)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(DEVICE)  # [T,1]


def diffusion_forward(batch_x, batch_t):
    """
    扩散正向过程
    :param batch_x: 输入图片张量，形状为 (B, C, H, W)
    :param batch_t: 时间步张量，形状为 (B,)
    :return: 加噪后的图片张量，形状为 (B, C, H, W), 噪声张量，形状为 (B, C, H, W)
    """

    bsz, _, _, _ = batch_x.size()

    # 1. 随机 正态分布噪声
    batch_noise = torch.randn_like(batch_x)

    # 2. 获取对应时间步的 alpha_hat
    batch_alpha_hat = alpha_hat[batch_t].view(bsz, 1, 1, 1)

    # 3. 根据公式计算加噪后的图片
    batch_x_t = torch.sqrt(batch_alpha_hat)*batch_x + torch.sqrt(1 - batch_alpha_hat) * batch_noise
    
    return batch_x_t, batch_noise


if __name__ == "__main__":
    # 测试 -->  打印加噪前后的图片
    import matplotlib.pyplot as plt
    from dataset import MNIST

    train_dataset = MNIST()

    # # 获取两张图片 组成一个batch  (B, C, H, W)
    # batch_x = torch.stack([train_dataset[0][0], train_dataset[1][0]]).to(DEVICE)
    
    # # 打印加噪前的图片
    # plt.figure(figsize=(10,10))
    # plt.subplot(1,2,1)
    # plt.imshow(tensor_to_pil(batch_x[0]))

    # plt.subplot(1,2,2)
    # plt.imshow(tensor_to_pil(batch_x[1]))
    # plt.show()


    # # 加噪
    # # 随机生成时间步
    # batch_x = batch_x * 2 - 1 # 归一化到 [-1, 1] 与 标准正态分布一个区间
    # batch_t = torch.randint(0, T, (2,)).to(DEVICE)  # (B,)
    # print(f"时间步: {batch_t}")
    # batch_x_t, batch_noise = diffusion_foward(batch_x, batch_t)



    # # 打印加噪后的图片
    # plt.figure(figsize=(10,10))
    # plt.subplot(1,2,1)
    # plt.imshow(tensor_to_pil((batch_x_t[0]+1)/2))  # 还原到 [0, 1]
    # plt.subplot(1,2,2)
    # plt.imshow(tensor_to_pil((batch_x_t[1]+1)/2))
    # plt.show()


