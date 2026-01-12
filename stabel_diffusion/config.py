import torch

IMG_SIZE = 48 # 图片尺寸
T = 1000 # 加噪步数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设备选择