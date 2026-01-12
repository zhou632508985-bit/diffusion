import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from config import *


pil_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图片大小
    transforms.ToTensor() # 转换为张量
])

tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda x: x*255), # 像素还原
    transforms.Lambda(lambda x: x.type(torch.uint8)), # 像素值取整数
    transforms.ToPILImage()   # tensor转PIL图片
])


train_dataset = torchvision.datasets.MNIST(root=".", train=True, transform=pil_to_tensor, download=True)


if __name__ == "__main__":

    img_tensor, label = train_dataset[1]

    plt.figure(figsize=(5,5))
    pil_img = tensor_to_pil(img_tensor)
    plt.imshow(pil_img)
    plt.show()