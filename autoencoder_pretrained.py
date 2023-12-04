import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np


import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器部分，使用卷积层和池化层
        self.encoder = nn.Sequential(#3x32x32
            nn.Conv2d(3, 16, 3, padding=1), #16x32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #16x16x16
            nn.Conv2d(16, 8, 3, padding=1), #16x16x16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #16x8x8
            nn.Conv2d(8, 4, 3, padding=1), #4x8x8
            nn.ReLU(),
            #nn.MaxPool2d(2, 2), #4x4x4
        )

        # 解码器部分，使用转置卷积层
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1), #8x4x4
            nn.ReLU(),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#8x8x8
            nn.Conv2d(8, 8, 3, padding=1), #8x8x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#8x16x16
            nn.Conv2d(8, 16, 3, padding=1), #16x16x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#16x32x32
            nn.Conv2d(16, 3, 3, padding=1), #3x32x32
            nn.Sigmoid() #输出范围为[0, 1]
        )
    def forward(self, x):
        # 前向传播过程，先编码后解码，返回重构后的图片
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_autoencoder():
    model = Autoencoder().cuda() # 创建模型实例
    model.load_state_dict(torch.load("params.pkl"))
    model.eval()
    return model