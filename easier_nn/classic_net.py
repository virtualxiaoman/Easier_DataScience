import torch
from torch import nn


# 此代码为copy，未验证
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # 模块化结构，这也是后面常用到的模型结构
        self.first_block_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.second_block_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.latent_space_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), torch.nn.GELU(),
        )

        self.second_block_up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), torch.nn.GELU(),
        )

        self.first_block_up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1), torch.nn.GELU(),
        )

        self.convUP_end = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, img_tensor):
        image = img_tensor

        image = self.first_block_down(image)  # ;print(image.shape)     # torch.Size([5, 32, 14, 14])
        image = self.second_block_down(image)  # ;print(image.shape)    # torch.Size([5, 16, 7, 7])
        image = self.latent_space_block(image)  # ;print(image.shape)   # torch.Size([5, 8, 7, 7])

        image = self.second_block_up(image)  # ;print(image.shape)      # torch.Size([5, 16, 14, 14])
        image = self.first_block_up(image)  # ;print(image.shape)       # torch.Size([5, 32, 28, 28])
        image = self.convUP_end(image)  # ;print(image.shape)           # torch.Size([5, 32, 28, 28])
        return image


# 这是一个在Mnist数据集上的卷积神经网络
class CNN_Net_for_Mnist(nn.Module):
    def __init__(self):
        super(CNN_Net_for_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 这是一个在Mnist数据集上的深度可分离卷积神经网络
class Depth_CNN_Net_for_Mnist(nn.Module):
    def __init__(self):
        super(Depth_CNN_Net_for_Mnist, self).__init__()
        depth_conv = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, groups=6, dilation=2)
        point_conv = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1)
        depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)  # 深度，可分离，膨胀卷积
        self.convs_stack = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7),
            nn.ReLU(),
            depthwise_separable_conv,
            nn.ReLU(),
            nn.Conv2d(24, 6, kernel_size=3)
        )

        self.logits_layer = nn.Linear(in_features=1536, out_features=10)

    def forward(self, inputs):
        image = inputs
        x = self.convs_stack(image)
        x = nn.Flatten()(x)  # x = elt.Rearrange("b c h w -> b (c h w)")(x)  (elt是einops.layers.torch)
        logits = self.logits_layer(x)
        return logits
