import torch
from torch import nn
from torch.nn import functional as F


# LeNet5，输入为1*28*28的图像
class LeNet5(nn.Module):
    def __init__(self):  # 初始化函数
        super(LeNet5, self).__init__()  # 多基层一般使用super
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.09)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # input(1,28,28),output1(6,24,24) output2(6,12,12)
        x = self.pool2(F.relu(self.conv2(x)))  # input(6,12,12),output1(16,8,8) output2(16,4,4)
        x = x.view(-1, 16 * 4 * 4)  # -1第一个维度
        x = self.drop(F.relu(self.fc1(x)))  # 全连接层1及其激活函数
        x = self.drop(F.relu(self.fc2(x)))  # 全连接层3得到输出
        x = self.drop(self.fc3(x))
        return x


# ResNet，代码由GPT生成
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet，代码由GPT生成
class ResNet(nn.Module):
    """net = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)  # ResNet18"""

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)  # *layers表示将layers中的元素展开

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
