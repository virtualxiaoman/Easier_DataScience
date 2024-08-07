# 使用在ImageNet数据集上预训练的ResNet‐18作为源模型，迁移学习到一个自定义的小型数据集上

import torch
from torch import nn
import torchvision
import os
from easier_excel.read_data import show_images
from easier_nn.train_net import NetTrainer
DATA_DIR = '../data/hotdog'

# 1. 读取数据集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'test'))
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
show_images(hotdogs + not_hotdogs, 2, 8)

# 2. 数据增强augmentation
# 使用RGB通道的均值和标准差，以标准化每个通道，这个数值是在ImageNet数据集上计算得到的。
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪图像，然后缩放为224x224
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    torchvision.transforms.ToTensor(),
    normalize])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),  # 缩放图像为256x256
    torchvision.transforms.CenterCrop(224),  # 从图像中心裁剪出224x224大小的图像
    torchvision.transforms.ToTensor(),
    normalize])

# 3. 预训练模型
weights = torchvision.models.ResNet18_Weights.DEFAULT
finetune_net = torchvision.models.resnet18(weights=weights)  # pretrained is deprecated，所以改为使用weights
print(finetune_net.fc)  # 最后的全连接层：Linear(in_features=512, out_features=1000, bias=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 修改最后一层为线性层，输出2类
nn.init.xavier_uniform_(finetune_net.fc.weight)  # 初始化最后一层的权重


# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_augs),
        batch_size=batch_size)
    loss = nn.CrossEntropyLoss()
    if param_group:
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD(
            [{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    net_trainer = NetTrainer(train_iter, test_iter, net, loss, trainer, epochs=num_epochs, eval_type="acc",
                             batch_size=batch_size, print_interval=1, eval_during_training=True)
    net_trainer.view_parameters(view_params_details=False)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


train_fine_tuning(finetune_net, 5e-5, batch_size=64)  # 微调模型，只在最后一层使用较大的学习率
# train_fine_tuning(finetune_net, 5e-5, batch_size=64, param_group=False)  # 所有参数都使用同一个学习率
# 可以看出，微调模型的精度与训练loss的下降速度都比从头训练的模型要好得多。
