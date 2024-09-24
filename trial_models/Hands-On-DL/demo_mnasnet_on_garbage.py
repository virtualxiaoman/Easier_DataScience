import os
from PIL import Image
import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

from easier_nn.train_net import NetTrainer

epochs = 30
batch_size = 32
image_path = './input/garbage_data/data'
save_path = './model/demo/fine_tuning/garbage.pth'


# 1. 数据转换
data_transform = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1), ratio=(0.5, 2)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差归一化
    ]),
    # 测试集的数据归一化
    'test': transforms.Compose([
        transforms.Resize((224, 224)),  # 缩放
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 2.形成训练集和测试集 与 迭代器
dataset = datasets.ImageFolder(root=os.path.join(image_path), transform=data_transform['train'])  # 1303
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size  # Subset，是torch.utils.data.Dataset的子类，用于划分数据集，不包含原始数据集的所有属性
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 1042
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 261


# 3. 加载预训练好的MnasNet模型
model = torchvision.models.mnasnet1_0(weights=torchvision.models.MNASNet1_0_Weights.IMAGENET1K_V1)
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
# 修改最后一层的全连接层
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 模型训练
net_trainer = NetTrainer(train_loader, test_loader, model, criterion, optimizer, epochs=epochs, eval_type="acc",
                         batch_size=batch_size, eval_interval=1, eval_during_training=True)
net_trainer.view_parameters(view_params_details=False)
net_trainer.train_net(net_save_path=save_path)

# 5. 对某张图片进行显式的预测以查看效果
img_path = r'./input/garbage_data/test/shoes1750.jpg'
img = Image.open(img_path)  # PIL.JpegImagePlugin.JpegImageFile
img = data_transform['test'](img)  # (C, H, W)
img = torch.unsqueeze(img, dim=0).to(net_trainer.device)  # 将图像升维，增加batch_size维度
# 建立分类标签与索引的关系
class_to_idx = dataset.class_to_idx  # {'roses': 0, 'sunflowers': 1}
idx_to_class = {}
for key, val in class_to_idx.items():
    idx_to_class[val] = key  # 将class_to_idx变为idx_to_class，即{0: 'roses', 1: 'sunflowers'}
print(idx_to_class)
pred = net_trainer.net(img)  # 预测结果
print(f"Prediction: {pred}")
img_pred = idx_to_class[pred.argmax(dim=1).item()]
print(f"Prediction: {img_pred}")
