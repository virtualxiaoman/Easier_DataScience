# © virtual小满 2024-09-23
# 以下代码是在花朵数据集上进行微调的示例代码
# keywords: 数据转换、模型微调、预训练模型、迁移学习、resnet152

import os
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from PIL import Image

from easier_nn.train_net import NetTrainer

epochs = 5
batch_size = 32
image_path = './input/flowerdata/data'
save_path = './model/demo/fine_tuning/flower_model.pth'

# 1. 数据转换
data_transform = {
    # 训练集数据增强、归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差归一化
    ]),
    # 验证集不增强，仅进行归一化
    'val/test': transforms.Compose([
        transforms.Resize((224, 224)),  # 缩放
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. 形成训练集验证集 与 迭代器
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform['train'])
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform['val/test'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, True)

# 3. 加载resnet模型
model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # 冻结模型参数
model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层的全连接层
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 优化器

# 4. 模型训练
net_trainer = NetTrainer(train_loader, val_loader, model, criterion, optimizer, epochs=epochs, eval_type="acc",
                         batch_size=batch_size, eval_interval=1, eval_during_training=True)
net_trainer.view_parameters(view_params_details=False)
net_trainer.train_net(net_save_path=save_path)
acc = net_trainer.evaluate_net(delete_train=True)
print(f"Accuracy: {acc}")

# 5. 对某张图片进行显式的预测以查看效果
img_path = r'./input/flowerdata/test/test01.jpg'
img = Image.open(img_path)  # (C, H, W)
img = data_transform['val/test'](img)
img = torch.unsqueeze(img, dim=0).to(net_trainer.device)  # 将图像升维，增加batch_size维度
class_to_idx = train_dataset.class_to_idx  # {'roses': 0, 'sunflowers': 1}
idx_to_class = {}
for key, val in class_to_idx.items():
    idx_to_class[val] = key  # 将class_to_idx变为idx_to_class，即{0: 'roses', 1: 'sunflowers'}
pred = net_trainer.net(img)  # 预测结果
print(f"Prediction: {pred}")
img_pred = idx_to_class[pred.argmax(dim=1).item()]
print(f"Prediction: {img_pred}")
