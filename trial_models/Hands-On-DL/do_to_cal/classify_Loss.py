import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
x = torch.tensor([[2.0, 1.0, 0.5, -1.0],
                  [1.0, 2.0, -0.5, 0.5],
                  [-0.5, 1.0, 1.5, 2.0]])
y = torch.tensor([0, 1, 3])  # 类别索引
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(x, y)
print(f'Cross Entropy Loss: {loss.item()}')
print(nn.CrossEntropyLoss(reduction='none')(x, y))

print(F.softmax(torch.tensor([2.0, 1.0, 0.5, -1.0]), dim=0))
print(F.softmax(torch.tensor([1.0, 2.0, -0.5, 0.5]), dim=0))
print(F.softmax(torch.tensor([-0.5, 1.0, 1.5, 2.0]), dim=0))

print("-" * 20)

# 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
x = torch.tensor([[2.0, 1.0],
                  [1.0, 2.0],
                  [-0.5, 1.0]])
# 目标值（每个类别的二进制标签）
y = torch.tensor([[1, 0],
                  [0, 1],
                  [0, 1]], dtype=torch.float)
# 定义 BCEWithLogitsLoss
bce_with_logits_loss = nn.BCEWithLogitsLoss()
# 计算损失
loss = bce_with_logits_loss(x, y)
print(f'BCE With Logits Loss: {loss}')
print(nn.BCEWithLogitsLoss(reduction='none')(x, y))
print(F.sigmoid(torch.tensor([2.0, 1.0])))
print(F.sigmoid(torch.tensor([1.0, 2.0])))
print(F.sigmoid(torch.tensor([-0.5, 1.0])))

print("-" * 20)

# 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
x = torch.tensor([[2.0, 1.0, 0.5, -1.0],
                  [1.0, 2.0, -0.5, 0.5],
                  [-0.5, 1.0, 1.5, 2.0]])
# 目标值（每个类别的二进制标签，类别还是0, 1, 3）
y = torch.tensor([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=torch.float)
bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
loss = bce_with_logits_loss(x, y)
print(f'BCE With Logits Loss: {loss}')
