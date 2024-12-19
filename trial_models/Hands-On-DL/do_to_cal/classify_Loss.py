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
bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
loss = bce_with_logits_loss(x, y)
print(f'BCE With Logits Loss: {loss}')

# sigmoid值
print(F.sigmoid(torch.tensor([2.0, 1.0, 0.5, -1.0])))
print(F.sigmoid(torch.tensor([1.0, 2.0, -0.5, 0.5])))
print(F.sigmoid(torch.tensor([-0.5, 1.0, 1.5, 2.0])))

print("-" * 20)

x = torch.tensor([[0.8808, 0.7311],
                  [0.7311, 0.8808],
                  [0.3755, 0.7311]])
y = torch.tensor([[1, 0],
                  [0, 1],
                  [0, 1]], dtype=torch.float)
bce_loss = nn.BCELoss()
loss = bce_loss(x, y)
print(f'BCE Loss: {loss}')
print(nn.BCELoss(reduction='none')(x, y))

print("=" * 20)
print("一些常见的错误用法")
try:
    # 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
    x = torch.tensor([[2.0, 1.0, 0.5, -1.0],
                      [1.0, 2.0, -0.5, 0.5],
                      [-0.5, 1.0, 1.5, 2.0]], dtype=torch.float16)
    y = torch.tensor([0, 1, 3], dtype=torch.int64)  # 类别索引
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(x, y)
    print("CE OK!")
except Exception as e:
    print(f'CrossEntropyLoss Error: {e}')

try:
    # 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
    x = torch.tensor([[2.0, 1.0, 0.5, -1.0],
                      [1.0, 2.0, -0.5, 0.5],
                      [-0.5, 1.0, 1.5, 2.0]], dtype=torch.float64)
    # 目标值（每个类别的二进制标签，类别还是0, 1, 3）
    y = torch.tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=torch.int32)
    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce_with_logits_loss(x, y)
    print("BCE With Logits Loss OK!")
except Exception as e:
    print(f'BCEWithLogitsLoss Error: {e}')

try:
    # 原始预测值logits(未经激活函数)，假设每行是一个样本，每列是一个类别
    x = torch.tensor([[2, 0.1, 0.5, 0.6],
                      [0.12, 0.02, 0.5, 0.5],
                      [0.15, 0.8, 0.5, 0.20]], dtype=torch.float64)
    # 目标值（每个类别的二进制标签，类别还是0, 1, 3）
    y = torch.tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=torch.float64)
    bce_loss = nn.BCELoss(reduction='none')
    loss = bce_loss(x, y)
    print("BCE Loss OK!")
except Exception as e:
    print(f'BCELoss Error: {e}')
