import torch
import torch.nn as nn

# 创建输入张量
x = torch.tensor([[1.0, 2.0, -3.0], [4.0, 5.0, -6.0]])
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
x_relu = relu(x)
print(x_relu)
x_sigmoid = sigmoid(x)
print(x_sigmoid)

