import numpy as np
import torch
import torch.nn as nn

# # 假设输入 X 是一个形状为 (batch_size, channels, height, width) 的张量
# batch_size, channels, height, width = 16, 3, 30, 30
# X = torch.randn(batch_size, channels, height, width)
#
# # 定义网络变换操作
# net = nn.Sequential(
#     nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.BatchNorm2d(16),
#     nn.Flatten(),
#     nn.Linear(16 * (height//2) * (width//2), 64),
#     nn.ReLU(),
#     nn.Linear(64, 10),
#     nn.Softmax(dim=1)
# )
#
# # 对输入 X 进行变换操作并打印形状
# for layer in net:
#     # print(X)
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)
# # print(X)
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_val = np.exp(x)
    softmax_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    return softmax_val

# 生成随机矩阵作为输入
x = np.random.rand(100, 10)

# 使用 softmax 函数进行转换
softmax_output = softmax(x)

# 可视化 softmax 输出
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.hist(softmax_output[:, i], bins=20)
    plt.title(f"Class {i+1}")
plt.tight_layout()
plt.show()
"""
在上面的代码中，展示的图像中：

横坐标代表 softmax 输出的值，可以理解为概率值。
纵坐标代表对应概率值的样本数量。
具体来说，每个柱子的高度表示了在该概率范围内的样本数量。例如，如果某个类别的柱子在横坐标值 0.6 处有一个较高的柱子，意味着有很多样本被分类为这个类别的概率约为 0.6。

这种图像通常被用来观察模型的分类倾向性，即模型对于不同类别的预测概率分布情况。
"""