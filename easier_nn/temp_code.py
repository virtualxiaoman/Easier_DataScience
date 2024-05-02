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
# import numpy as np
# import matplotlib.pyplot as plt
#
# def softmax(x):
#     exp_val = np.exp(x)
#     softmax_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
#     return softmax_val
#
# # 生成随机矩阵作为输入
# x = np.random.rand(100, 10)
#
# # 使用 softmax 函数进行转换
# softmax_output = softmax(x)
#
# # 可视化 softmax 输出
# plt.figure(figsize=(10, 6))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.hist(softmax_output[:, i], bins=20)
#     plt.title(f"Class {i+1}")
# plt.tight_layout()
# plt.show()
"""
在上面的代码中，展示的图像中：

横坐标代表 softmax 输出的值，可以理解为概率值。
纵坐标代表对应概率值的样本数量。
具体来说，每个柱子的高度表示了在该概率范围内的样本数量。例如，如果某个类别的柱子在横坐标值 0.6 处有一个较高的柱子，意味着有很多样本被分类为这个类别的概率约为 0.6。

这种图像通常被用来观察模型的分类倾向性，即模型对于不同类别的预测概率分布情况。
"""
# import numpy as np
#
# # 定义矩阵 T
# T = np.array([[0.7, 0.2],
#               [0.3, 0.8]])
#
# # 求解特征值和特征向量
# eigenvalues, eigenvectors = np.linalg.eig(T)
#
# # 打印特征值和特征向量
# print("特征值:", eigenvalues)
# print("特征向量:\n", eigenvectors)
#
#
import torch
import torch.nn as nn
from easier_tools.print_variables import print_variables_class
from easier_nn.train_net import train_net

input_size = 123  # 输入数据编码的维度
output_size = 123  # 输出数据编码的维度
hidden_size = 20  # 隐含层维度
num_layers = 2  # 隐含层层数
seq_len = 1000  # 句子长度
batch_size = 64  # 批量大小

rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
linear = nn.Linear(hidden_size, output_size)
print(print_variables_class(rnn, specific_param=["_all_weights"]))
print("rnn:", rnn)
x = torch.randn(seq_len, batch_size, input_size)  # 输入数据
h0 = torch.zeros(num_layers, batch_size, hidden_size)  # 隐含层初始化
out, h = rnn(x, h0)  # 输出数据
print("out.shape:", out.shape)
print("h.shape:", h.shape)
out = linear(out)  # 线性变换
print("out.shape:", out.shape)




