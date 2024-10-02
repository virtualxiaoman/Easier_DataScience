import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor(([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
                                              [[[0.1, -0.1, 0.1], [0.2, -0.2, 0.2], [0.3, -0.3, 0.3]]]])))  # (2, 1, 3, 3)
    conv.bias = nn.Parameter(torch.tensor([1.0, -1.0]))  # (2)
x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # (1, 1, 3, 3)
y = conv(x)  # (1, 2, 1, 1)
# print(x.shape, y.shape)
print(y)
# print(conv.weight.shape)
# print(conv.bias.shape)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]))
    conv.bias = nn.Parameter(torch.tensor([1.0]))
x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # (1, 1, 3, 3)
y = conv(x)  # (1, 1, 1, 1)
print(y)
print(conv.weight.shape)
print(conv.bias.shape)

print("---")

conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor([[[0.1, 0.2, 0.3]]]))
    conv.bias = nn.Parameter(torch.tensor([1.0]))
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # (1, 1, 5)
y = conv(x)  # (1, 1, 3)
print(y)
print(conv.weight.shape)
print(conv.bias.shape)

print("===")
conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                                               [[0.1, -0.1, 0.1], [0.2, -0.2, 0.2], [0.3, -0.3, 0.3]]]]))  # (1, 2, 3, 3)
    conv.bias = nn.Parameter(torch.tensor([1.0]))  # (1)
x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # (1, 2, 3, 3)
y = conv(x)  # (1, 1, 1, 1)
print(x.shape, y.shape)
print(y)  # tensor([[[[33.1000]]]], grad_fn=<ConvolutionBackward0>)
print(conv.weight.shape)
print(conv.bias.shape)
print(conv.weight)
print(conv.bias)
