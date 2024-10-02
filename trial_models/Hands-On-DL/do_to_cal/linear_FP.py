import torch
import torch.nn as nn

linear = nn.Linear(3, 4)
with torch.no_grad():
    linear.weight = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]))
    linear.bias = nn.Parameter(torch.tensor([2.0, 3.0, 4.0, 5.0]))
x = torch.tensor([[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]],
                  [[7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]],
                  [[13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0]]])
y = linear(x)
print(x.shape, y.shape)
print(y)
