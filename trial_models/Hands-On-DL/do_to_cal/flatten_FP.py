import torch
import torch.nn as nn

flatten = nn.Flatten()
x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])  # (2, 2, 3)
x_flat = flatten(x)  # (2, 6)
print(x_flat)
