import torch
import torch.nn as nn

x = torch.tensor([[0.5, 0.2], [0.7, 0.9]])
y = torch.tensor([[0.6, 0.4], [0.3, 0.5]])
loss = nn.MSELoss()(x, y)
print(f'MSE Loss: {loss.item()}')

loss = nn.MSELoss(reduction='none')(x, y)
print(f'MSE Loss: {loss}')

loss = nn.MSELoss(reduction='sum')(x, y)
print(f'MSE Loss: {loss}')

print("-" * 20)

loss = nn.L1Loss()(x, y)
print(f'L1 Loss: {loss.item()}')

loss = nn.L1Loss(reduction='none')(x, y)
print(f'L1 Loss: {loss}')

loss = nn.L1Loss(reduction='sum')(x, y)
print(f'L1 Loss: {loss}')

print("-" * 20)

loss = nn.HuberLoss()(x, y)
print(f'Huber Loss: {loss.item()}')

loss = nn.HuberLoss(delta=0.2)(x, y)
print(f'Huber Loss: {loss.item()}')

loss = nn.HuberLoss(delta=0.2, reduction='none')(x, y)
print(f'Huber Loss: {loss}')
