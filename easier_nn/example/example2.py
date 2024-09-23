from torch import nn
import torch
from easier_nn.classic_dataset import fashion_mnist
from easier_nn.train_net import NetTrainer


fm = fashion_mnist()
fm.load_fashion_mnist(flatten=False)

num_epochs = 5  # 迭代周期
batch_size = 64  # 每个小批量样本的数量

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

trainer = NetTrainer((fm.X_train, fm.X_test), (fm.y_train, fm.y_test), net, loss, optimizer,
                     batch_size=batch_size, epochs=num_epochs, eval_type="acc", eval_interval=1)
trainer.train_net()

