from torch import nn
import torch

from easier_nn.classic_dataset import fashion_mnist
from easier_nn.train_net import train_net_with_evaluation

fm = fashion_mnist()
fm.load_fashion_mnist()
train_iter, test_iter = fm.load_dataiter()

lr = 0.001  # 学习率
num_epochs = 10  # 迭代周期
batch_size = 10  # 每个小批量样本的数量
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(fm.X_train.shape[1], 10),
    nn.Softmax(dim=1)
)
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.fill_(0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

net, _, _, _ = train_net_with_evaluation(fm.X_train, fm.y_train, fm.X_test, fm.y_test, data_iter=train_iter,
                                         test_iter=test_iter, net=net, loss=loss, optimizer=optimizer, lr=lr,
                                         num_epochs=num_epochs, show_interval=2, draw='acc')
fm.predict(net, test_iter, n=18, num_rows=3, num_cols=6)
