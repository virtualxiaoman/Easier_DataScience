from torch import nn
import torch

from easier_nn.classic_dataset import fashion_mnist
from easier_nn.train_net import train_net_with_evaluation
from easier_nn.evaluate_net import Timer

timer = Timer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fm = fashion_mnist()
fm.load_fashion_mnist(flatten=False)
# 将数据移动到 GPU 上
fm.X_train = fm.X_train.to(device)
fm.y_train = fm.y_train.to(device)
fm.X_test = fm.X_test.to(device)
fm.y_test = fm.y_test.to(device)

lr = 0.001  # 学习率
num_epochs = 3  # 迭代周期
batch_size = 64  # 每个小批量样本的数量

train_iter, test_iter = fm.load_dataiter(batch_size=batch_size)
train_iter = [(X.to(device), y.to(device)) for X, y in train_iter]
test_iter = [(X.to(device), y.to(device)) for X, y in test_iter]
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))
net.to(device)
loss = nn.CrossEntropyLoss()
loss.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

print(f'预处理数据{timer.stop():.5f} sec')
timer.start()
net, _, _, _ = train_net_with_evaluation(fm.X_train, fm.y_train, fm.X_test, fm.y_test, data_iter=train_iter,
                                         test_iter=test_iter, net=net, loss=loss, optimizer=optimizer,
                                         num_epochs=num_epochs, show_interval=2, draw='acc', if_immediately=False)
fm.predict(net, test_iter, n=18, num_rows=3, num_cols=6)
print(f'模型训练{timer.stop():.5f} sec')
