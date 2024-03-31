from sklearn.model_selection import train_test_split
from torch import nn
import torch

from easier_nn.classic_dataset import load_mnist
from easier_nn.load_data import trainset_to_dataloader, testset_to_dataloader
from easier_nn.train_net import train_net, train_net_with_evaluation

X, y = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train = X_train.float()
X_test = X_test.float()

lr = 0.001  # 学习率
num_epochs = 100  # 迭代周期
batch_size = 10  # 每个小批量样本的数量
train_iter = trainset_to_dataloader(X_train, y_train, batch_size=batch_size)
test_iter = testset_to_dataloader(X_test, y_test, batch_size=batch_size)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(X_train.shape[1], 10),
    nn.Softmax(dim=1)
)
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.fill_(0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# net = train_net(X_train, y_train, data_iter=train_iter, net=net, loss=loss, optimizer=optimizer, lr=lr,
#                 num_epochs=num_epochs, show_interval=10)
net, _, _, _ = train_net_with_evaluation(X_train, y_train, X_test, y_test, data_iter=train_iter, test_iter=test_iter,
                                         net=net, loss=loss, optimizer=optimizer, lr=lr, num_epochs=num_epochs,
                                         show_interval=2, draw='acc')
