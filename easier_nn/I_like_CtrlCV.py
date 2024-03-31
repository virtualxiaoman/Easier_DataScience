"""这里是方便复制粘贴的地方"""

"""
常用框架（以10分类为例）：
lr = 0.001  # 学习率
num_epochs = 500  # 迭代周期
batch_size = 10  # 每个小批量样本的数量
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(X_train.shape[1], 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
"""


"""
分类任务：
LeNet：
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

"""
