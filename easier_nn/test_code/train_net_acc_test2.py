import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from easier_nn.train_net import NetTrainerFNN

print("这是多分类测试代码")
# 生成多分类数据
np.random.seed(42)
data = np.random.rand(1000, 20)  # 20个特征
target = np.sum(data, axis=1)  # 计算特征和

# 根据特征和划分为4个类别
bins = [0, 5, 10, 15, np.inf]
target = np.digitize(target, bins) - 1  # 将类别映射为0-3

# 转换为torch tensor
data = torch.tensor(data, dtype=torch.float)
target = torch.tensor(target, dtype=torch.long)

# 划分为训练集和测试集
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# 转化为train_loader和test_loader
train_dataset = TensorDataset(data_train, target_train)
test_dataset = TensorDataset(data_test, target_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class MultiClassificationNet(nn.Module):
    def __init__(self):
        super(MultiClassificationNet, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 4)  # 4个类别

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义模型、损失函数和优化器
net = MultiClassificationNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
trainer = NetTrainerFNN(train_loader, test_loader, net, loss_fn, optimizer,
                        epochs=50, eval_interval=5)
trainer.train_net()
test_acc = trainer.evaluate_net(eval_type="test")

print(trainer.test_acc_list)
print(f"Test accuracy: {test_acc}")
