import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from easier_nn.train_net import NetTrainerFNN

# 生成分类数据
np.random.seed(42)
data = np.random.rand(1000, 20)  # 20个特征
target = (np.sum(data, axis=1) > 10).astype(int)  # 如果特征和大于10，则类别为1，否则为0

# 将data和target划分训练集和测试集
data = torch.tensor(data, dtype=torch.float)
target = torch.tensor(target, dtype=torch.long)
# 划分为训练集和测试集
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# 转化为train_loader和test_loader
train_dataset = TensorDataset(data_train, target_train)
test_dataset = TensorDataset(data_test, target_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)  # 2个类别

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = ClassificationNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
trainer = NetTrainerFNN(train_loader, test_loader, net, loss_fn, optimizer,
                        epochs=50, eval_type="acc", eval_interval=5)
trainer.train_net()
test_acc = trainer.evaluate_net(eval_type="test")

print(trainer.test_acc_list)
print(f"Test accuracy: {test_acc}")
