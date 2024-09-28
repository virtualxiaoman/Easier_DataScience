import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

from easier_nn.train_net import NetTrainer

time_step = int(1440/15)  # 时间步长，就是利用多少时间窗口
batch_size = 32  # 批次大小
input_dim = 1  # 每个步长对应的特征数量，就是使用每天的4个特征，最高、最低、开盘、落盘
# hidden_dim = 64  # 隐层大小
output_dim = 1  # 由于是回归任务，最终输出层大小为1
# num_layers = 3  # BiGRU的层数
epochs = 30
best_loss = 0
# model_name = 'BiGRU'
save_path = './model/demo/forecasting/power_load.pth'

# 1. 加载2009-2015年某地区的电力数据 并 将数据进行标准化
# shape: (2201, 96)，表示2201天，每天96=1440/15个数据，即每15分钟采集一次数据
df = pd.read_excel('./input/power_load_data/power_load_data.xlsx', header=None)
df = pd.DataFrame(df.values.reshape(-1, 1))  # shape:(211296, 1), 即展平为2201*96行，1列
scaler = StandardScaler()
data = scaler.fit_transform(np.array(df))  # 标准化，shape: (211296, 1)


# 形成训练数据，例如time_step=3时[1,2,3,4,5]形成X:[[1,2,3],[2,3,4]]，Y:[4,5]
def split_data(data, timestep, train_rate=0.8):
    dataX = []  # 保存X
    dataY = []  # 保存Y
    # dataX存放的是前timestep个数据，dataY存放的是第timestep+1个数据，表示通过前timestep个数据预测第timestep+1个数据
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])
        dataY.append(data[index + timestep][0])
    dataX = np.array(dataX)  # (N, timestep, 1), N=len(data)-timestep
    dataY = np.array(dataY)  # (N, )

    # 划分训练集、测试集
    train_size = int(np.round(train_rate * dataX.shape[0]))
    x_train = dataX[: train_size, :].reshape(-1, timestep, 1)  # (N, timestep, 1)，N=N*train_rate
    y_train = dataY[: train_size].reshape(-1, 1)  # (N, 1)
    x_test = dataX[train_size:, :].reshape(-1, timestep, 1)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]


# x_train: (169034, 4, 1) y_train: (169034, 1) x_test: (42258, 4, 1) y_test: (42258, 1)，其中169034=(211296-4)*0.8
x_train, y_train, x_test, y_test = split_data(data, time_step)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)


# 定义一维卷积模块
class CNN(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 50, 1)
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=100)
        self.conv2 = nn.Conv1d(50, 100, 1)
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=50)
        self.fc = nn.Linear(50 * 100, output_dim)

    def forward(self, x):
        # 输入形状：32, 180 批次，序列长度
        #         x = x.transpose(1, 2) # 32, 16, 180 批次，词嵌入长度，序列长度

        x = self.conv1(x)  # 32, 50, 178
        x = self.maxpool1(x)  # 32, 50, 100
        x = self.conv2(x)  # 32, 100, 176
        x = self.maxpool2(x)  # 32, 100, 50

        x = x.reshape(-1, x.shape[1] * x.shape[2])  # 32, 100*50
        x = self.fc(x)  # 32, 2

        return x


model = CNN(output_dim=output_dim, input_dim=time_step)
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

net_trainer = NetTrainer(train_loader, test_loader, model, loss_function, optimizer, epochs=epochs, eval_type="loss",
                         batch_size=batch_size, eval_interval=1, eval_during_training=True)
net_trainer.view_parameters(view_params_details=False)
net_trainer.train_net(net_save_path=save_path)
loss_value = net_trainer.evaluate_net(delete_train=True)
print(f"Loss: {loss_value}")

# # 8.模型训练
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0
#     train_bar = tqdm(train_loader)  # 形成进度条
#     for data in train_bar:
#         x_train, y_train = data  # 解包迭代器中的X和Y
#         optimizer.zero_grad()
#         y_train_pred = model(x_train)
#         loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
#     exit(1)
#
#     # 模型验证
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         test_bar = tqdm(test_loader)
#         for data in test_bar:
#             x_test, y_test = data
#             y_test_pred = model(x_test)
#             test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
#
#     if test_loss < best_loss:
#         best_loss = test_loss
#         torch.save(model.state_dict(), save_path)

print('Finished Training')

model = net_trainer.net
# 9.绘制结果
plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform((model(x_train_tensor.to(net_trainer.device)).detach().numpy()).reshape(-1, 1)), "b",
#          label="predict")
# plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)), "r", label="real")
# plt.legend()
# plt.show()

test_length = 100  # 只选择100个数据进行预测，因为数据量太大，绘图不清晰，而且还可能out of memory
y_test_pred = model(x_test_tensor[:test_length].to(net_trainer.device)).detach().cpu().numpy()
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred), "b", label="predict")
plt.plot(scaler.inverse_transform(y_test_tensor[:test_length]), "r", label="real")
plt.legend()
plt.show()
