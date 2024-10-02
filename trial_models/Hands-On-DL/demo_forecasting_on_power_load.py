# © virtual小满 2024-09-30
# 以下代码是通过CNN-1D模型进行电力负荷预测的示例代码，，每次输入的时间步是96个（往前看96个），预测时间步是1个（只向后预测1个），
# 也就是不会将预测数据用于下一次的预测，每次预测的数据均为真实数据。
# keywords: 时序预测、CNN、电力负荷预测、训练数据生成

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from easier_nn.train_net import NetTrainer


time_step = int(1440/15)  # 时间步长，这里按一天来算(1440分钟÷15分钟/天=96个数据/天)
batch_size = 32
output_dim = 1  # 由于是回归任务，最终输出层大小为1
epochs = 5
save_path = './model/demo/forecasting/power_load.pth'

# 1. 加载2009-2015年某地区的电力数据 并 将数据进行标准化
# shape: (2201, 96)，表示2201天，每天96=1440/15个数据，即每15分钟采集一次数据
df = pd.read_excel('./input/power_load_data/power_load_data.xlsx', header=None)  # shape: (2201, 96)
df = pd.DataFrame(df.values.reshape(-1, 1))  # shape:(211296, 1), 即展平为2201*96行，1列
scaler = StandardScaler()
data = scaler.fit_transform(np.array(df))  # 标准化，shape: (211296, 1)


# 2. 形成训练数据，例如time_step=3时[1,2,3,4,5]形成X:[[1,2,3],[2,3,4]]，Y:[4,5]
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

x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)


class CNN(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 50, 1)
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=100)
        self.conv2 = nn.Conv1d(50, 100, 1)
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=50)
        self.fc = nn.Linear(50 * 100, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.fc(x)
        return x


# 8. 训练模型
model = CNN(output_dim=output_dim, input_dim=time_step)
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

net_trainer = NetTrainer(train_loader, test_loader, model, loss_function, optimizer, epochs=epochs, eval_type="loss",
                         batch_size=batch_size, eval_interval=1, eval_during_training=True)
net_trainer.view_parameters(view_params_details=False)
net_trainer.train_net(net_save_path=save_path)
loss_value = net_trainer.evaluate_net(delete_train=True)
print(f"Loss: {loss_value}")

# 4. 绘制结果
plt.figure(figsize=(12, 8))
test_length = 100  # 只选择100个数据进行预测，因为数据量太大，绘图不清晰，而且还可能out of memory。并且这个demo只是一个玩具
y_test_pred = net_trainer.net(x_test_tensor[:test_length].to(net_trainer.device)).detach().cpu().numpy()
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred), "b", label="predict")
plt.plot(scaler.inverse_transform(y_test_tensor[:test_length]), "r", label="real")
plt.legend()
plt.show()
