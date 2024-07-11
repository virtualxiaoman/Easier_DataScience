# Hands On bilibili Recommend System 动手实现b站的推荐系统(非官方)
# 这是第三部分，主要是使用神经网络

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from easier_excel import read_data, cal_data, draw_data
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.to_md import ToMd

ToMd.path = "output/bilibili_RS/Bili_RS_3.md"  # 更改输出路径
ToMd = ToMd()
ToMd.update_path()  # 这是更改path需要做的必要更新
ToMd.clear_md()  # 清空前如有需要，务必备份

model_path = "output/bilibili_RS/model"

# 设置pandas显示选项
read_data.set_pd_option(max_show=True, float_type=True, decimal_places=4)

print(CT("----------读取数据----------").pink())
path = "input/history_xm.xlsx"
df_origin = read_data.read_df("input/history_xm.xlsx")

print(CT("----------增加新列----------").pink())
# view_percent这一列是以百分比形式展示的，需要转换为数值型数据
df_origin['view_percent'] = df_origin['view_percent'].str.rstrip('%').astype('float') / 100.0
# 将弹幕、评论、点赞、投币、收藏、分享这六列的数据转化为比例
df_origin['dm_rate'] = df_origin['dm'] / df_origin['view']
df_origin['reply_rate'] = df_origin['reply'] / df_origin['view']
df_origin['like_rate'] = df_origin['like'] / df_origin['view']
df_origin['coin_rate'] = df_origin['coin'] / df_origin['view']
df_origin['fav_rate'] = df_origin['fav'] / df_origin['view']
df_origin['share_rate'] = df_origin['share'] / df_origin['view']
# time这一列是时间戳，数值较大。这里简单处理：减去最小值，使时间戳从0开始
df_origin['time'] = df_origin['time'] - df_origin['time'].min()

print(CT("----------数据处理----------").pink())
df_origin = df_origin[df_origin['like_rate'] <= 1]   # 顺带删除缺失值
# 将u_score二值化，u_score>=3的为1，否则为0
df_origin['u_score'] = df_origin['u_score'].apply(lambda x: 1 if x >= 3 else 0)
df_origin = df_origin.select_dtypes(include=['number'])  # 数值型数据


# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        residual = self.layer1(residual)  # 将残差也经过一层线性变换，以确保维度匹配
        x = x + residual  # 残差连接
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = ResidualBlock(input_size, 512)
        self.layer2 = ResidualBlock(512, 256)
        self.layer3 = ResidualBlock(256, 128)
        self.layer4 = ResidualBlock(128, 64)
        self.layer5 = ResidualBlock(64, 32)
        self.layer6 = nn.Linear(32, num_classes)
        self.bn_final = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.relu(self.bn_final(x))
        x = self.dropout(x)
        x = self.layer6(x)
        return x

X = df_origin.drop(['u_like', 'u_coin', 'u_fav', 'u_score', 'progress', 'duration', 'view_percent', 'view_time'], axis=1)
y = df_origin['u_score'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = pd.DataFrame(y_train, columns=['u_score'])
X_train.reset_index(drop=True, inplace=True)  # 重置索引，确保索引对齐
y_train.reset_index(drop=True, inplace=True)
dataset_df = pd.concat([X_train, y_train], axis=1)

desc_df = read_data.desc_df(dataset_df)
desc_df.transform_df(target='u_score')
dataset_smotetomek = desc_df.smotetomek_df
X_train = dataset_smotetomek.drop(['u_score'], axis=1)
y_train = dataset_smotetomek['u_score']

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)
# 定义模型、损失函数和优化器
input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model = NeuralNetwork(input_size, num_classes).to(device)
# 2分类问题，设置权重为[2.0, 3.0]
criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 3.0]).to(device))
# # 6分类问题，设置权重为[1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0, 10.0, 20.0, 30.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 800
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# # 保存模型
# torch.save(model, f"{model_path}/nn_resnet_2class_800epoch.pth")

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    _, predicted = torch.max(outputs.data, 1)
    acc = accuracy_score(y_train.cpu(), predicted.cpu())
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_train.cpu(), predicted.cpu()))
    print('Confusion Matrix:')
    print(confusion_matrix(y_train.cpu(), predicted.cpu()))

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    acc = accuracy_score(y_test.cpu(), predicted.cpu())
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test.cpu(), predicted.cpu()))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test.cpu(), predicted.cpu()))
