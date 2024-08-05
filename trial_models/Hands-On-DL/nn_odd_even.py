import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from easier_nn.train_net import NetTrainer

# # 创建数据集，第一列是一个随机正整数，第二个是标签，标签是奇数还是偶数(0表示偶数，1表示奇数)
# def create_data(num_samples=10000):
#     X = np.random.randint(100000, 1000000, size=(num_samples, 1))
#     y = (X % 2 == 1).astype(np.int32)
#     df_Xy = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['X', 'y'])
#     return df_Xy
#
#
# df = create_data()
# print(df.head())
# # 保存为nn_odd_even.csv
# df.to_csv("input/nn_odd_even.csv", index=False)

df = pd.read_csv("input/nn_odd_even.csv")
print(df.head())
print(df.shape)

# X是一个正整数，将每一位当做一个特征。y不变
X = df['X'].values
X = np.array([list(str(x)) for x in X])
X = X.astype(np.float32)
print(X.shape)
y = df['y'].values.astype(np.int64)

# Accuracy: 0.5345，并且始终不收敛
# net = torch.nn.Sequential(
#     torch.nn.Linear(X.shape[1], 2),
# )

# Accuracy: 0.797，epoch似乎可以再多一点
# net = nn.Sequential(
#     nn.Linear(X.shape[1], 64),
#     nn.ReLU(),
#     nn.Linear(64, 2),
# )

# Accuracy: 1，在epoch=180左右收敛
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#         self.linear1 = nn.Linear(in_features, in_features)
#         self.linear2 = nn.Linear(in_features, in_features)
#     def forward(self, x):
#         residual = x
#         out = F.relu(self.linear1(x))
#         out = self.linear2(out)
#         out += residual
#         return F.relu(out)
# class ResNet(nn.Module):
#     def __init__(self, input_size):
#         super(ResNet, self).__init__()
#         self.layer1 = nn.Linear(input_size, 64)
#         self.resblock = ResidualBlock(64)
#         self.output = nn.Linear(64, 2)
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = self.resblock(x)
#         x = self.output(x)
#         return x
# net = ResNet(X.shape[1])

# Accuracy: 1，在epoch=40左右收敛
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (input_size // 2), 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * (x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
X = X.reshape(X.shape[0], 1, X.shape[1])
net = CNN(X.shape[2])

# ACC: 0.9097782258064516, epoch似乎可以再多一点，而且训练过程不够稳定，训练时间也明显比之前的久
# class RNN(nn.Module):
#     # 包含PyTorch的GRU和拼接的MLP
#     def __init__(self, input_size, output_size, hidden_size):
#         super().__init__()
#         self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden):
#         # 传入的x的维度为(batch_size, seq_len, input_size)
#         # GRU接受的输入为(seq_len, batch_size, input_size)
#         # 因此需要使用transpose函数交换x的坐标轴
#         # out的维度是(seq_len, batch_size, hidden_size)
#         out, hidden = self.gru(torch.transpose(x, 0, 1), hidden)
#         # 取序列最后的中间变量输入给全连接层
#         out = self.linear(out[-1])
#         return out, hidden
# # 需要将数据 reshape 为 (batch_size, seq_len, input_size)
# X = X.reshape(-1, X.shape[1], 1)
# net = RNN(1, 2, 16)

# ACC: 0.647，并且始终不收敛，感觉我代码写错了。。然后最后predict_odd_even(123456)和predict_odd_even(123457)又是对的。
# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, nhead, hidden_dim, num_layers, output_dim):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(input_dim, hidden_dim)
#         self.transformer = nn.Transformer(hidden_dim, nhead, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#     def forward(self, src):
#         src = self.embedding(src.long())  # Embedding
#         output = self.transformer(src, src)  # Transformer
#         output = output.mean(dim=1)  # 取时间维度的平均值，即数字的每一位
#         output = self.fc(output)  # 全连接层
#         return output
# net = TransformerModel(input_dim=10, nhead=2, hidden_dim=16, num_layers=2, output_dim=2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

trainer = NetTrainer(X, y, net, loss_fn, optimizer, eval_type="acc", epochs=500, batch_size=64)
# # RNN模型需要显式设置rnn的参数，如下所示：
# trainer = NetTrainer(X, y, net, loss_fn, optimizer, eval_type="acc", epochs=500, batch_size=64,
#                      rnn_input_size=1, rnn_seq_len=X.shape[1], rnn_hidden_size=16)
trainer.view_parameters()
trainer.train_net()
acc = trainer.evaluate_net(delete_train=True)  # delete_train=True表示从cuda上删除训练集，只保留测试集
print(f"Accuracy: {acc}")


# 请注意：如果上面的X变换了，这里也要变换，否则会报错
# （为什么不使用NetTrainer的evaluate_net方法？此接口正在完善中，避免bug太多了）
def predict_odd_even(num):
    X = np.array([list(str(num))]).astype(np.float32)
    y_pred = net(torch.tensor(X).to(trainer.device)).cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


print(predict_odd_even(123456))
print(predict_odd_even(123457))


