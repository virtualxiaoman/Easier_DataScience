import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"train shape: {train.shape}, test shape: {test.shape}")

# 数据预处理：合并数据，处理类别变量，填充缺失值
all_data = pd.concat([train, test], ignore_index=True)
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
print(f"all_data shape: {all_data.shape}")

# 切分数据
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y_train = train['SalePrice']

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# KFold交叉验证设置
folds = KFold(n_splits=5, shuffle=True, random_state=42)


# 用于生成提交文件的函数
def generate_submission(test_preds, test, model_name):
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_preds
    })
    submission.to_csv(f"output/DNN/submission_{model_name}.csv", index=False)
    print("Submission file saved!")


# 1. 简单线性神经网络模型
class LinearNN(nn.Module):
    def __init__(self, input_dim):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 2. 卷积神经网络（CNN）模型
class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2240, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, features)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim):
        super(Transformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=2)
        self.fc1 = nn.Linear(4592, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, features)
        x = torch.relu(self.conv1(x))
        # 调整形状为 (features, batch_size, embed_dim) 适合 MultiheadAttention 的输入
        x = x.transpose(1, 2)  # 形状变为 (batch_size, 64, features) -> (batch_size, features, 64)
        x = x.transpose(0, 1)  # 形状变为 (features, batch_size, 64)
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)  # (features, batch_size, embed_dim) -> (batch_size, features, embed_dim)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练与评估过程
def train_and_evaluate(model_class, model_name, input_dim, lr=0.001, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_preds = np.zeros(X_train_tensor.shape[0])  # 存储每一折的验证集预测值
    test_preds = np.zeros(X_test_tensor.shape[0])  # 存储测试集的平均预测值

    for trn_idx, val_idx in folds.split(X_train_tensor, y_train_tensor):
        trn_df, trn_label = X_train_tensor[trn_idx], y_train_tensor[trn_idx]
        val_df, val_label = X_train_tensor[val_idx], y_train_tensor[val_idx]

        # DataLoader
        train_data = TensorDataset(trn_df, trn_label)
        val_data = TensorDataset(val_df, val_label)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # 创建并训练模型
        model = model_class(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # 评估模型
        model.eval()
        with torch.no_grad():
            val_df = val_df.to(device)
            val_preds = model(val_df).squeeze().cpu().numpy()
            oof_preds[val_idx] = val_preds
            test_preds += model(X_test_tensor.to(device)).squeeze().cpu().numpy() / folds.n_splits

        print(f"{model_name} - Fold RMSE: {np.sqrt(mean_squared_error(val_label, oof_preds[val_idx]))}")

    # 输出训练集上的RMSE评分
    rmse = np.sqrt(mean_squared_error(y_train_tensor, oof_preds))
    print(f'{model_name} - Overall RMSE on training data: {rmse:.4f}')

    # 生成提交文件
    generate_submission(test_preds, test, model_name)


# # 训练并生成线性神经网络提交文件
# train_and_evaluate(LinearNN, "Linear NN", X_train_tensor.shape[1], epochs=100)
#
# 训练并生成CNN模型提交文件
train_and_evaluate(CNN, "CNN", X_train_tensor.shape[1], epochs=200)

# # 训练并生成Transformer模型提交文件
# train_and_evaluate(Transformer, "Transformer", X_train_tensor.shape[1], epochs=100)
