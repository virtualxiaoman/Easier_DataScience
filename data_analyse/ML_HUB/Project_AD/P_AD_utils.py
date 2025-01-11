# 注意，本py只适用于P_AD_8及之后的文件，前面的如果使用这个不能保证可以运行
import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# import numpy as np


# 数据加载与转换函数
def load_data(X_train, y_train, X_test, y_test, batch_size=64):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# import torch
#
#
# # 模型训练函数
# def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=10, device='cuda'):
#     # 将模型移动到CUDA设备
#     model.to(device)
#
#     for epoch in range(epochs):
#         # 训练模式
#         model.train()
#         train_preds, train_labels = [], []
#         total_loss = 0
#
#         for batch_x, batch_y in train_loader:
#             # 将数据也转移到CUDA设备
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#             # 前向传播
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             train_preds.extend(outputs.detach().cpu().numpy())  # 转回CPU后再转为numpy
#             train_labels.extend(batch_y.cpu().numpy())  # 转回CPU
#
#         # 计算训练指标
#         train_preds = np.round(train_preds)
#         train_acc = accuracy_score(train_labels, train_preds)
#         train_auc = roc_auc_score(train_labels, train_preds)
#
#         # 验证模式
#         model.eval()
#         val_preds, val_labels = [], []
#         with torch.no_grad():
#             for batch_x, batch_y in test_loader:
#                 # 将验证数据也转移到CUDA设备
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#                 # 前向传播
#                 outputs = model(batch_x)
#                 val_preds.extend(outputs.detach().cpu().numpy())  # 转回CPU
#                 val_labels.extend(batch_y.cpu().numpy())  # 转回CPU
#
#         # 计算验证指标
#         val_preds_rounded = np.round(val_preds)
#         val_acc = accuracy_score(val_labels, val_preds_rounded)
#         val_auc = roc_auc_score(val_labels, val_preds)
#
#         print(
#             f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, "
#             f"Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, "
#             f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}"
#         )
#
#
# # 模型评测函数
# def evaluate_model(model, test_loader, device='cuda'):
#     # 将模型移动到CUDA设备
#     model.to(device)
#
#     model.eval()
#     test_preds, test_labels = [], []
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             # 将数据转移到CUDA设备
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#             # 前向传播
#             outputs = model(batch_x)
#             test_preds.extend(outputs.cpu().numpy())  # 转回CPU
#             test_labels.extend(batch_y.cpu().numpy())  # 转回CPU
#
#     # 转换预测值
#     test_preds_binary = np.round(test_preds)
#
#     # 输出分类报告与 AUC
#     print("Test Classification Report:")
#     print(classification_report(test_labels, test_preds_binary, digits=4))
#
#     test_auc = roc_auc_score(test_labels, test_preds)
#     print(f"Test AUC: {test_auc:.4f}")
