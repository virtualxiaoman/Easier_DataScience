import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 加载数据
processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"
ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')

# 选取前10000个数据
ad_u_data = ad_u_data.iloc[:1000000]

# 去重
ad_u_data = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)

# 将用户和广告 ID 映射为连续索引
ad_u_data['user_id'] = pd.Categorical(ad_u_data['user_id']).codes
ad_u_data['adgroup_id'] = pd.Categorical(ad_u_data['adgroup_id']).codes

# 用户和广告的总数
user_length = ad_u_data['user_id'].nunique()
ad_length = ad_u_data['adgroup_id'].nunique()

# 划分数据集
X = ad_u_data[['user_id', 'adgroup_id']].values
y = ad_u_data['clk'].values

# 划分训练集、验证集、测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)  # CrossEntropyLoss 要求目标为 long 类型
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{device}")


# 模型定义
class EmbeddingDotProductModel(nn.Module):
    def __init__(self, user_length, ad_length, embedding_dim=32):
        super(EmbeddingDotProductModel, self).__init__()
        self.user_embedding = nn.Embedding(user_length, embedding_dim)
        self.ad_embedding = nn.Embedding(ad_length, embedding_dim)

    def forward(self, user_ids, ad_ids):
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        ad_emb = self.ad_embedding(ad_ids)  # [batch_size, embedding_dim]
        dot_product = (user_emb * ad_emb).sum(dim=1)  # 点积
        dot_product = torch.sigmoid(dot_product)  # 应用 sigmoid 激活函数，将值域映射到 [0, 1]
        # 将点积结果变为 [batch_size, 2] 的形状，其中第一列为 1-dot_product，第二列为 dot_product
        dot_product = torch.stack([1 - dot_product, dot_product], dim=1)
        return dot_product  # 输出 logits


# 超参数
embedding_dim = 32
learning_rate = 0.001
num_epochs = 20
batch_size = 1024

# 初始化模型、损失函数和优化器
model = EmbeddingDotProductModel(user_length, ad_length, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 数据加载
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []

    for batch in train_loader:
        user_ids, ad_ids = batch[0][:, 0].to(device), batch[0][:, 1].to(device)
        labels = batch[1].to(device)

        # 前向传播
        outputs = model(user_ids, ad_ids)  # logits 输出
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算训练集准确率
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_train_preds.extend(preds)
        all_train_labels.extend(labels.cpu().numpy())

    # 计算训练集准确率
    train_acc = accuracy_score(all_train_labels, all_train_preds)

    # 验证模型
    model.eval()
    with torch.no_grad():
        val_loss = 0
        all_preds = []
        all_labels = []
        for batch in val_loader:
            user_ids, ad_ids = batch[0][:, 0].to(device), batch[0][:, 1].to(device)
            labels = batch[1].to(device)
            outputs = model(user_ids, ad_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
# 测试模型

model.eval()
with torch.no_grad():
    y_pred = []
    for batch in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test), batch_size=batch_size):
        user_ids, ad_ids = batch[0][:, 0].to(device), batch[0][:, 1].to(device)
        outputs = model(user_ids, ad_ids)
        y_pred.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    y_true = y_test.numpy()
    auc = roc_auc_score(y_true, y_pred)
    print(f"Test AUC: {auc:.4f}")

# 查看样本的输入、embedding 和点积
sample_indices = torch.arange(5)  # 测试集前 5 个样本
sample_user_ids = X_test[sample_indices, 0].to(device)
sample_ad_ids = X_test[sample_indices, 1].to(device)
sample_labels = y_test[sample_indices].to(device)

with torch.no_grad():
    np.set_printoptions(precision=5, suppress=True)  # 设置浮点数显示精度为 5 位小数
    user_emb = model.user_embedding(sample_user_ids)
    ad_emb = model.ad_embedding(sample_ad_ids)
    dot_product = (user_emb * ad_emb).sum(dim=1)
    softmax_values = torch.softmax(dot_product, dim=0)
    print("Sample User IDs:", sample_user_ids.cpu().numpy())
    print("Sample Ad IDs:", sample_ad_ids.cpu().numpy())
    # print("User Embeddings:", user_emb.cpu().numpy())
    # print("Ad Embeddings:", ad_emb.cpu().numpy())
    print("Dot Products:", dot_product.cpu().numpy())
    print("Softmax Values:", softmax_values.cpu().numpy())
    print("Labels:", sample_labels.cpu().numpy())

# 可视化嵌入
with torch.no_grad():
    user_embeddings = model.user_embedding.weight[:1000].cpu().numpy()  # 选取前 1000 个用户嵌入
    ad_embeddings = model.ad_embedding.weight[:1000].cpu().numpy()  # 选取前 1000 个广告嵌入

# 使用 t-SNE 降维
all_embeddings = np.vstack([user_embeddings, ad_embeddings])
labels = np.array(["User"] * len(user_embeddings) + ["Ad"] * len(ad_embeddings))
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# 绘制可视化
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    idx = labels == label
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, alpha=0.7)
plt.legend()
plt.title("User and Ad Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# 保存模型
torch.save(model.state_dict(), "embedding_model.pth")
