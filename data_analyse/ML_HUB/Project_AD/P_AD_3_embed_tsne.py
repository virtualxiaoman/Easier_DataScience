import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# 读取数据
processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"

ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')

# 数据预处理：去重并重置索引，避免重复的用户和广告组合
ad_user_sample_data_ = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)
# 将用户和广告 ID 映射为连续索引
ad_user_sample_data_['user_id'] = pd.Categorical(ad_user_sample_data_['user_id']).codes
ad_user_sample_data_['adgroup_id'] = pd.Categorical(ad_user_sample_data_['adgroup_id']).codes
user_length = ad_user_sample_data_['user_id'].nunique()  # 用户总数
ad_length = ad_user_sample_data_['adgroup_id'].nunique()  # 广告总数


# 定义 Embedding 模型
class AdUserEmbeddingModel(nn.Module):
    """
    初始化模型，定义广告和用户的嵌入层及前向传播的计算逻辑。

        参数：
        - user_size: int, 用户总数
        - ad_size: int, 广告总数
        - embedding_dim: int, 嵌入向量的维度大小
    """

    def __init__(self, user_length, ad_length, embedding_size=50):
        super(AdUserEmbeddingModel, self).__init__()
        # 定义用户和广告的嵌入层
        self.user_embedding = nn.Embedding(user_length, embedding_size)
        self.ad_embedding = nn.Embedding(ad_length, embedding_size)
        # 定义点积计算后的全连接层
        self.fc = nn.Linear(1, 1)
        # 使用 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, ad_id):
        # 获取用户和广告的嵌入向量
        user_embeds = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        ad_embeds = self.ad_embedding(ad_ids)  # [batch_size, embedding_dim]
        # 点积计算用户与广告的相似性
        dot_product = torch.sum(user_embeds * ad_embeds, dim=1, keepdim=True)  # [batch_size, 1]
        # 通过全连接层调整并使用 Sigmoid 输出概率
        output = self.sigmoid(self.fc(dot_product))
        # 输出各个size
        print(
            f"user_embeds: {user_embeds.size()}, ad_embeds: {ad_embeds.size()}, dot_product: {dot_product.size()}, output: {output.size()}")
        return output


# 初始化模型
embedding_size = 50

# # 检查 GPU 可用性
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # 需要的显存过大了

# 初始化模型并移动到 GPU
model = AdUserEmbeddingModel(user_length, ad_length, embedding_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据准备，移动张量到 GPU
user_ids = torch.tensor(ad_user_sample_data_['user_id'].values).to(device)
ad_ids = torch.tensor(ad_user_sample_data_['adgroup_id'].values).to(device)
clicks = torch.tensor(ad_user_sample_data_['clk'].values, dtype=torch.float32).to(device)

# 训练
epochs = 5
batch_size = 32

for epoch in range(epochs):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    for i in range(0, len(user_ids), batch_size):
        # 获取当前批次的数据
        user_batch = user_ids[i:i + batch_size]
        ad_batch = ad_ids[i:i + batch_size]
        clicks_batch = clicks[i:i + batch_size]

        print(f"user_batch: {user_batch.size()}, ad_batch: {ad_batch.size()}, clicks_batch: {clicks_batch.size()}")

        # 前向传播
        outputs = model(user_batch, ad_batch)
        loss = criterion(outputs.squeeze(), clicks_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'ad_user_embedding_model.pth')
# 提取 Embedding
ad_embedding_weights = model.ad_embedding.weight.detach().numpy()
user_embedding_weights = model.user_embedding.weight.detach().numpy()


# 降维。components: int, 降维后的目标维度。
def reduce_dim(weights, components=2, method='tsne'):
    if method == 'tsne':
        return TSNE(n_components=components, metric='cosine').fit_transform(weights)
    elif method == 'umap':
        pass
        # return UMAP(n_components=components, metric='cosine').fit_transform(weights)


ad_tsne = reduce_dim(ad_embedding_weights, method='tsne')
user_tsne = reduce_dim(user_embedding_weights, method='tsne')

# 可视化
plt.scatter(ad_tsne[:, 0], ad_tsne[:, 1], alpha=0.5, label='Ads')
plt.scatter(user_tsne[:, 0], user_tsne[:, 1], alpha=0.5, label='Users')
plt.legend()
plt.show()
