import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 数据路径
PROCESSED_DATA_PATH = "G:/DataSets/Ali_Display_Ad_Click/processed_data"


# 加载和预处理数据
def load_and_preprocess_data(path, sample_size):
    # 加载数据
    ad_u_data = pd.read_csv(f'{path}/ad_u_data.csv')
    ad_u_data = ad_u_data.iloc[:sample_size]

    # 数据去重
    ad_u_data = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)

    # 将用户和广告 ID 映射为连续索引
    ad_u_data['user_id'] = pd.Categorical(ad_u_data['user_id']).codes
    ad_u_data['adgroup_id'] = pd.Categorical(ad_u_data['adgroup_id']).codes

    return ad_u_data


# 定义用户-广告 Embedding 模型
class AdUserEmbeddingModel(nn.Module):
    def __init__(self, user_length, ad_length, embedding_size=50):
        super(AdUserEmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(user_length, embedding_size)
        self.ad_embedding = nn.Embedding(ad_length, embedding_size)
        self.fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, ad_id):
        user_embedded = self.user_embedding(user_id)
        ad_embedded = self.ad_embedding(ad_id)
        dot_product = torch.sum(user_embedded * ad_embedded, dim=1, keepdim=True)
        print(
            f"user_embedded: {user_embedded.size()}, ad_embedded: {ad_embedded.size()}, dot_product: {dot_product.size()}")
        output = self.fc(dot_product)
        return self.sigmoid(output)


# 评估模型性能
def evaluate_model(model, val_input, val_target):
    # 预测值
    val_pred = model(val_input['user_id'], val_input['adgroup_id']).detach().numpy()

    # 统计预测分布
    predicted_0 = np.sum(val_pred < 0.5)
    predicted_1 = np.sum(val_pred >= 0.5)
    predicted_0_ratio = predicted_0 / len(val_pred)
    predicted_1_ratio = predicted_1 / len(val_pred)

    print(f"Predicted 0: {predicted_0}, Predicted 1: {predicted_1}")
    print(f"Predicted 0 ratio: {predicted_0_ratio:.2f}, Predicted 1 ratio: {predicted_1_ratio:.2f}")

    # 计算 AUC
    val_auc = roc_auc_score(val_target, val_pred)
    print(f"Validation AUC: {val_auc:.4f}")

    # 输出分类报告
    val_pred = np.round(val_pred)
    print(classification_report(val_target, val_pred))

    return val_pred, val_auc


if __name__ == "__main__":
    # 数据加载与预处理
    SAMPLE_SIZE = 2048 * 300
    ad_user_sample_data_ = load_and_preprocess_data(PROCESSED_DATA_PATH, SAMPLE_SIZE)

    # 用户和广告的总数
    user_length = ad_user_sample_data_['user_id'].nunique()
    ad_length = ad_user_sample_data_['adgroup_id'].nunique()
    print(f"User length: {user_length}, Ad length: {ad_length}")

    # 准备输入与目标数据
    input_data = {
        'user_id': torch.tensor(ad_user_sample_data_['user_id'].values, dtype=torch.long),
        'adgroup_id': torch.tensor(ad_user_sample_data_['adgroup_id'].values, dtype=torch.long)
    }
    target_data = torch.tensor(ad_user_sample_data_['clk'].values, dtype=torch.float32)

    # 划分训练集与验证集
    combined_data = pd.DataFrame({
        'user_id': ad_user_sample_data_['user_id'],
        'adgroup_id': ad_user_sample_data_['adgroup_id']
    })
    train_input, val_input, train_target, val_target = train_test_split(
        combined_data, target_data, test_size=0.2, random_state=42
    )
    val_input = {
        'user_id': torch.tensor(val_input['user_id'].values, dtype=torch.long),
        'adgroup_id': torch.tensor(val_input['adgroup_id'].values, dtype=torch.long)
    }

    # 创建模型
    embedding_model = AdUserEmbeddingModel(user_length, ad_length)
    print(embedding_model)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(embedding_model.parameters(), lr=0.001)

    # 训练模型
    print("开始训练 Embedding 模型")
    for epoch in range(3):
        embedding_model.train()
        optimizer.zero_grad()
        output = embedding_model(input_data['user_id'], input_data['adgroup_id'])
        loss = criterion(output, target_data.unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 评估模型
    print("评估模型性能")
    evaluate_model(embedding_model, val_input, val_target.numpy())
