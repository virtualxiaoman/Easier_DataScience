import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

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
    le_user = LabelEncoder()
    le_ad = LabelEncoder()
    ad_u_data['user_id'] = le_user.fit_transform(ad_u_data['user_id'])
    ad_u_data['adgroup_id'] = le_ad.fit_transform(ad_u_data['adgroup_id'])

    return ad_u_data, len(le_user.classes_), len(le_ad.classes_)


# 定义用户-广告 Embedding 模型
class AdUserEmbeddingModel(nn.Module):
    def __init__(self, user_length, ad_length, embedding_size=50):
        super(AdUserEmbeddingModel, self).__init__()
        # 嵌入层，用户和广告的嵌入
        self.ad_embedding = nn.Embedding(ad_length, embedding_size)
        self.user_embedding = nn.Embedding(user_length, embedding_size)
        self.fc = nn.Linear(1, 1)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.ad_embedding.weight)

    def forward(self, user, ad):
        # 获取用户和广告的嵌入向量
        user_emb = self.user_embedding(user)  # [batch_size, embedding_size]
        ad_emb = self.ad_embedding(ad)  # [batch_size, embedding_size]
        user_emb = F.normalize(user_emb, p=2, dim=1)  # [batch_size, embedding_size]
        ad_emb = F.normalize(ad_emb, p=2, dim=1)  # [batch_size, embedding_size]
        # 计算用户和广告嵌入的点积
        dot_product = (user_emb * ad_emb).sum(dim=1, keepdim=True)  # [batch_size, 1]
        # 通过sigmoid激活进行点击预测
        output = torch.sigmoid(self.fc(dot_product))  # [batch_size, 1]
        return output


# 转换数据为 PyTorch Dataset 格式
def create_pytorch_dataset(input_data, target_data, batch_size):
    tensor_data = TensorDataset(torch.tensor(input_data['user_id'], dtype=torch.long),
                                torch.tensor(input_data['adgroup_id'], dtype=torch.long),
                                torch.tensor(target_data, dtype=torch.float32))
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    return dataloader


# 评估模型性能
def evaluate_model(model, val_input, val_target, device):
    model.eval()
    val_input = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in val_input.items()}
    val_target = torch.tensor(val_target, dtype=torch.float32).to(device)

    with torch.no_grad():
        val_pred = model(val_input['user_id'], val_input['adgroup_id']).cpu().numpy()

    # 统计预测分布
    predicted_0 = np.sum(val_pred < 0.5)
    predicted_1 = np.sum(val_pred >= 0.5)
    predicted_0_ratio = predicted_0 / len(val_pred)
    predicted_1_ratio = predicted_1 / len(val_pred)

    print(f"Predicted 0: {predicted_0}, Predicted 1: {predicted_1}")
    print(f"Predicted 0 ratio: {predicted_0_ratio:.2f}, Predicted 1 ratio: {predicted_1_ratio:.2f}")

    # 计算 AUC
    val_auc = roc_auc_score(val_target.cpu(), val_pred)
    print(f"Validation AUC: {val_auc:.4f}")

    # 输出分类报告
    val_pred = np.round(val_pred)
    print(classification_report(val_target.cpu(), val_pred))

    return val_pred, val_auc


if __name__ == "__main__":
    # 数据加载与预处理
    SAMPLE_SIZE = 2048 * 300
    BATCH_SIZE = 2048
    EPOCHS = 10

    ad_user_sample_data_, user_length, ad_length = load_and_preprocess_data(PROCESSED_DATA_PATH, SAMPLE_SIZE)

    # 输出用户和广告的长度
    print(f"User length: {user_length}, Ad length: {ad_length}")

    # 准备输入与目标数据
    input_data = {
        'user_id': ad_user_sample_data_['user_id'].values,
        'adgroup_id': ad_user_sample_data_['adgroup_id'].values
    }
    target_data = ad_user_sample_data_['clk'].values

    # 划分训练集与验证集
    combined_data = pd.DataFrame({
        'user_id': ad_user_sample_data_['user_id'],
        'adgroup_id': ad_user_sample_data_['adgroup_id']
    })
    train_input, val_input, train_target, val_target = train_test_split(
        combined_data, target_data, test_size=0.2, random_state=42
    )

    print(f"Input data: {input_data['user_id'].shape, input_data['adgroup_id'].shape}, ")
    # 输出train_input的shape
    print(train_input.shape)
    # 查找train_input和val_input有没有相同的值的shape
    print(train_input[train_input['user_id'].isin(val_input['user_id'])].shape)
    print(train_input[train_input['adgroup_id'].isin(val_input['adgroup_id'])].shape)
    # 查找train_input的(x, y)有没有和val_input的(x, y)相同的值
    print(train_input[train_input['user_id'].isin(val_input['user_id']) &
                      train_input['adgroup_id'].isin(val_input['adgroup_id'])].shape)

    # 创建模型
    model = AdUserEmbeddingModel(user_length, ad_length).cuda()

    # 输出模型结构
    print(model)

    # 创建 DataLoader
    train_dataloader = create_pytorch_dataset(train_input, train_target, BATCH_SIZE)
    val_input = {
        'user_id': val_input['user_id'].values,
        'adgroup_id': val_input['adgroup_id'].values
    }

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 训练模型
    print("开始训练 Embedding 模型")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for user, ad, target in train_dataloader:
            user, ad, target = user.cuda(), ad.cuda(), target.cuda()

            # 前向传播
            optimizer.zero_grad()
            output = model(user, ad).squeeze()
            loss = criterion(output, target)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_dataloader):.4f}")

    # 评估模型
    print("评估模型性能")
    evaluate_model(model, val_input, val_target, device='cuda')
