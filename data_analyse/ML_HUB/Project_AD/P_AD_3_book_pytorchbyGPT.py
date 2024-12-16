import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from data_analyse.ML_HUB.Project_AD.P_AD_3_book_optimbyGPT import create_tf_dataset, ad_user_embedding_model

# 数据路径
PROCESSED_DATA_PATH = "G:/DataSets/Ali_Display_Ad_Click/processed_data"


# 数据集类
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


class AdUserDataset(Dataset):
    def __init__(self, user_ids, ad_ids, targets):
        self.user_ids = user_ids
        self.ad_ids = ad_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'ad_id': self.ad_ids[idx],
            'target': self.targets[idx]
        }


# 定义模型
class AdUserEmbeddingModel(nn.Module):
    def __init__(self, user_length, ad_length, embedding_size=50):
        super(AdUserEmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(user_length, embedding_size)
        self.ad_embedding = nn.Embedding(ad_length, embedding_size)
        self.fc = nn.Linear(1, 1)
        # 初始化权重
        nn.init.kaiming_uniform_(self.ad_embedding.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.user_embedding.weight, a=np.sqrt(5))
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, user_ids, ad_ids):
        user_embedded = self.user_embedding(user_ids)
        ad_embedded = self.ad_embedding(ad_ids)
        dot_product = torch.sum(user_embedded * ad_embedded, dim=1, keepdim=True)
        output = torch.sigmoid(self.fc(dot_product))
        return output


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        user_ids = batch['user_id'].to(device)
        ad_ids = batch['ad_id'].to(device)
        targets = batch['target'].to(device).float()

        optimizer.zero_grad()
        outputs = model(user_ids, ad_ids).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    evaluate_model(model, val_loader, DEVICE)
    return total_loss / len(train_loader)


# 评估函数
def evaluate_model(model, val_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch['user_id'].to(device)
            ad_ids = batch['ad_id'].to(device)
            targets = batch['target'].to(device).float()

            outputs = model(user_ids, ad_ids).squeeze()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    # 计算 AUC
    val_auc = roc_auc_score(all_targets, all_predictions)
    print(f"Validation AUC: {val_auc:.4f}")

    return val_auc


if __name__ == "__main__":
    # 数据加载与预处理
    SAMPLE_SIZE = 2048 * 300
    BATCH_SIZE = 2048
    EPOCHS = 50
    EMBEDDING_SIZE = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ad_user_sample_data_ = load_and_preprocess_data(PROCESSED_DATA_PATH, SAMPLE_SIZE)

    # 用户和广告的总数
    user_length = ad_user_sample_data_['user_id'].nunique()
    ad_length = ad_user_sample_data_['adgroup_id'].nunique()
    print(f"User length: {user_length}, Ad length: {ad_length}")

    # # 准备输入与目标数据
    # input_user_ids = ad_user_sample_data_['user_id'].values
    # input_ad_ids = ad_user_sample_data_['adgroup_id'].values
    # target_data = ad_user_sample_data_['clk'].values
    #
    # # 划分训练集与验证集
    # train_user_ids, val_user_ids, train_ad_ids, val_ad_ids, train_targets, val_targets = train_test_split(
    #     input_user_ids, input_ad_ids, target_data, test_size=0.2, random_state=42
    # )

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
    val_input = {
        'user_id': val_input['user_id'].values,
        'adgroup_id': val_input['adgroup_id'].values
    }


    # 转换为 TensorFlow Dataset
    train_dataset = create_tf_dataset(input_data, target_data, BATCH_SIZE)
    val_dataset = create_tf_dataset(val_input, val_target, BATCH_SIZE)
    # train_dataset = AdUserDataset(train_user_ids, train_ad_ids, train_targets)
    # val_dataset = AdUserDataset(val_user_ids, val_ad_ids, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    model = AdUserEmbeddingModel(user_length, ad_length, EMBEDDING_SIZE).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 训练模型
    # print("开始训练模型")
    # for epoch in range(EPOCHS):
    #     train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
    #     print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}")
    embedding_model = ad_user_embedding_model(user_length, ad_length)
    embedding_model.fit(train_dataset, validation_data=(val_input, val_target), epochs=3)
    # 评估模型
    print("评估模型性能")
    from P_AD_3_book_optimbyGPT import evaluate_model

    evaluate_model(embedding_model, val_input, val_target)

    # evaluate_model(model, val_loader, DEVICE)

