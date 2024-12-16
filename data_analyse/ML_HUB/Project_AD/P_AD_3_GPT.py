import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import Dataset, DataLoader

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


# 自定义数据集类
class AdUserDataset(Dataset):
    def __init__(self, user_ids, ad_ids, targets):
        self.user_ids = user_ids
        self.ad_ids = ad_ids
        self.targets = targets

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'adgroup_id': torch.tensor(self.ad_ids[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }


# 定义用户-广告 Embedding 模型
class AdUserEmbeddingModel(nn.Module):
    def __init__(self, user_length, ad_length, embedding_size=50):
        super(AdUserEmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(user_length, embedding_size)
        self.ad_embedding = nn.Embedding(ad_length, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, ad_id):
        user_embedded = self.user_embedding(user_id)
        ad_embedded = self.ad_embedding(ad_id)
        dot_product = (user_embedded * ad_embedded).sum(dim=1)  # 点积
        out = self.fc(dot_product)
        out = self.sigmoid(out)  # 输出 sigmoid 激活
        return out


# 评估模型性能
def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch['user_id'].to(device)
            ad_ids = batch['adgroup_id'].to(device)
            targets = batch['target'].to(device)

            preds = model(user_ids, ad_ids)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 计算 AUC
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Validation AUC: {val_auc:.4f}")

    # 输出分类报告
    val_preds = np.round(all_preds)
    print(classification_report(all_targets, val_preds))

    return all_preds, val_auc


if __name__ == "__main__":
    # 数据加载与预处理
    SAMPLE_SIZE = 2048 * 300
    BATCH_SIZE = 2048
    EPOCHS = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ad_user_sample_data_ = load_and_preprocess_data(PROCESSED_DATA_PATH, SAMPLE_SIZE)

    # 用户和广告的总数
    user_length = ad_user_sample_data_['user_id'].nunique()
    ad_length = ad_user_sample_data_['adgroup_id'].nunique()
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

    # 创建 PyTorch 数据集
    train_dataset = AdUserDataset(train_input['user_id'].values, train_input['adgroup_id'].values, train_target)
    val_dataset = AdUserDataset(val_input['user_id'].values, val_input['adgroup_id'].values, val_target)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    model = AdUserEmbeddingModel(user_length, ad_length).to(DEVICE)
    print(model)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # 训练模型
    print("开始训练 Embedding 模型")
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            user_ids = batch['user_id'].to(DEVICE)
            ad_ids = batch['adgroup_id'].to(DEVICE)
            targets = batch['target'].to(DEVICE)

            optimizer.zero_grad()
            preds = model(user_ids, ad_ids)
            loss = criterion(preds.squeeze(), targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

    # 评估模型
    print("评估模型性能")
    evaluate_model(model, val_loader, DEVICE)
