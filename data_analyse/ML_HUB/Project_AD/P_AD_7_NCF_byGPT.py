import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 读取数据
processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"
ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')
print(f"Data shape: {ad_u_data.shape}")
# 选择前100000条数据，避免训练数据过长
ad_u_data = ad_u_data.head(100000)

# 选择有用的列
ad_u_data = ad_u_data[['user_id', 'adgroup_id', 'clk']]
# # 数据去重
# ad_u_data = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)

# 标签编码
user_encoder = LabelEncoder()
adgroup_encoder = LabelEncoder()
# # 将用户和广告 ID 映射为连续索引
# ad_u_data['user_id'] = pd.Categorical(ad_u_data['user_id']).codes
# ad_u_data['adgroup_id'] = pd.Categorical(ad_u_data['adgroup_id']).codes

ad_u_data['user_id'] = user_encoder.fit_transform(ad_u_data['user_id'])
ad_u_data['adgroup_id'] = adgroup_encoder.fit_transform(ad_u_data['adgroup_id'])

# 切分训练和验证集
train_data, val_data = train_test_split(ad_u_data, test_size=0.2, random_state=42)
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")


class AdClickDataset(Dataset):
    def __init__(self, data, user_encoder, adgroup_encoder):
        self.data = data
        self.user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.adgroup_ids = torch.tensor(data['adgroup_id'].values, dtype=torch.long)
        self.labels = torch.tensor(data['clk'].values, dtype=torch.float)
        self.num_users = len(user_encoder.classes_)
        self.num_ads = len(adgroup_encoder.classes_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.adgroup_ids[idx], self.labels[idx]


def oversample_data(train_data):
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X = train_data[['user_id', 'adgroup_id']].values
    y = train_data['clk'].values
    X_res, y_res = smote.fit_resample(X, y)

    # 创建新的DataFrame并返回
    resampled_data = pd.DataFrame(X_res, columns=['user_id', 'adgroup_id'])
    resampled_data['clk'] = y_res
    return resampled_data


# train_data = oversample_data(train_data)
# print(f"[log] 这里进行了过采样。Resampled train data shape: {train_data.shape}")
train_dataset = AdClickDataset(train_data, user_encoder, adgroup_encoder)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

val_dataset = AdClickDataset(val_data, user_encoder, adgroup_encoder)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


class NCFModel(nn.Module):
    def __init__(self, num_users, num_ads, embedding_dim=50, hidden_dim=128, dropout=0.2):
        super(NCFModel, self).__init__()

        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.adgroup_embedding = nn.Embedding(num_ads, embedding_dim)

        # 全连接层
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, user_id, adgroup_id):
        # 获取嵌入
        user_emb = self.user_embedding(user_id)
        adgroup_emb = self.adgroup_embedding(adgroup_id)

        # 拼接用户和广告组的嵌入
        x = torch.cat([user_emb, adgroup_emb], dim=-1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # # Sigmoid 输出概率
        # return self.sigmoid(x).squeeze()
        return x.squeeze()  # 返回原始得分，大小为 (batch_size,)


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = NCFModel(num_users=train_dataset.num_users, num_ads=train_dataset.num_ads).to(device)
model = NCFModel(num_users=train_dataset.num_users, num_ads=train_dataset.num_ads, dropout=0.5).to(device)

# 损失函数与优化器
class_weights = torch.tensor([1.0, len(ad_u_data) / ad_u_data['clk'].sum()]).to(device)
print(f"Class weights: {class_weights}")
# criterion = nn.BCELoss(weight=class_weights)  # 不能直接加权重，因为BCE的输入是概率值
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # 权重只作用于类别 1
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)


# 训练并评估函数
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    print("Start training...")
    all_preds = []
    all_labels = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for user_id, adgroup_id, labels in train_loader:
            user_id = user_id.to(device)
            adgroup_id = adgroup_id.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 模型预测
            outputs = model(user_id, adgroup_id)

            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += len(labels)

            # 保存预测和标签
            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

        # 计算训练集准确率
        train_acc = correct_preds / total_preds

        # 验证
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for user_id, adgroup_id, labels in val_loader:
                user_id = user_id.to(device)
                adgroup_id = adgroup_id.to(device)
                labels = labels.to(device)

                outputs = model(user_id, adgroup_id)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_preds += len(labels)

                # 保存预测和标签
                val_preds.extend(outputs.cpu().detach().numpy())
                val_labels.extend(labels.cpu().detach().numpy())

        val_acc = correct_preds / total_preds

        # 计算AUC
        train_auc = roc_auc_score(all_labels, all_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # 在每个 epoch 后输出分类报告
        val_preds_class = (torch.tensor(val_preds) > 0.5).float()
        # print("\nClassification Report (Validation):")
        # print(classification_report(val_labels, val_preds_class))

    return val_preds, val_labels


# 训练模型并评估
val_preds, val_labels = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=30)


# 测试
def evaluate_model(model, test_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for user_id, adgroup_id, labels in test_loader:
            user_id = user_id.to(device)
            adgroup_id = adgroup_id.to(device)
            labels = labels.to(device)

            outputs = model(user_id, adgroup_id)
            preds_list.extend(outputs.cpu().detach().numpy())
            labels_list.extend(labels.cpu().detach().numpy())
            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += len(labels)

    accuracy = correct_preds / total_preds
    auc = roc_auc_score(labels_list, preds_list)

    # 输出分类报告
    preds_class = (torch.tensor(preds_list) > 0.5).float()
    cr = classification_report(labels_list, preds_class)
    return accuracy, auc, cr


# 假设你有一个测试集
test_data = val_data  # 这里可以替换为你的测试集
test_dataset = AdClickDataset(test_data, user_encoder, adgroup_encoder)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

accuracy, auc, cr = evaluate_model(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
print("\nClassification Report (Test):")
print(cr)
