import torch
import torch.nn as nn
import torch.optim as optim
from data_analyse.ML_HUB.Project_AD.P_AD_utils import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from easier_nn.train_net import NetTrainerFNN


# 定义Wide&Deep模型
class WideDeepNN(nn.Module):
    def __init__(self, input_dim):
        super(WideDeepNN, self).__init__()

        # Wide部分：线性模型
        self.wide_layer = nn.Linear(input_dim, 1)

        # Deep部分：深度神经网络
        self.deep_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 假设wide_input和deep_input都用相同的特征
        wide_output = self.wide_layer(x)
        deep_output = self.deep_layers(x)
        return (wide_output + deep_output).view(-1)



# 主函数
if __name__ == "__main__":
    # 数据加载
    processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"
    ad_u_data = pd.read_csv(f"{processed_data_path}/ad_u_data.csv")
    ad_u_data = ad_u_data.head(100000)
    # 使用众数填补缺失值(暂时)
    for col in ad_u_data.columns:
        ad_u_data[col] = ad_u_data[col].fillna(ad_u_data[col].mode()[0])

    # 数据预处理
    # 选择特征列和目标列
    feature_columns = [
        "day", "hour",
        "final_gender_code", "age_level", "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
        "price"
    ]
    target_column = "clk"  # 假设目标是点击行为

    # 提取特征和标签
    X = ad_u_data[feature_columns].astype(float)
    y = ad_u_data[target_column].astype(float)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader, test_loader = load_data(X_train, y_train, X_test, y_test)

    # 定义模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train.shape[1]
    model = WideDeepNN(input_dim).to(device)
    class_weights = torch.tensor([1.0, len(ad_u_data) / ad_u_data['clk'].sum()]).to(device)
    print(f"Class weights: {class_weights}")
    # criterion = nn.BCELoss(weight=class_weights)  # 不能直接加权重，因为BCE的输入是概率值
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # 权重只作用于类别 1
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    net_trainer = NetTrainerFNN(train_loader, test_loader, model, criterion, optimizer,
                                epochs=20, eval_type="acc", eval_interval=1)
    net_trainer.train_net()

    # # 模型训练
    # train_model(model, train_loader, test_loader, optimizer, criterion, epochs=1)
    #
    # # 模型评测
    # evaluate_model(model, test_loader)
