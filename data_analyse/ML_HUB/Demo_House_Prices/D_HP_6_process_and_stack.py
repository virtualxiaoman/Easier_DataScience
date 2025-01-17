import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(f"train shape: {train.shape}, test shape: {test.shape}")
    return train, test


train_data, test_data = load_data()

# 特征工程：将类别变量进行编码
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# 处理缺失值，填充或删除
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# # 对齐训练集和测试集的列（避免列数不同）
# train, test = train.align(test, join='left', axis=1)

# 提取特征和目标变量
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']
X_test = test_data.drop('Id', axis=1)
X, X_test = X.align(X_test, join='left', axis=1)  # 对齐训练集和测试集的列（确保列名一致）
X, X_test = X.fillna(0), X_test.fillna(0)  # 填充缺失值
# 切分训练数据为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

# 标准化数据（对线性模型特别重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


class StackedModelRegressor:
    def __init__(self, base_learners, final_estimator=None):
        """
        初始化堆叠模型。
        base_learners: List of tuples (name, model) for base models
        final_estimator: Final estimator for stacking (默认LinearRegression)
        """
        self.base_learners = base_learners
        self.final_estimator = final_estimator if final_estimator else LinearRegression()
        self.stacking_model = StackingRegressor(
            estimators=self.base_learners,
            final_estimator=self.final_estimator
        )

    def fit(self, X_train, y_train, X_train_scaled, X_val, y_val, X_val_scaled):
        """
        训练堆叠模型。
        """
        # 训练基础学习器
        for name, model in self.base_learners:
            if hasattr(model, 'coef_'):
                model.fit(X_train_scaled, y_train)  # 对于线性模型，使用标准化后的数据
                pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)

            mse = mean_squared_error(y_val, pred)
            print(f'>>> {name}, RMSE: {np.sqrt(mse)}')

        self.stacking_model.fit(X_train, y_train)

    def predict(self, X_val, y_val, X_test):
        """
        预测。
        """
        # 评估Stacking模型
        stacking_pred = self.stacking_model.predict(X_val)
        stacking_mse = mean_squared_error(y_val, stacking_pred)
        print(f'>>> Stacking RMSE: {np.sqrt(stacking_mse)}')
        # 对测试集进行预测
        stacked_predictions = self.stacking_model.predict(X_test)
        print("Stack pred:", stacked_predictions[:5])
        print(stacked_predictions.shape)  # (1459,)
        return stacked_predictions


# 定义基础模型
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13))
lasso = LassoCV()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1)  # -1表示不输出训练信息

# 训练模型
base_learners = [
    ('ridge', ridge),
    ('lasso', lasso),
    ('rf', rf),
    ('xgb', xgb_model),
    ('lgb', lgb_model)
]
stacked_model = StackedModelRegressor(base_learners)
stacked_model.fit(X_train, y_train, X_train_scaled, X_val, y_val, X_val_scaled)
stacked_predictions = stacked_model.predict(X_val, y_val, X_test)


# # 训练模型
# ridge.fit(X_train_scaled, y_train)
# lasso.fit(X_train_scaled, y_train)
# rf.fit(X_train, y_train)
# xgb_model.fit(X_train, y_train)
# lgb_model.fit(X_train, y_train)
#
# # 在验证集上评估模型
# models = [ridge, lasso, rf, xgb_model, lgb_model]
# for model in models:
#     pred = model.predict(X_val_scaled if hasattr(model, 'coef_') else X_val)
#     mse = mean_squared_error(y_val, pred)
#     print(f'>>> {model.__class__.__name__}, RMSE: {np.sqrt(mse)}')
#
# # 基学习器：使用之前的模型
# base_learners = [
#     ('ridge', RidgeCV(alphas=np.logspace(-6, 6, 13))),
#     ('lasso', LassoCV()),
#     ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('xgb', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, random_state=42)),
#     ('lgb', lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1))
# ]
#
# # 元学习器：使用线性回归
# stacking_model = StackingRegressor(
#     estimators=base_learners,
#     final_estimator=LinearRegression()
# )
#
# # 训练Stacking模型
# stacking_model.fit(X_train, y_train)
#
# # 评估Stacking模型
# stacking_pred = stacking_model.predict(X_val)
# stacking_mse = mean_squared_error(y_val, stacking_pred)
# print(f'>>> Stacking RMSE: {np.sqrt(stacking_mse)}')
#
# # 对测试集进行预测
# stacked_predictions = stacking_model.predict(X_test)
# print("Stacking predictions:", stacked_predictions[:10])
# print(stacked_predictions.shape)


def generate_submission(id_col, pred_col, submission_name="submission_2"):
    output_dir = "output/stack_model"
    os.makedirs(output_dir, exist_ok=True)

    submission = pd.DataFrame({
        'Id': id_col,
        'SalePrice': pred_col
    })
    csv_name = f"{submission_name}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    submission.to_csv(csv_path, index=False)
    print("Submission file saved at", csv_path)


generate_submission(test_data['Id'], stacked_predictions)
