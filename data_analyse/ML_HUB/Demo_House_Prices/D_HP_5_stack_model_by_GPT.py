import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 查看数据概况
print(f"train shape: {train.shape}, test shape: {test.shape}")

# 特征工程：将类别变量进行编码
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# 处理缺失值，填充或删除
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# 对齐训练集和测试集的列（避免列数不同）
train, test = train.align(test, join='left', axis=1)

# 提取特征和目标变量
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']
X_test = test.drop('Id', axis=1)
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

# 定义基础模型
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13))
lasso = LassoCV()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1)  # -1表示不输出训练信息

# 训练模型
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# 在验证集上评估模型
models = [ridge, lasso, rf, xgb_model, lgb_model]
for model in models:
    pred = model.predict(X_val_scaled if hasattr(model, 'coef_') else X_val)
    mse = mean_squared_error(y_val, pred)
    print(f'>>> {model.__class__.__name__}, RMSE: {np.sqrt(mse)}')

from sklearn.ensemble import StackingRegressor

# 基学习器：使用之前的模型
base_learners = [
    ('ridge', RidgeCV(alphas=np.logspace(-6, 6, 13))),
    ('lasso', LassoCV()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1))
]

# 元学习器：使用线性回归
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=LinearRegression()
)

# 训练Stacking模型
stacking_model.fit(X_train, y_train)

# 评估Stacking模型
stacking_pred = stacking_model.predict(X_val)
stacking_mse = mean_squared_error(y_val, stacking_pred)
print(f'>>> Stacking RMSE: {np.sqrt(stacking_mse)}')

# 对测试集进行预测
stacked_predictions = stacking_model.predict(X_test)

# 生成Kaggle提交文件
output_dir = "output/stack_model"
os.makedirs(output_dir, exist_ok=True)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': stacked_predictions.flatten()
})

submission.to_csv(os.path.join(output_dir, "submission_2.csv"), index=False)
print("Submission file saved at", os.path.join(output_dir, "submission_2.csv"))
