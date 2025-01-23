# House Price - Advanced Regression Techniques

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"train shape: {train.shape}, test shape: {test.shape}")

# 数据预处理：合并数据，处理类别变量，填充缺失值
all_data = pd.concat([train, test], ignore_index=True)
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
print(f"all_data shape: {all_data.shape}")

# 切分数据
X_train = all_data[:train.shape[0]].drop(['SalePrice', 'Id'], axis=1)
X_test = all_data[train.shape[0]:].drop(['SalePrice', 'Id'], axis=1)
y_train = train['SalePrice']

# 设置LightGBM模型参数
params = {
    'num_leaves': 63,
    'min_child_samples': 50,
    'objective': 'regression',
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'verbose': -1,
}

# KFold交叉验证设置
folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(X_train.shape[0])  # 存储每一折的验证集预测值，oof: out-of-fold（每个训练样本的验证集预测值）
test_preds = np.zeros(X_test.shape[0])  # 存储测试集的平均预测值（所有折叠的预测结果）

# 每次分割返回训练集索引 trn_idx 和验证集索引 val_idx
for trn_idx, val_idx in folds.split(X_train, y_train):
    trn_df, trn_label = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
    val_df, val_label = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # 创建LightGBM数据集
    dtrn = lgb.Dataset(trn_df, label=trn_label)
    dval = lgb.Dataset(val_df, label=val_label)

    # 训练模型
    bst = lgb.train(params, dtrn,
                    num_boost_round=1000,
                    valid_sets=[dtrn, dval])

    # 预测。num_iteration=bst.best_iteration 是告诉LightGBM在训练过程中使用表现最好的迭代次数（通过验证集上的评估确定）
    oof_preds[val_idx] = bst.predict(val_df, num_iteration=bst.best_iteration)
    test_preds += bst.predict(X_test, num_iteration=bst.best_iteration) / folds.n_splits  # 这里folds.n_splits=5
    print(f"Fold RMSE: {np.sqrt(mean_squared_error(val_label, oof_preds[val_idx]))}")

# 输出训练集上的RMSE评分
rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f'Overall RMSE on training data: {rmse:.4f}')

# 生成Kaggle提交文件
submission = pd.DataFrame({
    'Id': test['Id'],  # 保持测试集的ID列
    'SalePrice': test_preds  # 预测的结果列
})
submission.to_csv("output/quick_start/submission.csv", index=False)
