# Elo Merchant Category Recommendation
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

DATA_PATH = 'G:/DataSets/kaggle/Elo Merchant Category Recommendation'

# 加载数据
train = pd.read_csv(f"{DATA_PATH}/train.csv")
test = pd.read_csv(f"{DATA_PATH}/test.csv")
print(f"Train data shape: {train.shape}, Test data shape: {test.shape}")


# 预处理时间字段
def preprocess_time(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    return df


train = preprocess_time(train)
test = preprocess_time(test)

# 定义特征和标签
features = ['feature_1', 'feature_2', 'feature_3', 'year', 'month']
X = train[features]
y = train['target']
X_test = test[features]

# LightGBM参数配置
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# 初始化KFold
folds = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))  # 存储测试集预测结果

# 交叉验证训练
for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"\n------ Fold {fold + 1} ------")

    # 划分当前fold的训练集和验证集
    X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # 创建Dataset
    train_data = lgb.Dataset(X_trn, label=y_trn)
    val_data = lgb.Dataset(X_val, label=y_val)

    # 回调函数，在新版本的LightGBM中需要使用回调函数来实现早停
    # 参考：https://blog.csdn.net/weixin_51723388/article/details/124578560
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),  # 早停设置，相当于early_stopping_rounds=100
        lgb.log_evaluation(period=50)  # 每50轮输出一次日志，相当于verbose_eval=50
    ]

    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks
    )

    # 预测测试集并累加
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

# 生成提交文件
submission = pd.DataFrame({
    'card_id': test['card_id'],
    'target': test_preds
})
submission.to_csv('output/submission_quick_start.csv', index=False)

print("\nSubmission file saved as submission_quick_start.csv")
