import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from math import sqrt
import os

random_state = 2024

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 合并数据并处理
all_data = pd.concat((train, test), ignore_index=True)
all_data = pd.get_dummies(all_data)  # One-Hot编码
all_data = all_data.fillna(all_data.mean())  # 填充缺失值

# 数据切分
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y_train = train['SalePrice']
print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)


# 包装器类
class SklearnWrapper:
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf):
    oof_train = np.zeros((x_train.shape[0],))  # 存储训练集的Out-Of-Fold预测结果
    oof_test = np.zeros((x_test.shape[0],))  # 存储测试集的Out-Of-Fold预测结果
    oof_test_skf = np.empty((5, x_test.shape[0]))  # 存储每一折交叉验证的测试集预测结果

    for i, (train_idx, valid_idx) in enumerate(kf.split(x_train, y_train)):
        trn_x, val_x = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        trn_y, val_y = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        clf.train(trn_x, trn_y)

        oof_train[valid_idx] = clf.predict(val_x)  # 预测验证集
        oof_test_skf[i, :] = clf.predict(x_test)  # 预测测试集
        # print(f"Validation predictions for fold {i}: {oof_train[valid_idx]}")

    oof_test[:] = oof_test_skf.mean(axis=0)  # 对5折预测结果求均值
    print(f"Model {clf.clf.__class__.__name__} RMSE: {sqrt(mean_squared_error(y_train, oof_train))}")
    # return oof_train, oof_test
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 模型参数
et_params = {'n_estimators': 100, 'max_features': 0.5, 'max_depth': 12, 'min_samples_leaf': 2}
rf_params = {'n_estimators': 100, 'max_features': 0.2, 'max_depth': 12, 'min_samples_leaf': 2}
rd_params = {'alpha': 0.01}
ls_params = {'alpha': 0.005}

# 初始化模型
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=random_state, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=random_state, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=random_state, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=random_state, params=ls_params)

# 获取模型输出
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)
print("ExtraTreesRegressor predictions:", et_oof_test[:10])
print("RandomForestRegressor predictions:", rf_oof_test[:10])
print("Ridge predictions:", rd_oof_test[:10])
print("Lasso predictions:", ls_oof_test[:10])


# 模型融合
def stack_model(oof_train_list, oof_test_list, y):
    train_stack = np.hstack(oof_train_list)  # 水平堆叠
    test_stack = np.hstack(oof_test_list)
    print(f"Train shape: {train_stack.shape}, Test shape: {test_stack.shape}")

    oof = np.zeros(train_stack.shape[0])  # 存储最终堆叠模型的训练集OOF结果
    predictions = np.zeros(test_stack.shape[0])  # 存储最终堆叠模型的测试集预测结果
    scores = []  # 用于保存每一折交叉验证的RMSE分数

    for fold_, (trn_idx, val_idx) in enumerate(kf.split(train_stack, y)):
        trn_data, trn_y = train_stack[trn_idx], y.iloc[trn_idx]
        val_data, val_y = train_stack[val_idx], y.iloc[val_idx]

        clf = Ridge(random_state=random_state)  # 使用Ridge回归作为堆叠模型的学习器
        clf.fit(trn_data, trn_y)

        oof[val_idx] = clf.predict(val_data)  # 预测验证集
        predictions += clf.predict(test_stack) / kf.n_splits  # 预测测试集，累加结果（进行5折平均）

        score = sqrt(mean_squared_error(val_y, oof[val_idx]))  # 计算当前折的RMSE
        scores.append(score)
        print(f'Fold {fold_ + 1}/{kf.n_splits}, RMSE: {score:.6f}')

    print(f'Mean RMSE: {np.mean(scores):.6f}')
    return oof, predictions


# 融合训练
stacked_oof, stacked_predictions = stack_model(
    [et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train],
    [et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test],
    y_train
)
# stacked_oof, stacked_predictions = stack_model(
#     [et_oof_train, rf_oof_train, ls_oof_train],
#     [et_oof_test, rf_oof_test, ls_oof_test],
#     y_train
# )
print("Sample predictions from stacked model:")
print(stacked_predictions[:10])

# 生成Kaggle提交文件
output_dir = "output/stack_model"
os.makedirs(output_dir, exist_ok=True)
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': stacked_predictions.flatten()
})
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
print("Submission file saved at", os.path.join(output_dir, "submission.csv"))
