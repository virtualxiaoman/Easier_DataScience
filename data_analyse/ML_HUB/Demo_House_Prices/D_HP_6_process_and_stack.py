import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from scipy.stats import zscore, boxcox
import xgboost as xgb
import lightgbm as lgb


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(f"[log] train shape: {train.shape}, test shape: {test.shape}")
    return train, test


class DataProcessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.train_len = train.shape[0]

    def _init_data(self):
        self.data = pd.concat([self.train, self.test], axis=0, sort=False)
        missing_cols = [c for c in self.data if self.data[c].isna().mean() * 100 > 50]  # 删除缺失值比例大于50%的特征列
        self.data = self.data.drop(missing_cols, axis=1)

        self.object_df = self.data.select_dtypes(include=['object'])
        self.numerical_df = self.data.select_dtypes(exclude=['object'])

        self.object_df = self.object_df.fillna('unknown')  # 用unknown填充
        missing_cols = [c for c in self.numerical_df if self.numerical_df[c].isna().sum() > 0]
        for c in missing_cols:
            self.numerical_df[c] = self.numerical_df[c].fillna(self.numerical_df[c].median())  # 用中位数填充
        self.object_df = self.object_df.drop(['Heating', 'RoofMatl', 'Condition2', 'Street', 'Utilities'],
                                             axis=1)  # 删除类别比不均衡的特征（对分数没影响）

    def process_num_data(self):
        # 将销售日期小于建造日期的数据的销售日期改为2009(销售日期的最大值)
        self.numerical_df.loc[self.numerical_df['YrSold'] < self.numerical_df['YearBuilt'], 'YrSold'] = 2009
        self.numerical_df['Age_House'] = (self.numerical_df['YrSold'] - self.numerical_df['YearBuilt'])  # 计算房屋的年龄
        # 对浴池求和得到地下室的总浴室数
        self.numerical_df['TotalBsmtBath'] = self.numerical_df['BsmtFullBath'] + self.numerical_df['BsmtHalfBath'] * 0.5
        # 对浴池求和得到地上的总浴室数
        self.numerical_df['TotalBath'] = self.numerical_df['FullBath'] + self.numerical_df['HalfBath'] * 0.5
        # 计算总面积
        self.numerical_df['TotalSA'] = self.numerical_df['TotalBsmtSF'] + self.numerical_df['1stFlrSF'] + \
                                       self.numerical_df['2ndFlrSF']

        train_num_df = self.numerical_df[:self.train_len]
        test_num_df = self.numerical_df[self.train_len:]
        train_object_df = self.object_df[:self.train_len]
        test_object_df = self.object_df[self.train_len:]
        # 对train部分进行异常值检测，删除异常值
        z_scores = np.abs(zscore(train_num_df))
        valid_rows = (z_scores < 8).all(axis=1)  # 只保留Z-Score小于3的数据
        train_num_df = train_num_df.loc[valid_rows]
        train_object_df = train_object_df.loc[valid_rows]
        # 获取train的长度
        self.train_len = train_num_df.shape[0]
        self.numerical_df = pd.concat([train_num_df, test_num_df], axis=0, sort=False)
        self.object_df = pd.concat([train_object_df, test_object_df], axis=0, sort=False)

        # # 对train部分进行异常值检测，删除异常值
        # train_indices = self.data.index[:self.train_len]  # 获取train部分的索引
        # z_scores = np.abs(zscore(self.numerical_df.loc[train_indices]))  # 只计算train部分的z-score
        # valid_rows = (z_scores < 3).all(axis=1)  # 只保留Z-Score小于3的数据
        #
        # # 检查valid_rows中True值的索引是否存在于train_indices中
        # assert np.all(np.isin(self.numerical_df.loc[train_indices].index[valid_rows], train_indices))
        #
        # # 根据valid_rows筛选train数据
        # self.numerical_df = self.numerical_df.loc[valid_rows]
        # self.object_df = self.object_df.loc[valid_rows]

    def process_obj_data(self):
        bin_map = {'TA': 2, 'Gd': 3, 'Fa': 1, 'Ex': 4, 'Po': 1, 'None': 0,
                   'Y': 1, 'N': 0, 'Reg': 3, 'IR1': 2, 'IR2': 1,
                   'IR3': 0, "None": 0, "No": 2, "Mn": 2,
                   "Av": 3, "Gd": 4, "Unf": 1, "LwQ": 2,
                   "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
        object_df = self.object_df
        object_df['ExterQual'] = object_df['ExterQual'].map(bin_map)
        object_df['ExterCond'] = object_df['ExterCond'].map(bin_map)
        object_df['BsmtCond'] = object_df['BsmtCond'].map(bin_map)
        object_df['BsmtQual'] = object_df['BsmtQual'].map(bin_map)
        object_df['HeatingQC'] = object_df['HeatingQC'].map(bin_map)
        object_df['KitchenQual'] = object_df['KitchenQual'].map(bin_map)
        object_df['FireplaceQu'] = object_df['FireplaceQu'].map(bin_map)
        object_df['GarageQual'] = object_df['GarageQual'].map(bin_map)
        object_df['GarageCond'] = object_df['GarageCond'].map(bin_map)
        object_df['CentralAir'] = object_df['CentralAir'].map(bin_map)
        object_df['LotShape'] = object_df['LotShape'].map(bin_map)
        object_df['BsmtExposure'] = object_df['BsmtExposure'].map(bin_map)
        object_df['BsmtFinType1'] = object_df['BsmtFinType1'].map(bin_map)
        object_df['BsmtFinType2'] = object_df['BsmtFinType2'].map(bin_map)

        PavedDrive = {"N": 0, "P": 1, "Y": 2}
        object_df['PavedDrive'] = object_df['PavedDrive'].map(PavedDrive)
        # 选择剩余的object特征
        rest_object_columns = object_df.select_dtypes(include=['object'])
        # 进行one-hot编码
        object_df = pd.get_dummies(object_df, columns=rest_object_columns.columns)
        # 众数填充缺失值
        object_df = object_df.fillna(object_df.mode().iloc[0])
        self.object_df = object_df

    def auto_run(self):
        self._init_data()
        self.process_num_data()
        self.process_obj_data()
        self.data = pd.concat([self.object_df, self.numerical_df], axis=1, sort=False)
        # self.data.reset_index(drop=True, inplace=True)


# def process_data(train, test):
#     data = pd.concat([train, test], axis=0, sort=False)
#     missing_cols = [c for c in data if data[c].isna().mean() * 100 > 50]  # 删除缺失值比例大于50%的特征列
#     data = data.drop(missing_cols, axis=1)
#
#     object_df = data.select_dtypes(include=['object'])
#     numerical_df = data.select_dtypes(exclude=['object'])
#     object_df = object_df.fillna('unknown')  # 用unknown填充
#     missing_cols = [c for c in numerical_df if numerical_df[c].isna().sum() > 0]
#     for c in missing_cols:
#         numerical_df[c] = numerical_df[c].fillna(numerical_df[c].median())  # 用中位数填充
#
#     object_df = object_df.drop(['Heating', 'RoofMatl', 'Condition2', 'Street', 'Utilities'],
#                                axis=1)  # 删除类别比不均衡的特征（对分数没影响）
#
#     # 将销售日期小于建造日期的数据的销售日期改为2009(销售日期的最大值)
#     numerical_df.loc[numerical_df['YrSold'] < numerical_df['YearBuilt'], 'YrSold'] = 2009
#     numerical_df['Age_House'] = (numerical_df['YrSold'] - numerical_df['YearBuilt'])  # 计算房屋的年龄
#     # 对浴池求和得到地下室的总浴室数
#     numerical_df['TotalBsmtBath'] = numerical_df['BsmtFullBath'] + numerical_df['BsmtHalfBath'] * 0.5
#     # 对浴池求和得到地上的总浴室数
#     numerical_df['TotalBath'] = numerical_df['FullBath'] + numerical_df['HalfBath'] * 0.5
#     # 计算总面积
#     numerical_df['TotalSA'] = numerical_df['TotalBsmtSF'] + numerical_df['1stFlrSF'] + numerical_df['2ndFlrSF']
#
#     bin_map = {'TA': 2, 'Gd': 3, 'Fa': 1, 'Ex': 4, 'Po': 1, 'None': 0,
#                'Y': 1, 'N': 0, 'Reg': 3, 'IR1': 2, 'IR2': 1,
#                'IR3': 0, "None": 0, "No": 2, "Mn": 2,
#                "Av": 3, "Gd": 4, "Unf": 1, "LwQ": 2,
#                "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
#     object_df['ExterQual'] = object_df['ExterQual'].map(bin_map)
#     object_df['ExterCond'] = object_df['ExterCond'].map(bin_map)
#     object_df['BsmtCond'] = object_df['BsmtCond'].map(bin_map)
#     object_df['BsmtQual'] = object_df['BsmtQual'].map(bin_map)
#     object_df['HeatingQC'] = object_df['HeatingQC'].map(bin_map)
#     object_df['KitchenQual'] = object_df['KitchenQual'].map(bin_map)
#     object_df['FireplaceQu'] = object_df['FireplaceQu'].map(bin_map)
#     object_df['GarageQual'] = object_df['GarageQual'].map(bin_map)
#     object_df['GarageCond'] = object_df['GarageCond'].map(bin_map)
#     object_df['CentralAir'] = object_df['CentralAir'].map(bin_map)
#     object_df['LotShape'] = object_df['LotShape'].map(bin_map)
#     object_df['BsmtExposure'] = object_df['BsmtExposure'].map(bin_map)
#     object_df['BsmtFinType1'] = object_df['BsmtFinType1'].map(bin_map)
#     object_df['BsmtFinType2'] = object_df['BsmtFinType2'].map(bin_map)
#
#     PavedDrive = {"N": 0, "P": 1, "Y": 2}
#     object_df['PavedDrive'] = object_df['PavedDrive'].map(PavedDrive)
#     # 选择剩余的object特征
#     rest_object_columns = object_df.select_dtypes(include=['object'])
#     # 进行one-hot编码
#     object_df = pd.get_dummies(object_df, columns=rest_object_columns.columns)
#     # 众数填充缺失值
#     object_df = object_df.fillna(object_df.mode().iloc[0])
#
#     data = pd.concat([object_df, numerical_df], axis=1, sort=False)  # 将处理后的数据合并
#
#     return data


train_data, test_data = load_data()
# data = process_data(train_data, test_data)
data_p = DataProcessor(train_data, test_data)
data_p.auto_run()
data = data_p.data
train_len = data_p.train_len

X_train = data[:train_len].drop(['SalePrice', 'Id'], axis=1)
X_test = data[train_len:].drop(['SalePrice', 'Id'], axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = data[:train_len]['SalePrice']
# y_train = np.log1p(y_train)  # 对数变换, ln(y+1)
y_train, lambda_ = boxcox(y_train)  # 对 y 进行 Box-Cox 变换
print(f"lambda: {lambda_}")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")


# X_train shape: (1168, 287), X_val shape: (292, 287)
# y_train shape: (1168,), y_val shape: (292,)
# X_test shape: (1459, 287)

# # 标准化数据（对线性模型特别重要）
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)


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

    def fit_and_predict(self, X_train, y_train, X_val, y_val, X_test):
        """
        训练堆叠模型。
        """
        # 训练基础学习器
        for name, model in self.base_learners:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            mse = mean_squared_error(y_val, pred)
            print(f'[eval] 基模型{name}, RMSE: {np.sqrt(mse)}')

        self.stacking_model.fit(X_train, y_train)
        print("[log] 正在训练Stacking模型")

        # 评估Stacking模型
        stacking_pred = self.stacking_model.predict(X_val)
        stacking_mse = mean_squared_error(y_val, stacking_pred)
        print(f'>>> [eval] Stacking RMSE: {np.sqrt(stacking_mse)}')

        # 对测试集进行预测
        stacked_predictions = self.stacking_model.predict(X_test)
        print("[log] Stack pred:", stacked_predictions[:5])
        print(f"[log] Stacked predictions shape: {stacked_predictions.shape}")  # (1459,)

        return stacked_predictions


# 定义基础模型
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13))
lasso = LassoCV()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1)  # -1表示不输出训练信息
svr = SVR(kernel='rbf')

# 训练模型
base_learners = [
    ('ridge', ridge),
    ('lasso', lasso),
    ('rf', rf),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('svr', svr)
]
stacked_model = StackedModelRegressor(base_learners)
stacked_predictions = stacked_model.fit_and_predict(X_train, y_train, X_val, y_val, X_test)
# stacked_predictions = np.expm1(stacked_predictions)  # 反向对数变换， e^x - 1
# 逆变换公式：y = (y' * lambda + 1)^(1/lambda)
# 如果 lambda == 0，使用 exp 逆变换（即对数逆变换）
if lambda_ != 0:
    stacked_predictions = (stacked_predictions * lambda_ + 1) ** (1 / lambda_)
else:
    stacked_predictions = np.expm1(stacked_predictions)


def generate_submission(id_col, pred_col, submission_name="submission"):
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


generate_submission(test_data['Id'], stacked_predictions, submission_name="submission_2")
