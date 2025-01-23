# 该类是在比赛后重构的，希望能够更好、更便捷、更实用地处理数据

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder


# 更便捷的数据描述类
class DescData:
    def __init__(self, df):
        self.df = df

    # 懒人包，
    def auto_desc(self, df=None):
        if df is not None:
            self.df = df
            print(f"[Warning] 将self.df更新为传入的df")
        print(f"数据集的形状为：{self.df.shape}")
        stats = []
        for col in self.df.columns:
            stats.append((col, self.df[col].nunique(), self.df[col].isnull().sum() * 100 / self.df.shape[0],
                          self.df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                          self.df[col].dtype))
        stats_df = pd.DataFrame(stats, columns=['Feature', 'n_unique', '缺失值占比(%)',
                                                '最大类别占比(%)', 'type'])
        stats_df.sort_values('缺失值占比(%)', ascending=False)
        return stats_df

    # 查看含有缺失值的列与缺失值数量
    def desc_missing_values(self, df=None):
        if df is not None:
            self.df = df
            print(f"[Warning] 将self.df更新为传入的df")
        missing_values = self.df.isnull().sum()
        missing_percentage = self.df.isnull().mean()
        missing_info = pd.DataFrame({
            'count': missing_values,
            'rate': missing_percentage
        })
        missing_info = missing_info[missing_info['count'] > 0].sort_values(by='count', ascending=False)
        # 检查是不是空的dataframe
        if missing_info.empty:
            return "无缺失值"
        else:
            return missing_info


# 可视化数据分布
def describe_and_visualize(df, table_name=None, exclude_columns=None):
    print("\n[log] -------------------- \n")
    if table_name is not None:
        print(f"正在分析表格：{table_name}")
    print(f"表格的形状：{df.shape}")

    cat_columns = []  # 类别型字段
    num_columns = []  # 数值型字段

    # 分类字段与数值字段区分
    for column in df.columns:
        if exclude_columns is not None and column in exclude_columns:
            continue
        if df[column].dtype == 'object' or df[column].nunique() < 30:  # 类别型或唯一值较少
            cat_columns.append(column)
        else:
            num_columns.append(column)

    # 输出类别型字段的信息
    print(f"\n>>> 类别型字段（{len(cat_columns)} 个）：{cat_columns}")
    for column in cat_columns:
        print(f"字段：{column}")
        print(f"{column}是类别型数据，共有{df[column].nunique()}个不同的值")
        print(df[column].value_counts())

    # 绘制类别型字段的条形图
    if cat_columns:
        print("\n[log] 正在绘制类别型字段的统计图...")
        rows = math.ceil(len(cat_columns) / 4)
        fig, axes = plt.subplots(rows, min(len(cat_columns), 4), figsize=(20, 5 * rows),
                                 squeeze=False)  # squeeze=False 确保始终返回二维数组，这样flatten就不会出错
        axes = axes.flatten()  # 展平方便处理
        for i, column in enumerate(cat_columns):
            df[column].value_counts().plot(kind='bar', ax=axes[i], title=f'{column} - Count Values')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
        # 隐藏多余的子图
        for j in range(len(cat_columns), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    # 输出数值型字段的信息
    print(f"\n>>> 数值型字段（{len(num_columns)} 个）：{num_columns}")
    for column in num_columns:
        print(f"\n字段：{column}")
        print(f"{column}是数值型数据，共有{df[column].nunique()}个不同的值")
        print(df[column].describe())

    # 绘制数值型字段的分布图
    if num_columns:
        print("\n[log] 正在绘制数值型字段的分布图...")
        rows = math.ceil(len(num_columns) / 4)
        fig, axes = plt.subplots(rows, min(len(num_columns), 4), figsize=(20, 5 * rows),
                                 squeeze=False)  # squeeze=False 确保始终返回二维数组，这样flatten就不会出错
        axes = axes.flatten()  # 展平方便处理
        for i, column in enumerate(num_columns):
            sns.histplot(df[column], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f'{column} - Distribution')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
        # 隐藏多余的子图
        for j in range(len(num_columns), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()


# 检测特征分布的差异(训练集与测试集)
def plot_feature_distributions(train_df, test_df, include_features=None, exclude_features=None):
    """
    特征分布对比函数
    train_df : 训练集DataFrame
    test_df : 测试集DataFrame
    include_features : 需要分析的特征列表，默认为None，表示使用所有特征
    exclude_features : 不需要分析的特征列表，默认为None，表示不排除任何特征

    """
    # 确定需要分析的列
    common_cols = list(set(train_df.columns) & set(test_df.columns))
    if include_features is None:
        include_features = common_cols
    else:
        include_features = [f for f in include_features if f in common_cols]
    if exclude_features is not None:
        include_features = [f for f in include_features if f not in exclude_features]
    print(f"分析的{len(include_features)}个特征：{include_features}")

    # 分类特征判断逻辑
    def get_column_type(series):
        if series.dtype == 'object' or series.nunique() < 30:
            return 'categorical'
        return 'numerical'

    # 遍历每个特征
    for feature in include_features:
        print(f"\n[Feature] {feature} >>> ", end='')
        plt.figure()

        train_vals = train_df[feature].dropna()
        test_vals = test_df[feature].dropna()

        # 判断特征类型
        data_type = get_column_type(train_df[feature])

        # 数值型特征处理
        if data_type == 'numerical':
            # 使用直方图替代KDE
            bins = min(50, int(np.sqrt(len(train_vals))))

            plt.hist(train_vals, bins=bins, density=True,
                     alpha=0.5, label='Train', color='blue')
            plt.hist(test_vals, bins=bins, density=True,
                     alpha=0.5, label='Test', color='orange')

            # 计算统计指标
            ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
            num_log = f'[{feature}] KS={ks_stat:.6f} (p={ks_p:.6f})'
            # KS < 0.1 -> 分布差异较小，p ≥ 0.05 -> 接受原假设(认为分布可能相同)。反之则拒绝原假设(认为分布不同)
            if ks_p < 0.05:
                print(f"Warning: {feature} distributions are different (p={ks_p:.6f})")
            plt.title(num_log)
            print(num_log)

        # 分类型特征处理
        else:
            # 计算类别比例
            train_counts = train_vals.value_counts(normalize=True).sort_index()
            test_counts = test_vals.value_counts(normalize=True).sort_index()

            # 统一索引
            all_cats = train_counts.index.union(test_counts.index)
            train_counts = train_counts.reindex(all_cats, fill_value=0)
            test_counts = test_counts.reindex(all_cats, fill_value=0)

            # 使用折线图替代条形图
            plt.plot(train_counts.index.astype(str), train_counts.values,
                     label='Train', alpha=0.7)
            plt.plot(test_counts.index.astype(str), test_counts.values,
                     label='Test', alpha=0.7)

            # 计算统计指标
            js_div = jensenshannon(train_counts, test_counts) ** 2
            cal_log = f'[{feature}] JS Div={js_div:.6f}'
            # JS Divergence < 0.1 -> 分布差异较小。反之则分布差异较大
            if js_div > 0.1:
                print(f"Warning: {feature} distributions are different (JS Div={js_div:.6f})")
            plt.title(cal_log)
            plt.xticks(rotation=45)
            print(cal_log)

        # 通用设置
        plt.xlabel(feature)
        plt.ylabel('Density' if data_type == 'numerical' else 'Proportion')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    DATA_PATH = 'G:/DataSets/kaggle/Elo Merchant Category Recommendation'

    # 加载数据
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    test = pd.read_csv(f"{DATA_PATH}/test.csv")
    # desc = DescData(train)
    # print(desc.auto_desc())
    # print(desc.desc_missing_values())
    # describe_and_visualize(train, 'train')
    plot_feature_distributions(train, test, exclude_features=['card_id'])
