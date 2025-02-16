# 本代码消耗内存过大，暂时未能成功运行
import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DATA_PATH = 'G:/DataSets/kaggle/Elo Merchant Category Recommendation'
# 配置参数
config = {
    'data_path': {
        'train': f"{DATA_PATH}/train.csv",
        'test': f"{DATA_PATH}/test.csv",
        'merchant': f"{DATA_PATH}/merchants.csv",
        'new_trans': f"{DATA_PATH}/new_merchant_transactions.csv",
        'hist_trans': f"{DATA_PATH}/historical_transactions.csv"
    },
    'output_path': {
        'dict': f"{DATA_PATH}/preprocess/features_dict",
        'groupby': f"{DATA_PATH}/preprocess/features_groupby",
        'nlp': f"{DATA_PATH}/preprocess/features_nlp"
    },
    'nlp_features': ['merchant_id', 'merchant_category_id', 'state_id', 'subsector_id', 'city_id']
}


def timer(func):
    """计时装饰器"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] 耗时: {time.time() - start:.2f}s")
        return result

    return wrapper


def load_data():
    """加载原始数据"""
    print("\n====== 加载原始数据 ======")
    data = {
        'train': pd.read_csv(config['data_path']['train']),
        'test': pd.read_csv(config['data_path']['test']),
        'merchant': pd.read_csv(config['data_path']['merchant']),
        'new_trans': pd.read_csv(config['data_path']['new_trans']),
        'hist_trans': pd.read_csv(config['data_path']['hist_trans'])
    }
    print(f"训练集: {data['train'].shape}, 测试集: {data['test'].shape}")
    print(f"商户数据: {data['merchant'].shape}, 新交易数据: {data['new_trans'].shape}, "
          f"交易数据: {data['hist_trans'].shape}")
    return data


def preprocess_merchant(merchant):
    """商户数据预处理"""
    print("\n====== 处理商户数据 ======")

    # 列分类
    category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                     'subsector_id', 'category_1', 'most_recent_sales_range',
                     'most_recent_purchases_range', 'category_4', 'city_id',
                     'state_id', 'category_2']
    numeric_cols = ['numerical_1', 'numerical_2', 'avg_sales_lag3',
                    'avg_purchases_lag3', 'active_months_lag3', 'avg_sales_lag6',
                    'avg_purchases_lag6', 'active_months_lag6', 'avg_sales_lag12',
                    'avg_purchases_lag12', 'active_months_lag12']

    # 离散字段编码
    for col in ['category_1', 'most_recent_sales_range',
                'most_recent_purchases_range', 'category_4']:
        merchant[col] = change_object_cols(merchant[col])

    # 缺失值处理
    merchant[category_cols] = merchant[category_cols].fillna(-1)

    # 处理无穷值
    inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
    merchant[inf_cols] = merchant[inf_cols].replace(
        np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())

    # 数值列填充
    for col in numeric_cols:
        merchant[col] = merchant[col].fillna(merchant[col].mean())

    # 去除重复列
    merchant = merchant.drop(['merchant_category_id', 'subsector_id', 'category_1',
                              'city_id', 'state_id', 'category_2'], axis=1)
    return merchant.drop_duplicates('merchant_id').reset_index(drop=True)


def preprocess_transaction(transaction):
    """交易数据预处理"""
    print("\n====== 处理交易数据 ======")

    # 时间特征提取
    transaction['purchase_month'] = transaction['purchase_date'].apply(
        lambda x: '-'.join(x.split(' ')[0].split('-')[:2]))
    transaction['purchase_hour_section'] = transaction['purchase_date'].apply(
        lambda x: int(x.split(' ')[1].split(':')[0]) // 6)
    transaction['purchase_day'] = transaction['purchase_date'].apply(
        lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday() // 5)

    # 离散字段处理
    for col in ['authorized_flag', 'category_1', 'category_3']:
        transaction[col] = change_object_cols(transaction[col].fillna(-1).astype(str))

    # 类型转换
    transaction['category_2'] = transaction['category_2'].astype(int)
    transaction['purchase_month'] = change_object_cols(
        transaction['purchase_month'].fillna(-1).astype(str))

    return transaction.drop('purchase_date', axis=1)


@timer
def generate_dict_features(data):
    """生成字典统计特征"""
    print("\n====== 生成字典特征 ======")

    # 合并交易数据
    trans = pd.concat([data['new_trans'], data['hist_trans']], ignore_index=True)
    trans = pd.merge(trans, data['merchant'][['merchant_id', 'most_recent_sales_range',
                                              'most_recent_purchases_range', 'category_4']],
                     on='merchant_id', how='left')

    # 特征字典初始化
    features = {card: {} for card in pd.concat([data['train']['card_id'],
                                                data['test']['card_id']]).unique()}

    # 特征遍历统计
    category_cols = ['authorized_flag', 'city_id', 'category_1', 'category_3',
                     'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                     'most_recent_purchases_range', 'category_4', 'purchase_month',
                     'purchase_hour_section', 'purchase_day']
    numeric_cols = ['purchase_amount', 'installments']

    for _, row in trans.iterrows():
        card = row['card_id']
        for cate_col in category_cols:
            for num_col in numeric_cols:
                key = f"{cate_col}&{row[cate_col]}&{num_col}"
                features[card][key] = features[card].get(key, 0) + row[num_col]

    # 转换为DataFrame并保存
    df = pd.DataFrame(features).T.reset_index()
    df.columns = ['card_id'] + list(df.columns[1:])

    data['train'] = pd.merge(data['train'], df, on='card_id', how='left')
    data['test'] = pd.merge(data['test'], df, on='card_id', how='left')

    data['train'].to_csv(f"{config['output_path']['dict']}_train.csv", index=False)
    data['test'].to_csv(f"{config['output_path']['dict']}_test.csv", index=False)
    print(f"字典特征生成完成，训练集形状: {data['train'].shape}")


@timer
def generate_groupby_features(data):
    """生成GroupBy聚合特征"""
    print("\n====== 生成聚合特征 ======")

    trans = pd.concat([data['new_trans'], data['hist_trans']], ignore_index=True)
    trans = pd.merge(trans, data['merchant'][['merchant_id', 'most_recent_sales_range',
                                              'most_recent_purchases_range', 'category_4']],
                     on='merchant_id', how='left')

    # 差异特征
    trans['purchase_day_diff'] = trans.groupby("card_id")['purchase_day'].diff()
    trans['purchase_month_diff'] = trans.groupby("card_id")['purchase_month'].diff()

    # 聚合配置
    aggs = {
        'numeric': ['nunique', 'mean', 'min', 'max', 'var', 'skew', 'sum'],
        'categorical': ['nunique']
    }

    # 历史交易特征
    hist = trans[trans['month_lag'] < 0].groupby('card_id').agg(aggs)
    hist.columns = [f"{col}_hist" for col in hist.columns]

    # 新交易特征
    new = trans[trans['month_lag'] >= 0].groupby('card_id').agg(aggs)
    new.columns = [f"{col}_new" for col in new.columns]

    # 合并特征
    df = pd.merge(hist, new, on='card_id', how='outer')
    df.to_csv(f"{config['output_path']['groupby']}.csv", index=False)
    print(f"聚合特征生成完成，形状: {df.shape}")


@timer
def generate_nlp_features(data):
    """生成文本特征"""
    print("\n====== 生成文本特征 ======")

    trans = pd.concat([data['new_trans'], data['hist_trans']], ignore_index=True)

    # 文本特征生成
    vectorizers = {
        'count': CountVectorizer(),
        'tfidf': TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                 use_idf=True, smooth_idf=True, sublinear_tf=True)
    }

    for feat in config['nlp_features']:
        print(f"处理特征: {feat}")
        trans[feat] = trans[feat].astype(str)

        # 分别生成历史、新、全部交易文本
        for scope in ['_hist', '_new', '_all']:
            col_name = feat + scope
            temp = trans.groupby("card_id")[feat].apply(
                lambda x: ' '.join(x) if scope == '_all' else
                ' '.join(x[trans['month_lag'] < 0]) if scope == '_hist' else
                ' '.join(x[trans['month_lag'] >= 0])
            ).reset_index(name=col_name)

            data['train'] = pd.merge(data['train'], temp, how='left', on='card_id')
            data['test'] = pd.merge(data['test'], temp, how='left', on='card_id')

    # 向量化处理
    train_text = data['train'][[f"{f}_{s}" for f in config['nlp_features']
                                for s in ['hist', 'new', 'all']]]
    test_text = data['test'][[f"{f}_{s}" for f in config['nlp_features']
                              for s in ['hist', 'new', 'all']]]

    for name, vec in vectorizers.items():
        vec.fit(pd.concat([train_text.values.flatten(), test_text.values.flatten()]))
        X_train = vec.transform(train_text.apply(lambda x: ' '.join(x), axis=1))
        X_test = vec.transform(test_text.apply(lambda x: ' '.join(x), axis=1))

        sparse.save_npz(f"{config['output_path']['nlp']}_{name}_train.npz", X_train)
        sparse.save_npz(f"{config['output_path']['nlp']}_{name}_test.npz", X_test)
    print(f"文本特征保存完成，训练形状: {X_train.shape}")


def change_object_cols(se):
    """离散字段编码"""
    value = sorted(se.unique().tolist())
    return se.map(pd.Series(range(len(value)), index=value)).values


if __name__ == "__main__":
    raw_data = load_data()

    # 数据预处理
    raw_data['merchant'] = preprocess_merchant(raw_data['merchant'])
    raw_data['transaction'] = preprocess_transaction(pd.concat([raw_data['new_trans'],
                                                                raw_data['hist_trans']], ignore_index=True))

    # 特征生成
    generate_dict_features(raw_data)
    generate_groupby_features(raw_data)
    generate_nlp_features(raw_data)

    print("\n====== 特征工程全部完成 ======")
