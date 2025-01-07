from pprint import pprint

import pandas as pd
import numpy as np
from math import sqrt
import operator


def preprocess_data(data):
    """
    预处理数据：
      1. 筛选点击记录
      2. 去重，确保每个用户对某广告组只保留一条点击记录
      3. 选择需要的属性列
    返回值是：
             user_id  adgroup_id  clk
        22    642854         102    1
        23    443793         102    1
        48    355080         102    1
        84    843732         102    1
        191  1137518         102    1
    :param data: 原始数据
    :return: 处理后的数据，dataframe格式，三列'user_id', 'adgroup_id', 'clk'
    """
    # 筛选有点击的数据
    data_clk = data[data['clk'] == 1].copy()
    # 根据 'user_id', 'adgroup_id'去重
    data_clk.drop_duplicates(subset=['user_id', 'adgroup_id'], keep='first', inplace=True)
    # 选取 'user_id', 'adgroup_id', 'clk' 三列
    data_clk = data_clk[['user_id', 'adgroup_id', 'clk']]
    # print(data_clk.head())

    return data_clk


# -------------------数据集构建-------------------
def build_dataset_from_csv(file_path):
    """
    从csv文件构建数据集。数据集以字典的形式存储，键为用户id，值为广告组id和点击次数的字典。返回值是：
    {
     '1000001': {'26565': 1},
     '100001': {'792889': 1},
     '1000020': {'447788': 1},
     '1000021': {'355722': 1,
                '373082': 1,
                '387795': 1,
                ...
                 '778120': 1}
    }
    也就是说，用户1000001点击了广告组26565一次，用户100001点击了广告组792889一次，以此类推。
    每个 user_id 映射一个字典，其中保存了该用户所有广告组的点击记录。
    :param file_path: csv文件路径
    :return: 以userid为键，广告组id为值的字典
    """
    dataSet = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过文件的第一行（属性名称行）
        for line in f:
            userid, adgroup_id, clk = line.strip().split(",")
            dataSet.setdefault(userid, {})
            dataSet[userid][adgroup_id] = int(clk)

    return dataSet


# -------------------共现矩阵计算-------------------
def calculate_cooccurrence_matrix(dataSet):
    """
    计算广告共现矩阵。返回值是两个字典：
        N: 喜欢广告i的总人数，key是广告id，value是人数
    {
     '10000': 1,
     '100006': 1,
     '100013': 1,
     '100019': 1,
     '10002': 2,
     '100021': 2,
     '100022': 1,
    }
    如点击广告id=10000的人数是1，点击广告id=10002的人数是2
        C: 喜欢广告i也喜欢广告j的人数，key是广告id，value是字典，字典的key是广告id，value是共现次数
    {
     '10000': {'153247': 1,
               '322827': 1,
               ...
               '749768': 1},
     '100006': {'200633': 1,
                '201408': 1,
                ...
                '826898': 1},
    }
    如点击广告id=10000的人也点击了广告id=153247一次，点击广告id=100006的人也点击了广告id=200633一次。
    注意因为去重了，所以共现次数都是1。

    :param dataSet: 用户点击数据集
    :return: N (广告点击人数) 和 C (广告共现次数)
    """
    N = {}  # N：字典，记录每个广告被点击的总人数
    C = {}  # C：嵌套字典，记录每对广告共同被点击的次数

    for userid, item in dataSet.items():
        # 这里的score都是1，因为是clk=1的记录
        for ad_i, score in item.items():
            N.setdefault(ad_i, 0)
            N[ad_i] += 1  # 如果广告ad_i被用户点击，点击人数加1
            C.setdefault(ad_i, {})

            for ad_j, scores in item.items():
                if ad_j != ad_i:  # 不计算广告自身的共现
                    C[ad_i].setdefault(ad_j, 0)
                    C[ad_i][ad_j] += 1  # 如果用户同时点击了广告ad_i和广告ad_j，那么这两个广告的共现次数就加1

    return N, C


# -------------------相似度计算-------------------
def calculate_similarity(C, N):
    """
    计算广告之间的相似度矩阵
    W[i][j] = C[i][j] / sqrt(N[i] * N[j])，也就是余弦相似度，输出的是一个字典，key是广告i，value是广告j和相似度的字典：
    {
     '10000': {'153247': 1.0,
               '322827': 0.5773502691896258,
               '405680': 0.7071067811865475,
               '503010': 0.7071067811865475,
               '576560': 1.0,
               '588978': 1.0,
               '622652': 0.1690308509457033,
               '648820': 0.5773502691896258,
               '688732': 0.5773502691896258,
               '695765': 0.7071067811865475,
               '749768': 0.5773502691896258},
     '100006': {'200633': 1.0,
                '201408': 0.5773502691896258,
                '245217': 0.5773502691896258,
                '271087': 1.0,
                '380126': 0.3333333333333333,
                '381835': 0.3333333333333333,
                '419298': 0.4472135954999579,
                '438576': 1.0,
                '588084': 0.7071067811865475,
                '628731': 0.2581988897471611,
                '690538': 0.2886751345948129,
                '732522': 0.7071067811865475,
                '754542': 1.0,
                '80301': 0.7071067811865475,
                '826898': 1.0},
    }
    :param C: 广告共现次数矩阵
    :param N: 每个广告的点击人数
    :return: 广告相似度矩阵 W
    """
    W = {}  # 广告的相似度矩阵

    # 遍历每个广告i
    for ad_i, item in C.items():
        W.setdefault(ad_i, {})
        # 遍历广告i的共现广告j
        for ad_j, count in item.items():
            W[ad_i].setdefault(ad_j, 0)
            W[ad_i][ad_j] = C[ad_i][ad_j] / sqrt(N[ad_i] * N[ad_j])  # 使用余弦相似度公式

    return W


# -------------------根据广告相似度进行推荐-------------------
def recommend_by_ad_similarity(W, target_adgroup_id, top_n=10):
    """
    根据广告相似度为某个广告推荐相似广告。
    recommend_df的格式如下，以target_adgroup_id=10000为例：
             相似度
    153247  1.000000
    576560  1.000000
    588978  1.000000
    405680  0.707107
    503010  0.707107
    这表示target_adgroup_id=10000与广告153247、576560、588978的相似度为1.0，与广告405680、503010的相似度为0.707107。
    注意，这与calculate_similarity函数中的W[i][j]的结果是一样的，只是格式不同（这里是dataframe，W是字典）。

      recommend_by_similarity 函数根据广告的相似度矩阵为给定广告推荐相似广告。具体步骤如下：
    1. 获取目标广告的相似度：根据目标广告的 adgroup_id 从相似度矩阵 W 中提取相似广告。使用W[str(target_adgroup_id)]，得到：
    {'688732': 0.5773502691896258, '749768': 0.5773502691896258, '153247': 1.0, '405680': 0.7071067811865475,
     ... '588978': 1.0, '648820': 0.5773502691896258, '695765': 0.7071067811865475}
    2. 排序：按相似度值对广告进行排序，得到与目标广告最相似的广告。返回推荐结果：返回前 top_n 个与目标广告最相似的广告。

    :param W: 广告相似度矩阵
    :param target_adgroup_id: 目标广告的adgroup_id
    :param top_n: 推荐广告的数量
    :return: 推荐的广告
    """
    recommend_df = pd.DataFrame([W[str(target_adgroup_id)]]).T  # 使注意W的key是字符串
    # print(W[str(target_adgroup_id)])  # 用来获取目标广告的相似广告
    # print(recommend_df.head())
    recommend_df.rename(columns={0: '相似度'}, inplace=True)  # 重命名列名0为'相似度'
    recommend_df.sort_values(by='相似度', ascending=False, inplace=True)  # 按相似度降序排序

    return recommend_df.head(top_n)


# -------------------根据用户历史行为进行广告推荐-------------------
def recommend_by_user_action(dataSet, W, user_id, top_n=10):
    """
    根据用户历史行为和广告相似度为用户推荐广告。
    rank是一个字典，key是要进行推荐的广告id，value是推荐分数。推荐分数是用户点击的广告和其他广告的相似度加权和：
     {
      '110597': 0.3779644730092272,
      '114923': 0.13483997249264842,
      '156293': 0.5773502691896258,
      ...
      '8052': 0.13483997249264842,
      '818401': 0.13483997249264842
      }
    recommend_for_user 函数根据用户的历史广告点击行为和广告之间的相似度为用户推荐广告。具体步骤如下：
    1. 获取用户的历史点击记录，用于计算推荐概率。
    2. 计算推荐概率：对于每个用户已点击的广告，按广告的相似度对其他广告进行推荐。
         即，如果广告i和广告j相似，且用户已经点击了广告i，那么推荐广告j的概率就会增加。
    3. 排序：根据推荐概率对广告进行排序，得到推荐广告的排名。返回推荐结果：返回前 top_n 个最有可能被用户点击的广告。

    :param dataSet: 用户数据集
    :param W: 广告相似度矩阵
    :param user_id: 用户ID
    :param top_n: 推荐广告的数量
    :return: 推荐的广告列表
    """
    rank = {}  # 用来存储广告的推荐分数

    # 获取用户user_id的历史广告点击记录
    for ad_i, score in dataSet[str(user_id)].items():
        # 对于用户已经点击的广告ad_i，找到与之相似的广告ad_j，并根据相似度加权计算推荐分数。
        # 这里只考虑与广告ad_i相似度最高的top_n个广告，按照相似度的值降序排列
        for ad_j, w in sorted(W[str(ad_i)].items(), key=operator.itemgetter(1), reverse=True)[:top_n]:
            if ad_j not in dataSet[str(user_id)].keys():  # 排除已点击的广告
                rank.setdefault(ad_j, 0)
                rank[ad_j] += score * w  # 权重是广告相似度，注意这里的score都是1，因为使用的是clk=1的记录
    # pprint(rank)

    # 根据推荐分数排序，选出前top_n个广告
    recommend_to_user = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    # 将推荐结果转换为dataframe格式
    recommend_df = pd.DataFrame(recommend_to_user)
    recommend_df.rename(columns={0: 'adgroup_id', 1: '推荐分数'}, inplace=True)

    return recommend_df


if __name__ == '__main__':
    processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"  # 数据集存放主路径
    csv_path = f"{processed_data_path}/data_clk_col.csv"  # 保存处理后的数据

    # # 读取数据
    # ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')
    # # 处理数据
    # data_clk_col = preprocess_data(ad_u_data)
    # print("数据处理完毕")
    # # 保存数据
    # data_clk_col.to_csv(csv_path, sep=',', index=False)
    # print(f"数据已保存到{csv_path}")

    # 构建数据集
    dataSet = build_dataset_from_csv(csv_path)
    print("数据集构建完毕")
    # pprint(dataSet)

    # 计算共现矩阵
    N, C = calculate_cooccurrence_matrix(dataSet)
    print("---构造的共现矩阵---")
    # pprint(N)
    # pprint(C)

    # 计算广告相似度矩阵
    W = calculate_similarity(C, N)
    print("---构造广告的相似矩阵---")
    # pprint(W)

    # 查找与广告ad_id相似的广告
    ad_id = 10000  # 原作者这里给出的ad_id是118317，这里是为了和calculate_similarity对应起来方便查看，修改为了10000
    recommend_ad = recommend_by_ad_similarity(W, ad_id, top_n=10)
    print(f"---与广告{ad_id}相似的广告---")
    print(recommend_ad)

    # 为用户user_id推荐广告
    user_id = 387456
    recommend_to_387456 = recommend_by_user_action(dataSet, W, user_id, top_n=10)
    print(f"---为用户{user_id}推荐的广告---")
    print(recommend_to_387456)
