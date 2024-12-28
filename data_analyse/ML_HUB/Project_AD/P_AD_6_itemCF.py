import pandas as pd
import numpy as np
from math import sqrt
import operator


# -------------------数据预处理-------------------
def preprocess_data(data):
    """
    预处理数据：
      1. 筛选点击记录
      2. 去重，确保每个用户对某广告组只保留一条点击记录
      3. 选择需要的属性列
    :param data: 原始数据
    :return: 处理后的数据
    """
    # 筛选有点击的数据
    data_clk = data[data['clk'] == 1].copy()

    # 根据【userid】、【adgroup_id】去重
    data_clk.drop_duplicates(subset=['user_id', 'adgroup_id'], keep='first', inplace=True)

    # 选取【'userid', 'adgroup_id', 'clk'】三列数据
    data_clk = data_clk[['user_id', 'adgroup_id', 'clk']]

    return data_clk


# 将处理后的数据保存为txt
def save_data_to_txt(data, file_path):
    """
    将处理后的数据保存到txt文件
    :param data: 处理后的数据
    :param file_path: 保存的文件路径
    """
    data.to_csv(file_path, sep=',', index=False)
    print(f"数据已保存到 {file_path}")


# -------------------数据集构建-------------------
def build_dataset_from_txt(file_path):
    """
    从txt文件构建数据集
    :param file_path: txt文件路径
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
    计算广告共现矩阵
    :param dataSet: 用户点击数据集
    :return: N (广告点击人数) 和 C (广告共现次数)
    """
    N = {}  # 喜欢广告i的总人数
    C = {}  # 喜欢广告i也喜欢广告j的人数

    for userid, item in dataSet.items():
        for i, score in item.items():
            N.setdefault(i, 0)
            N[i] += 1
            C.setdefault(i, {})

            for j, scores in item.items():
                if j != i:  # 不计算广告自身的共现
                    C[i].setdefault(j, 0)
                    C[i][j] += 1

    return N, C


# -------------------相似度计算-------------------
def calculate_similarity(C, N):
    """
    计算广告之间的相似度矩阵
    :param C: 广告共现次数矩阵
    :param N: 每个广告的点击人数
    :return: 广告相似度矩阵 W
    """
    W = {}  # 广告的相似度矩阵
    for i, item in C.items():
        W.setdefault(i, {})
        for j, count in item.items():
            W[i].setdefault(j, 0)
            W[i][j] = C[i][j] / sqrt(N[i] * N[j])  # 使用余弦相似度公式

    return W


# -------------------根据广告相似度进行推荐-------------------
def recommend_by_similarity(W, target_adgroup_id, top_n=10):
    """
    根据广告相似度为某个广告推荐相似广告
    :param W: 广告相似度矩阵
    :param target_adgroup_id: 目标广告的adgroup_id
    :param top_n: 推荐广告的数量
    :return: 推荐的广告
    """
    recommend_df = pd.DataFrame([W[str(target_adgroup_id)]]).T
    recommend_df.rename(columns={0: '相似度'}, inplace=True)
    recommend_df.sort_values(by='相似度', ascending=False, inplace=True)

    return recommend_df.head(top_n)


# -------------------根据用户历史行为进行广告推荐-------------------
def recommend_for_user(dataSet, W, user_id, top_n=10):
    """
    根据用户历史行为和广告相似度为用户推荐广告
    :param dataSet: 用户数据集
    :param W: 广告相似度矩阵
    :param user_id: 用户ID
    :param top_n: 推荐广告的数量
    :return: 推荐的广告列表
    """
    rank = {}
    # 获取用户的历史广告点击记录
    for i, score in dataSet[str(user_id)].items():
        for j, w in sorted(W[str(i)].items(), key=operator.itemgetter(1), reverse=True)[:top_n]:
            if j not in dataSet[str(user_id)].keys():  # 排除已点击的广告
                rank.setdefault(j, 0)
                rank[j] += score * w

    # 将推荐广告按概率排序
    recommend_to_user = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    recommend_df = pd.DataFrame(recommend_to_user)
    recommend_df.rename(columns={0: 'adgroup_id', 1: '用户点击的概率'}, inplace=True)

    return recommend_df


if __name__ == '__main__':
    # 读取数据
    processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"
    ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')

    # # 处理后的数据
    # data_clk_col = preprocess_data(ad_u_data)
    # print("数据处理完毕")
    #
    # # 保存数据
    # save_data_to_txt(data_clk_col, f'{processed_data_path}/data_clk_col.txt')
    # print(f"数据已保存到 {processed_data_path}/data_clk_col.txt")

    # 构建数据集
    dataSet = build_dataset_from_txt(f'{processed_data_path}/data_clk_col.txt')
    print("数据集构建完毕")

    # 计算共现矩阵
    N, C = calculate_cooccurrence_matrix(dataSet)
    print("---构造的共现矩阵---")

    # print('N:', N)
    # print('C:', C)

    # 计算广告相似度矩阵
    W = calculate_similarity(C, N)
    print("---构造广告的相似矩阵---")

    # print(W)

    # 推荐与广告【118317】相似的广告
    recommend_118317 = recommend_by_similarity(W, 118317, top_n=10)
    print("---推荐给广告【118317】的相似广告----")
    print(recommend_118317)

    # 为用户【387456】推荐广告
    recommend_to_387456 = recommend_for_user(dataSet, W, 387456, top_n=10)
    print("---推荐给用户【387456】的广告----")
    print(recommend_to_387456)
