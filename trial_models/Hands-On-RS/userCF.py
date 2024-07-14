import numpy as np
import pandas as pd

# 用户-物品评分数据，因为是稀疏矩阵，所以使用字典存储，后面也使用字典来计算用户相似度
user_data = {'y': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
             'X1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'k': 3},
             'X2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'k': 5},
             'X3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'k': 4},
             'X4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'k': 1}
             }
user_names = list(user_data.keys())
# 用户相似度矩阵，行列均为用户名称
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)),
    index=user_names,
    columns=user_names
)

# 遍历每条用户-物品评分数据
for u1, info1 in user_data.items():
    for u2, info2 in user_data.items():
        if u1 == u2:
            continue
        vec1, vec2 = [], []  # 用户u1, u2对应的评分向量
        for item1, rating1 in info1.items():
            rating2 = info2.get(item1, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]  # 计算不同用户之间的皮尔逊相关系数
print(similarity_matrix)

target_user = 'y'
topN = 2
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:topN+1].index.tolist()  # 去除自身
print(f'与用户{target_user}最相似的{topN}个用户为：{sim_users}')

weighted_scores = 0.
corr_values_sum = 0.
target_item = 'k'
for user in sim_users:
    sim_y_Xi = similarity_matrix[target_user][user]
    Xi_mean_rating = np.mean(list(user_data[user].values()))  # 用户Xi的平均评分
    weighted_scores += sim_y_Xi * (user_data[user][target_item] - Xi_mean_rating)  # 分子
    corr_values_sum += sim_y_Xi  # 分母
y_mean_rating = np.mean(list(user_data[target_user].values()))
target_item_pred = y_mean_rating + weighted_scores / corr_values_sum  # 预测用户y对物品k的评分
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
