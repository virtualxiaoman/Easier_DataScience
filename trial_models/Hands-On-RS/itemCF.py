import numpy as np
import pandas as pd

# 物品-用户评分数据
item_data = {'A': {'y': 5.0, 'X1': 3.0, 'X2': 4.0, 'X3': 3.0, 'X4': 1.0},
             'B': {'y': 3.0, 'X1': 1.0, 'X2': 3.0, 'X3': 3.0, 'X4': 5.0},
             'C': {'y': 4.0, 'X1': 2.0, 'X2': 4.0, 'X3': 1.0, 'X4': 5.0},
             'D': {'y': 4.0, 'X1': 3.0, 'X2': 3.0, 'X3': 5.0, 'X4': 2.0},
             'E': {'X1': 3.0, 'X2': 5.0, 'X3': 4.0, 'X4': 1.0}
             }
item_names = list(item_data.keys())
# 物品相似度矩阵，行列均为物品名称
similarity_matrix = pd.DataFrame(
    np.identity(len(item_data)),
    index=item_names,
    columns=item_names
)

# 遍历每条物品-用户评分数据
for i1, info1 in item_data.items():
    for i2, info2 in item_data.items():
        if i1 == i2:
            continue
        vec1, vec2 = [], []  # 物品i1, i2对应的评分向量
        for user, rating1 in info1.items():
            rating2 = info2.get(user, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        similarity_matrix[i1][i2] = np.corrcoef(vec1, vec2)[0][1]  # 计算不同物品之间的皮尔逊相关系数
print(similarity_matrix)

target_user = 'y'
target_item = 'E'
topN = 2
sim_items = []
sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
for item in sim_items_list:
    # 如果target_user对物品item评分过，也就是排除了target_item本身
    if target_user in item_data[item]:
        sim_items.append(item)
    if len(sim_items) == topN:
        break
print(f'与物品{target_item}最相似的{topN}个物品为：{sim_items}')

k_mean_rating = np.mean(list(item_data[target_item].values()))  # 物品k的平均评分
weighted_scores = 0.
corr_values_sum = 0.
for item in sim_items:
    sim_k_Ii = similarity_matrix[target_item][item]  # 物品k与物品I_i的相似度
    Ii_mean_rating = np.mean(list(item_data[item].values()))  # 物品I_i的平均评分
    weighted_scores += sim_k_Ii * (item_data[item][target_user] - Ii_mean_rating)  # 分子
    corr_values_sum += sim_k_Ii  # 分母
target_item_pred = k_mean_rating + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
