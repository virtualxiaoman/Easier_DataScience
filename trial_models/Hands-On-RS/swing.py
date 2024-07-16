import numpy as np
import pandas as pd

# 物品-用户点击数据
item_data = {
    'A': {'y': 1, 'X1': 1, 'X2': 1, 'X3': 1},
    'B': {'y': 1, 'X1': 1, 'X4': 1},
    'C': {'y': 0, 'X2': 1, 'X3': 1, 'X4': 1},
    'D': {'y': 1, 'X1': 1, 'X3': 1, 'X4': 1},
    'E': {'X2': 1, 'X3': 1, 'X4': 1}
}
item_names = list(item_data.keys())
# 物品相似度矩阵，行列均为物品名称
similarity_matrix = pd.DataFrame(
    np.zeros((len(item_data), len(item_data))),
    index=item_names,
    columns=item_names
)

# 计算物品相似度矩阵
alpha = 0.5
for i1 in item_data.keys():
    for i2 in item_data.keys():
        if i1 == i2:
            continue
        common_users = set(item_data[i1].keys()).intersection(set(item_data[i2].keys()))  # 共同点击过i1, i2的用户
        if not common_users:
            similarity_matrix[i1][i2] = 0
            continue
        sum_weights = 0
        for u in common_users:
            for v in common_users:
                if u == v:
                    continue  # 跳过同一个用户
                I_u = set(item for item, users in item_data.items() if u in users)
                I_v = set(item for item, users in item_data.items() if v in users)
                I_u_v = I_u.intersection(I_v)
                weight = 1 / (np.sqrt(len(I_u)) * np.sqrt(len(I_v)) * (alpha + len(I_u_v)))
                sum_weights += weight
        similarity_matrix[i1][i2] = sum_weights

print("物品相似度矩阵：")
print(similarity_matrix)

# 计算推荐评分
target_user = 'y'
target_item = 'E'
topN = 2

# 获取与目标物品最相似的topN物品
sim_items = []
sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
for item in sim_items_list:
    # 排除target_item本身
    if target_user in item_data[item]:
        sim_items.append(item)
    if len(sim_items) == topN:
        break
print(f'与物品{target_item}最相似的{topN}个物品为：{sim_items}')

k_mean_rating = np.mean([item_data[target_item][user] for user in item_data[target_item]])  # 物品k的平均评分
weighted_scores = 0.
corr_values_sum = 0.
for item in sim_items:
    sim_k_Ii = similarity_matrix[target_item][item]  # 物品k与物品I_i的相似度
    Ii_mean_rating = np.mean([item_data[item][user] for user in item_data[item]])  # 物品I_i的平均评分
    weighted_scores += sim_k_Ii * (item_data[item][target_user] - Ii_mean_rating)  # 分子
    corr_values_sum += sim_k_Ii  # 分母
target_item_pred = k_mean_rating + weighted_scores / corr_values_sum if corr_values_sum != 0 else k_mean_rating
print(f'用户{target_user}对物品{target_item}的购买预测为：{target_item_pred}')
