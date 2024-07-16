from itertools import combinations
import pandas as pd


def load_data(train_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "itemid", "rate"])  # 提取用户交互记录数据
    train_data = train_data.drop(0)  # 第一行是列名
    train_data = train_data.astype(int)  # 转换数据类型
    print(train_data.head(3))
    return train_data


def get_uitems_iusers(train):
    u_items = dict()
    i_users = dict()
    for index, row in train.iterrows():  # 处理用户交互记录
        u_items.setdefault(row["userid"], set())  # 初始化该用户row["userid"]
        i_users.setdefault(row["itemid"], set())  # 初始化该物品
        u_items[row["userid"]].add(row["itemid"])  # 得到该用户交互过的所有item
        i_users[row["itemid"]].add(row["userid"])  # 得到该物品交互过的所有user
    print("使用的用户个数为：{}".format(len(u_items)))
    print("使用的item个数为：{}".format(len(i_users)))
    return u_items, i_users


def swing_model(u_items, i_users, alpha):
    # print([i for i in i_users.values()][:5])
    # print([i for i in u_items.values()][:5])
    item_pairs = list(combinations(i_users.keys(), 2))  # C(n,2) = (n!)/(2!(n-2)!) = (n(n-1))/2 = 50*49/2 = 1225
    print("item pairs length：{}".format(len(item_pairs)))
    item_sim_dict = dict()
    for (i, j) in item_pairs:
        user_pairs = list(combinations(i_users[i] & i_users[j], 2))  # 共同交互过i, j的用户对
        result = 0
        for (u, v) in user_pairs:
            result += 1 / (alpha + list(u_items[u] & u_items[v]).__len__())  # \frac1{\alpha+|I_u\cap I_v|},
        if result != 0:
            item_sim_dict.setdefault(i, dict())  # 初始化物品i
            item_sim_dict[i][j] = format(result, '.6f')
    return item_sim_dict  # 返回item相似度字典, key为item_id, value为与该item相似的item_id及相似度


def save_item_sims(item_sim_dict, top_k, path):
    new_item_sim_dict = dict()
    try:
        writer = open(path, 'w', encoding='utf-8')
        for item, sim_items in item_sim_dict.items():
            new_item_sim_dict.setdefault(item, dict())
            new_item_sim_dict[item] = dict(
                sorted(sim_items.items(), key=lambda k: k[1], reverse=True)[:top_k])  # 排序取出 top_k个相似的item
            writer.write('item_id:%d\t%s\n' % (item, new_item_sim_dict[item]))
        print("SUCCESS: top_{} item saved".format(top_k))
    except Exception as e:
        print(e.args)


if __name__ == "__main__":
    train_data_path = "input/ratings_final.txt"
    item_sim_save_path = "output/item_sim_dict.txt"
    alpha = 0.5
    top_k = 10  # 与item相似的前 k 个item
    train = load_data(train_data_path)
    u_items, i_users = get_uitems_iusers(train)
    item_sim_dict = swing_model(u_items, i_users, alpha)
    save_item_sims(item_sim_dict, top_k, item_sim_save_path)
