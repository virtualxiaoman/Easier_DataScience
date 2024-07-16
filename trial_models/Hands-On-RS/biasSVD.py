import random
import math


class BiasSVD:
    def __init__(self, rating_data, F=5, alpha=0.1, lmbda=0.1, max_iter=1000):
        self.F = F  # 这个表示隐向量的维度
        self.P = dict()  # 用户矩阵P  大小是[users_num, F]
        self.Q = dict()  # 物品矩阵Q  大小是[item_nums, F]
        self.bu = dict()  # 用户偏置系数
        self.bi = dict()  # 物品偏置系数
        self.mu = 0  # 全局偏置系数
        self.alpha = alpha  # 学习率
        self.lmbda = lmbda  # 正则项系数
        self.max_iter = max_iter  # 最大迭代次数
        self.rating_data = rating_data  # 评分矩阵

        for user, items in self.rating_data.items():
            # 初始化矩阵P和Q。随机数根据经验，和1/sqrt(F)成正比
            self.P[user] = [random.random() / math.sqrt(self.F) for x in range(0, F)]
            self.bu[user] = 0
            for item, rating in items.items():
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F) for x in range(0, F)]
                    self.bi[item] = 0

    # 采用随机梯度下降的方式训练模型参数
    def train(self):
        cnt, mu_sum = 0, 0
        for user, items in self.rating_data.items():
            for item, rui in items.items():
                mu_sum, cnt = mu_sum + rui, cnt + 1
        self.mu = mu_sum / cnt

        for step in range(self.max_iter):
            # 遍历所有的用户及历史交互物品
            for user, items in self.rating_data.items():
                # 遍历历史交互物品
                for item, rui in items.items():
                    rhat_ui = self.predict(user, item)  # 评分预测
                    e_ui = rui - rhat_ui  # 评分预测偏差
                    # 参数更新
                    # $$b_u=b_u+\alpha(e_{ui}-\lambda b_u)$$
                    # $$b_i=b_i+\alpha(e_{ui}-\lambda b_i)$$
                    self.bu[user] += self.alpha * (e_ui - self.lmbda * self.bu[user])
                    self.bi[item] += self.alpha * (e_ui - self.lmbda * self.bi[item])
                    for k in range(0, self.F):
                        # p_{u,k}=p_{u,k}+\alpha e_{ui}q_{i,k}-\lambda p_{u,k}
                        # q_{i,k}=q_{i,k}+\alpha e_{ui}p_{u,k}-\lambda q_{i,k}
                        self.P[user][k] += self.alpha * (e_ui * self.Q[item][k] - self.lmbda * self.P[user][k])
                        self.Q[item][k] += self.alpha * (e_ui * self.P[user][k] - self.lmbda * self.Q[item][k])
            # 逐步降低学习率
            self.alpha *= 0.9

    # 评分预测
    def predict(self, user, item):
        # $$r_{ij}^{\prime}=\mu+b_u+b_i+\boldsymbol{p}^{(i)}\boldsymbol{q}_j$$
        return sum(self.P[user][f] * self.Q[item][f] for f in range(0, self.F))\
            + self.bu[user] + self.bi[item] + self.mu


# 加载数据
rating_data = {'y': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
               'X1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'k': 3},
               'X2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'k': 5},
               'X3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'k': 4},
               'X4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'k': 1}
               }
biassvd = BiasSVD(rating_data, F=10)
biassvd.train()
# 预测用户y对物品k的评分
for item in ['k']:
    print(item, biassvd.predict('y', item))

