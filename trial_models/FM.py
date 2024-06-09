import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  # sklearn中的评价指标函数库
from tqdm import tqdm

# 数据集的每一行包含一个商品，前 24 列是其特征，最后一列是 0 或 1，分别表示用户没有或有点击该商品。
# 我们的目标是根据输入特征预测用户在测试集上的行为，是一个二分类问题。
data = np.loadtxt('input/MF_data/fm_dataset.csv', delimiter=',')

# 划分数据集
np.random.seed(0)
ratio = 0.8
split = int(ratio * len(data))
x_train = data[:split, :-1]
y_train = data[:split, -1]
x_test = data[split:, :-1]
y_test = data[split:, -1]
# 特征数
feature_num = x_train.shape[1]
print('训练集大小：', len(x_train))
print('测试集大小：', len(x_test))
print('特征数：', feature_num)


class FM:

    def __init__(self, feature_num, vector_dim):
        # vector_dim代表公式中的k，为向量v的维度
        self.theta0 = 0.0  # 常数项
        self.theta = np.zeros(feature_num)  # 线性参数
        self.v = np.random.normal(size=(feature_num, vector_dim))  # 双线性参数
        self.eps = 1e-6  # 精度参数

    def _logistic(self, x):
        # 工具函数，用于将预测转化为概率
        return 1 / (1 + np.exp(-x))

    def pred(self, x):
        # 线性部分
        linear_term = self.theta0 + x @ self.theta
        # 双线性部分
        square_of_sum = np.square(x @ self.v)
        sum_of_square = np.square(x) @ np.square(self.v)
        # 最终预测
        y_pred = self._logistic(linear_term \
                                + 0.5 * np.sum(square_of_sum - sum_of_square, axis=1))
        # 为了防止后续梯度过大，对预测值进行裁剪，将其限制在某一范围内
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return y_pred

    def update(self, grad0, grad_theta, grad_v, lr):
        self.theta0 -= lr * grad0
        self.theta -= lr * grad_theta
        self.v -= lr * grad_v


# 超参数设置，包括学习率、训练轮数等
vector_dim = 16
learning_rate = 0.01
lbd = 0.05
max_training_step = 200
batch_size = 32

# 初始化模型
np.random.seed(0)
model = FM(feature_num, vector_dim)

train_acc = []
test_acc = []
train_auc = []
test_auc = []

with tqdm(range(max_training_step)) as pbar:
    for epoch in pbar:
        st = 0
        while st < len(x_train):
            ed = min(st + batch_size, len(x_train))
            X = x_train[st: ed]
            Y = y_train[st: ed]
            st += batch_size
            # 计算模型预测
            y_pred = model.pred(X)
            # 计算交叉熵损失
            cross_entropy = -Y * np.log(y_pred) \
                            - (1 - Y) * np.log(1 - y_pred)
            loss = np.sum(cross_entropy)
            # 计算损失函数对y的梯度，再根据链式法则得到总梯度
            grad_y = (y_pred - Y).reshape(-1, 1)
            # 计算y对参数的梯度
            # 常数项
            grad0 = np.sum(grad_y * (1 / len(X) + lbd))
            # 线性项
            grad_theta = np.sum(grad_y * (X / len(X) + lbd * model.theta), axis=0)
            # 双线性项
            grad_v = np.zeros((feature_num, vector_dim))
            for i, x in enumerate(X):
                # 先计算sum(x_i * v_i)
                xv = x @ model.v
                grad_vi = np.zeros((feature_num, vector_dim))
                for s in range(feature_num):
                    grad_vi[s] += x[s] * xv - (x[s] ** 2) * model.v[s]
                grad_v += grad_y[i] * grad_vi
            grad_v = grad_v / len(X) + lbd * model.v
            model.update(grad0, grad_theta, grad_v, learning_rate)

            pbar.set_postfix({
                '训练轮数': epoch,
                '训练损失': f'{loss:.4f}',
                '训练集准确率': train_acc[-1] if train_acc else None,
                '测试集准确率': test_acc[-1] if test_acc else None
            })
        # 计算模型预测的准确率和AUC
        # 预测准确率，阈值设置为0.5
        y_train_pred = (model.pred(x_train) >= 0.5)
        acc = np.mean(y_train_pred == y_train)
        train_acc.append(acc)
        auc = metrics.roc_auc_score(y_train, y_train_pred)  # sklearn中的AUC函数
        train_auc.append(auc)

        y_test_pred = (model.pred(x_test) >= 0.5)
        acc = np.mean(y_test_pred == y_test)
        test_acc.append(acc)
        auc = metrics.roc_auc_score(y_test, y_test_pred)
        test_auc.append(auc)

print(f'测试集准确率：{test_acc[-1]}，\t测试集AUC：{test_auc[-1]}')

# 绘制训练曲线
plt.figure(figsize=(13, 5))
x_plot = np.arange(len(train_acc)) + 1

plt.subplot(121)
plt.plot(x_plot, train_acc, color='blue', label='train acc')
plt.plot(x_plot, test_acc, color='red', ls='--', label='test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(x_plot, train_auc, color='blue', label='train AUC')
plt.plot(x_plot, test_auc, color='red', ls='--', label='test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()
