# 1. 垃圾回收
print("---------- 1. 垃圾回收 -------------")
import gc


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


# 启用调试模式，保存所有不可达对象
gc.set_debug(gc.DEBUG_SAVEALL)

# 创建循环引用
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1
print(f"node1.next.value: {node1.next.value}")
print(f"node2.next.value: {node2.next.value}")

# 删除引用，但对象仍存在（循环引用）
del node1
del node2

# 手动触发垃圾回收
gc.collect()

# 打印不可达对象
for obj in gc.garbage:
    print(f"未释放的对象: {obj}")

for obj in gc.garbage:
    del obj
gc.garbage.clear()
print(f"gc.garbage 长度: {len(gc.garbage)}")  # 应输出 0


# 2. CountVectorizer
print("---------- 2. CountVectorizer -------------")
from sklearn.feature_extraction.text import CountVectorizer

# 样本文本数据
documents = [
    "I love programming.",
    "I love machine learning.",
    "Machine learning is fun."
]

# 创建 CountVectorizer 对象
vectorizer = CountVectorizer()

# 将文本数据转化为特征矩阵
X = vectorizer.fit_transform(documents)

# 查看词汇表
print("Vocabulary:", vectorizer.get_feature_names_out())

# 查看文档-词矩阵
print("Document-Term Matrix:")
print(X.toarray())

import pandas as pd

# 创建一个包含标签列的 DataFrame
data = {
    'ID': [1, 2, 3],
    'tags': ['python, pandas, data', 'machine learning, deep learning', 'data analysis, python']
}

df = pd.DataFrame(data)

# 打印原始 DataFrame
print("原始 DataFrame:")
print(df)

# 使用 str.split() 将 'tags' 列按逗号分割成列表
df['tags_split'] = df['tags'].str.split(', ')
print("\n使用 str.split() 处理后的 DataFrame:")
print(df)

# 使用 str.get_dummies() 创建每个标签的独热编码
df_tags_dummies = df['tags_split'].str.join('|').str.get_dummies()
print("\n使用 str.get_dummies() 处理后的 DataFrame:")
print(df_tags_dummies)


print("---------- 3. ElasticNet -------------")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成模拟回归数据
X, y = make_regression(n_samples=10000, n_features=100, noise=0.2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置不同的alpha和lambda值
alpha_values = [0.1, 0.5, 1.0]
lambda_value = 0.1

# 用于存储不同模型的均方误差
mse_train = []
mse_test = []

# 分别训练不同alpha的ElasticNet模型
for alpha in alpha_values:
    model = ElasticNet(alpha=lambda_value, l1_ratio=alpha)
    model.fit(X_train, y_train)

    # 计算训练集和测试集的均方误差
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

    print(f"Alpha: {alpha}")
    print(f"Train MSE: {mse_train[-1]:.4f}, Test MSE: {mse_test[-1]:.4f}")
    print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}")
    print("-" * 50)

# 绘制不同alpha值下的训练和测试误差
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, mse_train, label='Train MSE', marker='o')
plt.plot(alpha_values, mse_test, label='Test MSE', marker='o')
plt.xlabel('Alpha (L1 ratio)')
plt.ylabel('Mean Squared Error')
plt.title('ElasticNet - Effect of Alpha on MSE')
plt.legend()
plt.grid(True)
plt.show()
