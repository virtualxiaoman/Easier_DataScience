import numpy as np
from sklearn.manifold import LocallyLinearEmbedding  # LLE
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# 生成一个瑞士卷数据集作为示例
X, y = make_swiss_roll(n_samples=1000, noise=0.2)
# 打印数据维度
print("Data shape:", X.shape, y.shape)
# 可视化原始数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Original Data')
plt.show()

# 创建LocallyLinearEmbedding对象并拟合数据，n_neighbors为最近邻个数，n_components为降维后的维度
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_transformed = lle.fit_transform(X)

# 打印转换后的数据维度
print("Transformed data shape:", X_transformed.shape)

# 可视化转换后的数据
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Transformed Data')
plt.show()
