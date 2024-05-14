from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

from easier_nn.classic_dataset import load_mnist


class KNN:
    def __init__(self, k, distance=None):
        """
        初始化KNN模型
        通过初始化，可以得到以下属性：
            self.k: K值
            self.label_num: 类别的数量
            self.x_train: 训练数据
            self.y_train: 训练标签
            self.distance_func: 距离函数
        [使用方法]：
            from sklearn.model_selection import train_test_split
            from sklearn import datasets
            import numpy as np
            Iris = datasets.load_iris()  # 导入鸢尾花数据集
            X = Iris.data  # 获得样本特征向量
            y = Iris.target  # 获得样本label
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
            knn = KNN(k=3)
            knn.fit(X_train, y_train, label_num=4)
            y_pred = knn.predict(X_test)
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            print("准确率：", accuracy)  # 准确率： 0.975
        :param k: K值
        :param distance: 距离函数，默认是欧氏距离
        """
        self.k = k  # K值
        self.label_num = None  # 类别的数量
        self.x_train = None  # 训练数据
        self.y_train = None  # 训练标签
        self.distance_func = None  # 距离函数

        self.__init_distance(distance)  # 初始化距离函数

    def fit(self, x_train, y_train, label_num=None):
        """
        在类中保存训练数据，并获取类别y_train的数量
        :param x_train: 训练数据
        :param y_train: 训练标签
        :param label_num: 标签的数量，不传入就按照y_train的类别数量来计算
        """
        self.x_train = x_train
        self.y_train = y_train
        if isinstance(label_num, int):
            self.label_num = label_num
        else:
            self.label_num = len(np.unique(y_train))  # 获取类别的数量

    def predict(self, x_test):
        """
        预测样本 test_x 的类别
        :param x_test: 测试数据
        """
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self._get_label(x)
        return predicted_test_labels

    def _get_knn_indices(self, x):
        """
        获取距离目标样本点最近的K个样本点的下标
        :param x: 目标样本点
        :return: 距离目标样本点最近的K个样本点的下标
        """
        # 计算已知样本的距离，是列表[d_1, d_2, ..., d_n]，其中n=len(self.x_train)
        dis = list(map(lambda a: self.distance(a, x), self.x_train))
        knn_indices = np.argsort(dis)  # 按距离从小到大排序，并得到对应的下标
        knn_indices = knn_indices[:self.k]  # 从n个里取最近的K个的下标
        return knn_indices  # [index_1, index_2, ..., index_k]

    def _get_label(self, x):
        """
        获取某个样本x的类别
        :param x: 目标样本x
        :return: 样本x的类别
        """
        knn_indices = self._get_knn_indices(x)
        # 类别计数
        label_statistic = np.zeros(shape=[self.label_num])
        for index in knn_indices:
            label = int(self.y_train[index])
            label_statistic[label] += 1
        # label_statistic是一个列表[num_1, num_2, ..., num_label_num]，表示每个类别出现的数量
        return np.argmax(label_statistic)  # 返回数量最多的类别

    def __init_distance(self, distance):
        """
        初始化距离函数，请勿在其余地方调用。
        """
        if distance is not None:
            if callable(distance) and hasattr(distance, '__code__') and distance.__code__.co_argcount == 2:
                self.distance_func = distance  # 如果distance是有两个参数的函数，那么就直接使用这个函数
            else:
                raise ValueError("距离函数有误，请检查是否是两个参数的函数")
        else:
            self.distance = self.__euclidean_distance  # 距离度量，默认是欧氏距离

    @staticmethod
    def __euclidean_distance(a, b):
        """
        初始化distance的方法，只用于__init_distance，请勿在其余地方调用。
        计算两个向量之间的欧氏距离。
        """
        return np.sqrt(np.sum(np.square(a - b)))

# X, y = load_mnist()
# X = np.array(X)
# y = np.array(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# knn = KNN(k=3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# accuracy = np.sum(y_pred == y_test) / len(y_test)
# print("准确率：", accuracy)

# 使用简单的二维数据测试
Iris = datasets.load_iris()  # 导入鸢尾花数据集
X = Iris.data  # 获得样本特征向量
y = Iris.target  # 获得样本label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
knn = KNN(k=3)
knn.fit(X_train, y_train, label_num=4)
y_pred = knn.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("准确率：", accuracy)


