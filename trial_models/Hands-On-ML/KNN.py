from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from skimage import io  # 图像输入输出
from skimage.color import rgb2lab, lab2rgb  # 图像通道转换
from sklearn.neighbors import KNeighborsRegressor  # KNN 回归器
import os

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
            self.distance: 距离函数
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
        self.distance = None  # 距离函数

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
                self.distance = distance  # 如果distance是有两个参数的函数，那么就直接使用这个函数
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


# 马氏距离
def mahalanobis_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


if __name__ == '__main__':
    # 使用自定义KNN回归器
    # Iris = datasets.load_iris()  # 导入鸢尾花数据集
    # X = Iris.data  # 获得样本特征向量
    # y = Iris.target  # 获得样本label
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    # knn = KNN(k=3, distance=mahalanobis_distance)
    # knn.fit(X_train, y_train, label_num=4)
    # y_pred = knn.predict(X_test)
    # accuracy = np.sum(y_pred == y_test) / len(y_test)
    # print("准确率：", accuracy)

    # KNN实现风格迁移，https://hml.boyuai.com/books/chapter3/
    def read_style_image(file_path, size):
        # 读入风格图像, 得到映射 X->Y
        # 其中X储存3*3像素格的灰度值，Y储存中心像素格的色彩值
        # 读取图像文件，设图像宽为W，高为H，得到W*H*3的RGB矩阵
        img = io.imread(file_path)
        # 将RGB矩阵转换成LAB表示法的矩阵，大小仍然是W*H*3，三维分别是L、A、B
        img = rgb2lab(img)
        # 取出图像的宽度和高度
        w, h = img.shape[:2]

        X = []
        Y = []
        # 枚举全部可能的中心点
        for x in range(size, w - size):
            for y in range(size, h - size):
                X.append(img[x - size: x + size + 1, y - size: y + size + 1, 0].flatten())  # 保存所有窗口。当size=1时，窗口大小为3*3
                Y.append(img[x, y, 1:])  # 保存窗口对应的色彩值a和b
        return X, Y


    def rebuild(img, size, knn_model):
        # 将内容图像转为LAB表示
        img = rgb2lab(img)
        w, h = img.shape[:2]
        # 初始化输出图像对应的矩阵
        photo = np.zeros([w, h, 3])
        # 枚举内容图像的中心点，保存所有窗口
        print('Constructing window...')
        X = []
        for x in range(size, w - size):
            for y in range(size, h - size):
                # 得到中心点对应的窗口
                window = img[x - size: x + size + 1, y - size: y + size + 1, 0].flatten()
                X.append(window)
        X = np.array(X)

        # 用KNN回归器预测颜色
        print('Predicting...')
        pred_ab = knn_model.predict(X).reshape(w - 2 * size, h - 2 * size, -1)
        # 设置输出图像
        photo[:, :, 0] = img[:, :, 0]
        photo[size: w - size, size: h - size, 1:] = pred_ab

        # 由于最外面size层无法构造窗口，简单起见，我们直接把这些像素裁剪掉
        photo = photo[size: w - size, size: h - size, :]
        return photo


    path = '../input/KNN_style_transfer'
    # block_size表示向外扩展的层数，扩展1层即3*3
    block_size = 1
    X, Y = read_style_image(os.path.join(path, 'style.jpg'), size=block_size)  # 建立映射

    # weights='distance'表示邻居的权重与其到样本的距离成反比
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance')

    knn.fit(X, Y)
    content = io.imread(os.path.join(path, 'arona.jpg'))
    new_photo = rebuild(content, size=block_size, knn_model=knn)
    # 为了展示图像，我们将其再转换为RGB表示
    new_photo = lab2rgb(new_photo)

    fig = plt.figure()
    plt.imshow(new_photo)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
