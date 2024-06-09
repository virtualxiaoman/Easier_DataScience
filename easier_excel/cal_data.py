import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
import torch
from torch import nn
from torch.utils import data


from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.Colorful_Console import func_warning as fw

def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    :param data_arrays: 一个包含数据数组的元组或列表。通常包括输入特征和对应的标签(features, labels)。
    :param batch_size: 每个小批量样本的数量。
    :param is_train: True数据将被随机洗牌(用于训练);False数据将按顺序提供(用于模型的评估或测试)。
    """
    dataset = data.TensorDataset(*data_arrays)  # 将数据数组转换为TensorDataset对象(将数据存储为Tensor对象，并允许按索引访问)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


class CalData:
    """
    数据处理类。
    """
    def __init__(self, df):
        self.df = df

    @staticmethod
    def has_chinese(text):
        """
        判断文本中是否含有中文字符。
        :param text: str，文本。
        :return: bool，True表示含有中文字符，False表示不含有中文字符。
        """
        pattern = re.compile(r'[\u4e00-\u9fa5]')  # 匹配中文字符的正则表达式
        return bool(pattern.search(text))

class Linear(CalData):
    """
    线性回归、逻辑回归、多项式回归。
    """
    def __init__(self, df):
        super().__init__(df)

    def cal_linear(self, X_name, y_name, use_bias=True):
        """
        线性回归。
        公式：y = X @ w + b
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param use_bias: bool，是否使用偏置参数。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, 1)
        self._cal_linear(X, y, use_bias)

    def cal_logistic(self, X_name, y_name, pos_label=1):
        """
        逻辑回归。
        公式：y = 1 / (1 + exp(-X @ w + b))
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param pos_label: int，正类别标签。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, )
        self._cal_logistic(X, y, pos_label)

    def cal_poly(self, X_name, y_name, degree=2, include_linear_bias=False, include_poly_bias=False):
        """
        多项式回归。
        [Tips]:
            比如X的shape是(8,3), degree=3, include_bias=True，那么X_poly的shape是(8, 20)，其中包含了bias项。具体顺序如下：
            ['1'
             'x0' 'x1' 'x2'
             'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2'
             'x0^3' 'x0^2 x1' 'x0^2 x2' 'x0 x1^2' 'x0 x1 x2' 'x0 x2^2' 'x1^3' 'x1^2 x2' 'x1 x2^2' 'x2^3']
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param degree: int，多项式的次数。
        :param include_linear_bias: bool，是否包含线性偏置参数。
        :param include_poly_bias: bool，是否包含多项式偏置参数。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, 1)
        self._cal_poly(X, y, degree, include_linear_bias, include_poly_bias)

    @staticmethod
    def _cal_linear(X, y, use_bias=True):
        """
        线性回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param use_bias: bool，是否使用偏置参数。
        :return: None
        """
        lin_reg = LinearRegression(fit_intercept=use_bias)
        lin_reg.fit(X, y)
        print(CT("线性回归:").blue())
        print("偏置参数：", lin_reg.intercept_)  # b
        print("权重参数：", lin_reg.coef_)  # w

    @staticmethod
    def _cal_logistic(X, y, pos_label=1, draw_roc=False):
        """
        逻辑回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param pos_label: int，正类别标签。
        :return: None
        """
        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        print(CT("逻辑回归:").blue())
        print("偏置参数：", log_reg.intercept_)  # b
        print("权重参数：", log_reg.coef_)  # w
        print("类别：", log_reg.classes_)  # 类别
        print("准确率：", log_reg.score(X, y))
        # print("预测结果：", log_reg.predict(X))

        y_pred_prob = log_reg.predict_proba(X)[:, 1]  # 计算预测概率
        # 计算ROC曲线和AUC值
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        if draw_roc:
            # 绘制ROC曲线
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
        print(f"AUC值：{roc_auc:.5f}")

    @staticmethod
    def _cal_poly(X, y, degree=2, include_linear_bias=False, include_poly_bias=False):
        """
        多项式回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param degree: int，多项式的次数。
        :param include_linear_bias: bool，是否包含线性偏置参数。
        :param include_poly_bias: bool，是否包含多项式偏置参数。
        :return: None
        """
        poly_features = PolynomialFeatures(degree=degree, include_bias=include_poly_bias)
        X_poly = poly_features.fit_transform(X)
        lin_reg = LinearRegression(fit_intercept=include_linear_bias)
        lin_reg.fit(X_poly, y)
        print(CT("多项式回归:").blue())
        print("偏置参数：", lin_reg.intercept_)  # b
        print("权重参数：", lin_reg.coef_)  # w
        print("特征名称：", poly_features.get_feature_names_out())

class SVM(CalData):
    def __init__(self, df):
        super().__init__(df)

    def cal_svc(self, X_name, y_name, draw_svm=False, **kwargs):
        """
        支持向量机
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_svm: bool，是否绘制SVM的支持向量与contourf。
        :param kwargs: 其他参数。包括：
            C: Any = 1.0,                         # 惩罚系数，C越大，容错空间越小，模型越复杂
            kernel: Any = "rbf",                  # 核函数
            degree: Any = 3,                      # 多项式核函数的次数
            gamma: Any = "scale",                 # 核函数的系数，scale的意思是1/(n_features * X.var())
            coef0: Any = 0.0,                     # 核函数的常数项
            shrinking: Any = True,                # 是否使用启发式
            probability: Any = False,             # 是否启用概率估计
            tol: Any = 1e-3,                      # 迭代停止的容忍度
            cache_size: Any = 200,                # 内核缓存的大小
            class_weight: Any = None,             # 类别权重
            verbose: Any = False,                 # 是否启用详细输出
            max_iter: Any = -1,                   # 最大迭代次数
            decision_function_shape: Any = "ovr", # 决策函数的形状
            break_ties: Any = False,              # 是否打破平局
            random_state: Any = None              # 随机种子
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_svc(X, y, draw_svm, **kwargs)

    def cal_svr(self, X_name, y_name, draw_svr=False, **kwargs):
        """
        支持向量回归
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_svr: bool，是否绘制SVR的拟合曲线。
        :param kwargs: 其他参数。包括：
             kernel: Any = "rbf",     # 核函数
             degree: Any = 3,         # 多项式核函数的次数
             gamma: Any = "scale",    # 核函数的系数，scale的意思是1/(n_features * X.var())
             coef0: Any = 0.0,        # 核函数的常数项
             tol: Any = 1e-3,         # 迭代停止的容忍度
             C: Any = 1.0,            # 惩罚系数
             epsilon: Any = 0.1,      # 不敏感区间
             shrinking: Any = True,   # 是否使用启发式
             cache_size: Any = 200,   # 内核缓存的大小
             verbose: Any = False,    # 是否启用详细输出
             max_iter: Any = -1       # 最大迭代次数
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_svr(X, y, draw_svr, **kwargs)

    def _cal_svc(self, X, y, draw_svm, **kwargs):
        """
        支持向量机
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_svm: bool，是否绘制SVM的支持向量与contourf。
        :param kwargs: 其他参数。
        :return: None
        """
        model = SVC(**kwargs)
        model.fit(X, y)
        print(CT("支持向量机-分类:").blue())
        # print("支持向量：", model.support_vectors_)
        # print("支持向量的索引：", model.support_)
        print("支持向量的个数：", model.n_support_)
        # 如果是线性核函数，可以输出权重参数
        if model.kernel == "linear":
            print("权重参数：", model.coef_)
        print("偏置参数：", model.intercept_)
        print("类别：", model.classes_)
        print("准确率：", model.score(X, y))
        if draw_svm:
            # 绘制SVM
            if X.shape[1] > 2:
                fw(self._cal_svc, "X的维度大于2，无法绘制SVM-Classification")
                return None
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(1, 1, 1)
            # 1.绘制等高线填充图
            padding = 0.1  # 设置边界范围的扩展比例
            X0_min, X0_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
            X1_min, X1_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
            x_min, x_max = X0_min - padding * (X0_max - X0_min), X0_max + padding * (X0_max - X0_min)
            y_min, y_max = X1_min - padding * (X1_max - X1_min), X1_max + padding * (X1_max - X1_min)
            # xx.shape=(100, 100)，yy.shape=(100, 100)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            # np.c_[xx.ravel(), yy.ravel()]将shape=(100, 100)转换为shape=(10000, 2)
            Z_contourf = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx, yy, Z_contourf, cmap=ListedColormap(['#FF1493', '#66ccff']), alpha=0.5)
            # 2.绘制决策边界
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            Z_contour = model.decision_function(xy).reshape(xx.shape)
            ax.contour(xx, yy, Z_contour, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
            # 3.绘制散点图
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
            # 4.绘制支持向量
            sv = model.support_vectors_
            plt.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
            plt.title("SVM Classification")
            if hasattr(X, "columns"):
                plt.xlabel(X.columns[0])
                plt.ylabel(X.columns[1])
                # 如果含有中文，设置字体为宋体
                if self.has_chinese(X.columns[0]):
                    plt.xlabel(X.columns[0], fontproperties='SimSun')
                if self.has_chinese(X.columns[1]):
                    plt.ylabel(X.columns[1], fontproperties='SimSun')
            else:
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
            plt.show()
            plt.close()

    def _cal_svr(self, X, y, draw_svr, **kwargs):
        """
        支持向量回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_svr: bool，是否绘制SVR的拟合曲线。
        :param kwargs: 其他参数。
        :return: None
        """
        model = SVR(**kwargs)
        model.fit(X, y)
        print(CT("支持向量机-回归:").blue())
        # print("支持向量：", model.support_vectors_)
        # print("支持向量的索引：", model.support_)
        print("支持向量的个数：", model.n_support_)
        # 如果是线性核函数，可以输出权重参数
        if model.kernel == "linear":
            print("权重参数：", model.coef_)
        print("偏置参数：", model.intercept_)
        print("R^2分数：", model.score(X, y))  # R^2分数越接近1，表示模型拟合得越好
        if draw_svr:
            # 绘制SVR的拟合曲线
            # 如果X超过了一维，就只绘制第一个特征
            if X.shape[1] > 1:
                fw(self._cal_svr, "X的维度大于1，无法绘制SVM-Regression")
                return None
            plt.scatter(X, y, c='b')
            plt.plot(X, model.predict(X), c='r', label='SVR')
            plt.legend()
            plt.show()
            plt.close()


def cal_linear(X, y, use="sklearn", use_bias=True):
    """
    [基本弃用，请参考Linear类]
    计算线性回归的参数θ。
    :param X:
    :param y:
    :param use:
    :param use_bias:
    :return:
    """
    if use == "formula":
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # (X^T @ X)^(-1) @ X^T @ y
        print("公式计算的θ：", theta_best)
    elif use == "sklearn":
        lin_reg = LinearRegression(fit_intercept=use_bias)
        lin_reg.fit(X, y)
        print("偏置参数：", lin_reg.intercept_)
        print("权重参数：", lin_reg.coef_)


def cal_skew_kurtosis(data_series):
    """
    [用的很少，暂时不做修订了]
    计算数据列的偏度、峰度以及正态分布程度检验结果。
    标准正态分布偏度和峰度均为0。如果峰度绝对值小于10并且偏度绝对值小于3，说明数据基本可接受为正态分布。
    1.skew: float，偏度。
      偏度：描述数据分布形态偏斜程度的统计量。
        \text{Skewness} = \frac{E[(X-\mu)^3]}{\sigma^3}
        偏度>0:数据分布右偏（正偏，众数<中位数<均值）。偏度<0:表示左偏（负偏，众数>中位数>均值）。偏度=0:对称
      偏度标准误： sqrt( 6*N*(N-1)/((N-2)*(N+1)*(N+3)) )
    2.skewtest: tuple，偏度检验的结果。如果 Z 分数的绝对值较大，且 p 值小于显著性水平（通常设定为 0.05），就可以拒绝原假设
      statistic：z-score是一个标准化的分数，用于衡量一个数据点与其所在数据集均值的偏离程度，通常用于假设检验。
        单个样本的z-score：z = \frac{X-\mu}{\sigma} ]
        多个样本的z-score：Z = \frac{g_1}{\sqrt{\frac{6n(n-1)}{(n-2)(n+1)(n+3)}}}，式中g_1为样本偏度，n为样本大小
      pvalue：p值是在假设检验中用于衡量观察到的数据与某一假设模型一致程度的概率。p值越小(通常设定显著性水平为0.05)，
        表示观察到的数据越不可能出现在原假设的情况下，从而提供了拒绝原假设(null hypothesis)的证据。
    kurtosis: float，峰度。
      峰度标准误 = 4*(N**2 -1)*偏度标准误 /((N-3)*(N+5))
    normaltest: tuple，正态分布程度检验结果，包含统计量和 p-value。
    :param data_series: pandas Series，要计算统计量的数据列。
    """
    n = len(data_series)

    skew = stats.skew(data_series)  # 偏度
    skew_se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))  # 偏度标准误
    skewtest = stats.skewtest(data_series)  # 偏度检验的结果
    kurtosis = stats.kurtosis(data_series)  # 峰度
    kurtosis_se = np.sqrt(24 * n * (n - 1) / ((n - 2) * (n - 3) * (n + 5) * (n + 7)))  # 峰度标准误
    kurtosistest = stats.kurtosistest(data_series)  # 峰度检验
    normaltest = stats.normaltest(data_series)  # 正态分布程度检验

    print("偏度", skew)
    print("偏度标准误", skew_se)
    print("偏度检验", skewtest)
    print("峰度", kurtosis)
    print("峰度标准误", kurtosis_se)
    print("峰度检验", kurtosistest)
    print("正态分布程度检验", normaltest)


def cal_net(X_train, y_train, net=None, lr=0.001, batch_size=16, num_epochs=100, use_bias=True, loss_fuc='MSE',
            optim_fun="SGD", show_interval=10):
    """
    [基本弃用，请参考easier_nn]
    :param X_train:
    :param y_train:
    :param net:
    :param lr:
    :param batch_size:
    :param num_epochs:
    :param use_bias:
    :param loss_fuc: 均方误差"MSE"适用回归问题; 交叉熵适用分类，衡量两个概率分布之间的差异。二分类问题(标签是0,1):
        二元交叉熵损失"BCE"; 多分类问题(标签是one-hot编码的向量):类别交叉熵损失"CE"。
    :param optim_fun: 随机梯度下降"SGD"; 结合动量和指数衰减"Adam"
    :param show_interval:
    :return: net
    """
    if net is None:
        net = nn.Sequential(nn.Flatten(),
                            nn.Linear(X_train.shape[1], 1, bias=use_bias))
        net[1].weight.data.normal_(0, 0.01)
        if use_bias:
            net[1].bias.data.fill_(0)

    if loss_fuc == "MSE":
        loss = nn.MSELoss()
    elif loss_fuc == "BCE":
        loss = nn.BCELoss()
    elif loss_fuc == "CE":
        loss = nn.CrossEntropyLoss()
    else:
        print("暂不支持此种损失函数，这里使用MSE代替")
        loss = nn.MSELoss()

    if optim_fun == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif optim_fun == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        print("暂不支持此种优化方法，这里使用SGD代替")
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if isinstance(X_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
        data_iter = load_array((X_train, y_train), batch_size)
    elif isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        data_iter = load_array((X_train, y_train), batch_size)
    else:
        # 如果数据类型不符合要求，可以抛出错误或者做其他处理
        raise TypeError("Input data type not supported.")

    for epoch in range(num_epochs):
        for X, y in data_iter:
            loss_value = loss(net(X), y)  # 这里的net返回输入x经过定义的网络所计算出的值
            optimizer.zero_grad()  # 清除上一次的梯度值
            loss_value.backward()  # 反向传播，求参数的梯度
            # for param in net.parameters():
            #     print(param.grad)
            optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
        if epoch % show_interval == 0:
            loss_value = loss(net(X_train), y_train)
            print(f'epoch {epoch + 1}, loss {loss_value:f}')

    # w1 = net[1].weight.data
    # w3 = net[3].weight.data
    # print('w1:\n', w1, '\nw2:\n', w3)
    # b1 = net[1].bias.data
    # print('b1:', b1, '\nb2:None')

    return net



