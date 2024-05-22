import math
import numpy as np
import pandas as pd
from scipy import stats
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    :param data_arrays: 一个包含数据数组的元组或列表。通常包括输入特征和对应的标签(features, labels)。
    :param batch_size: 每个小批量样本的数量。
    :param is_train: True数据将被随机洗牌(用于训练);False数据将按顺序提供(用于模型的评估或测试)。
    """
    dataset = data.TensorDataset(*data_arrays)  # 将数据数组转换为TensorDataset对象(将数据存储为Tensor对象，并允许按索引访问)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def cal_skew_kurtosis(data_series):
    """
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


def cal_linear(X, y, use="sklearn", use_bias=True):
    if use == "formula":
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # (X^T @ X)^(-1) @ X^T @ y
        print("公式计算的θ：", theta_best)
    elif use == "sklearn":
        lin_reg = LinearRegression(fit_intercept=use_bias)
        lin_reg.fit(X, y)
        print("偏置参数：", lin_reg.intercept_)
        print("权重参数：", lin_reg.coef_)



