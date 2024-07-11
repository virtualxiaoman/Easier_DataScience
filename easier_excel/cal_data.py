import math
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy import stats
from scipy.stats import shapiro, kstest, normaltest, anderson, chisquare

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc, classification_report, matthews_corrcoef, hamming_loss, confusion_matrix, \
    f1_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import torch
from torch import nn
from torch.utils import data


from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.Colorful_Console import func_warning as fw
from easier_tools.to_md import ToMd
from easier_excel.draw_data import draw_scatter

ToMd = ToMd()


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
        self.y_pred = None  # 预测值，np.ndarray类型
        self.ACC = None  # 准确率，float类型
        self.F1_weight = None  # F1分数，float类型，取各类别F1分数的加权平均值
        self.F1_unweighted = None  # F1分数，float类型，取各类别F1分数的平均值
        self.MSE = None  # 均方误差MSE，float类型

    @staticmethod
    # 判断文本中是否含有中文字符
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
        # 模型
        self.lin_reg = None  # 线性回归模型
        self.log_reg = None  # 逻辑回归模型
        self.poly_features = None  # 多项式特征转换器，多项式回归模型还包括了self.lin_reg

        # 预测
        self.y_pred = None  # 预测值，np.ndarray类型

        # 统计量
        self.weight = None  # 权重参数，DataFrame类型
        self.MAE = None  # 平均绝对误差MAE，float类型
        self.MSE = None  # 均方误差MSE，float类型
        self.RMSE = None  # 均方根误差RMSE，float类型
        self.residuals = None  # 残差residuals，np.ndarray类型
        self.AUC = None  # ROC曲线下的面积AUC，float类型
        self.R_squared = None  # R^2分数，float类型
        self.adjusted_R_squared = None  # 调整后的R^2分数，float类型
        self.MCC = None  # Matthews相关系数MCC，float类型

        # 检验参数
        self.VIF = None  # 变量膨胀因子VIF，DataFrame类型
        self.breusch_pagan_pvalue = None  # Breusch-Pagan检验的p值，float类型
        self.shapiro_pvalue = None  # Shapiro-Wilk检验的p值，float类型
        self.dagostino_pvalue = None  # D'Agostino's K² P值，float类型
        self.durbin_watson_statistic = None  # Durbin-Watson统计量，float类型

    def cal_linear(self, X_name, y_name, use_bias=True, detailed=True, **kwargs):
        """
        线性回归。
        公式：y = X @ w + b
        假设：
          无异常值(要求在使用cal_linear之前已经处理了异常值，这里不会处理异常值)
          线性(X,Y的关系是线性的)，
          同方差(误差项的方差不随X的不同而变化)，
          正态(在任何固定的X值下，残差应该呈正态分布，且均值为0)，
          无自相关(残差不应随时间或其他自变量的顺序发生系统性变化。如时间序列，残差不应随着时间推移表现出某种趋势)，
          无多重共线性(多个自变量之间不应存在线性关系，否则会导致模型系数不稳定，难以确定哪个变量在预测因变量中起主要作用)。
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param use_bias: bool，是否使用偏置参数b。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, 1)
        self._cal_linear(X, y, use_bias, **kwargs)

    def cal_logistic(self, X_name, y_name, use_bias=True, pos_label=1, detailed=True, **kwargs):
        """
        逻辑回归。
        公式：y = 1 / (1 + exp(-X @ w + b))
        二分类：
        \hat{y}=\sigma(X\cdot\mathbf{w}+b)
        多分类：
        \hat{y}_{i}=\frac{e^{\left(X \cdot \mathbf{w}_{i}+b_{i}\right)}}{\sum_{j=1}^{C} e^{\left(X \cdot \mathbf{w}_{j}+b_{j}\right)}}
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param use_bias: bool，是否使用偏置参数b。
        :param pos_label: int，正类别标签。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, )
        self._cal_logistic(X, y, use_bias, pos_label, **kwargs)

    def cal_poly(self, X_name, y_name, degree=2, include_linear_bias=False, include_poly_bias=False, detailed=True,
                 **kwargs):
        """
        多项式回归。因为使用的不多，这里不进行统计量的计算与检验，仅给出模型的拟合。
        [Tips]:
            比如X的shape是(8,3), degree=3, include_bias=True，那么X_poly的shape是(8, 20)，其中包含了bias项。具体顺序如下：
            ['1'
             'x0' 'x1' 'x2'
             'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2'
             'x0^3' 'x0^2 x1' 'x0^2 x2' 'x0 x1^2' 'x0 x1 x2' 'x0 x2^2' 'x1^3' 'x1^2 x2' 'x1 x2^2' 'x2^3']
             你可以通过以下代码验证：
                X = np.arange(24).reshape(8, 3)
                poly = PolynomialFeatures(3)
                print(poly.fit_transform(X))  # 进行多项式特征转换
                print(poly.fit_transform(X).shape)  # shape
                print(poly.get_feature_names_out())  # 转换后的特征名称
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param degree: int，多项式的次数。
        :param include_linear_bias: bool，是否包含线性偏置参数。
        :param include_poly_bias: bool，是否包含多项式偏置参数。
        """
        X = self.df[X_name]
        y = self.df[y_name].values.reshape(-1, 1)
        self._cal_poly(X, y, degree, include_linear_bias, include_poly_bias,  **kwargs)

    def _cal_linear(self, X, y, use_bias=True, detailed=True, md_flag=False, **kwargs):
        """
        线性回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param use_bias: bool，是否使用偏置参数。
        :return: None
        """
        # 线性回归的fit部分
        self.lin_reg = LinearRegression(fit_intercept=use_bias)
        self.lin_reg.fit(X, y)
        print(CT("[LinearRegression]线性回归:").blue())
        print("偏置参数：", self.lin_reg.intercept_)  # b
        print("权重参数：", self.lin_reg.coef_)  # w
        self.y_pred = self.lin_reg.predict(X)  # 预测值

        # 得到权重的DataFrame
        self.weight = pd.DataFrame({"feature": X.columns, "weight": self.lin_reg.coef_[0]})
        bias_df = pd.DataFrame({"feature": ["bias"], "weight": self.lin_reg.intercept_})
        self.weight = pd.concat([bias_df, self.weight], ignore_index=True)
        ToMd.text_to_md("线性回归权重", md_flag)
        ToMd.df_to_md(self.weight, md_flag, md_index=True)

        # 计算部分统计量
        self.residuals = y - self.y_pred
        self.MAE = np.mean(np.abs(self.residuals))
        self.MSE = np.mean(self.residuals ** 2)
        self.RMSE = math.sqrt(self.MSE)

        if detailed:
            # 线性：计算R²和调整后的R²
            self.__check_linearity(X, y, self.lin_reg, md_flag)
            # 同方差性: 使用Breusch-Pagan检验
            self.__check_homoscedasticity(X, y, md_flag)
            # 正态性: 使用Shapiro-Wilk检验和D'Agostino's K²检验
            self.__check_normality(self.residuals, md_flag)
            # 自相关性: 使用Durbin-Watson检验
            self.__check_autocorrelation(self.residuals, md_flag)
            # 多重共线性: 使用VIF检测
            self.__check_multicollinearity(X, use_bias, md_flag)

    def _cal_logistic(self, X, y, use_bias=True, pos_label=1, draw_roc=False, detailed=True, md_flag=False):
        """
        逻辑回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param pos_label: int，正类别标签。
        :return: None
        """
        # neg_label是y里面除了pos_label之外的另外的全部标签
        neg_label = [i for i in set(y) if i != pos_label]
        labels = [pos_label] + neg_label

        self.log_reg = LogisticRegression(fit_intercept=use_bias)
        self.log_reg.fit(X, y)

        print(CT("[LogisticRegression]逻辑回归:").blue())
        print("偏置参数：", self.log_reg.intercept_)  # b
        print("权重参数：", self.log_reg.coef_)  # w, shape:(C, D)。C是类别数，D是特征数
        print("类别：", self.log_reg.classes_)  # 类别
        self.ACC = self.log_reg.score(X, y)
        print("准确率：", self.ACC)
        self.y_pred = self.log_reg.predict(X)  # 预测值
        self.y_pred_prob = self.log_reg.predict_proba(X)[:, 1]  # 预测概率
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')  # F1分数
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")

        # 如果是二分类
        if len(labels) == 2:
            # 二分类的权重
            self.weight = pd.DataFrame({"feature": X.columns, "weight": self.log_reg.coef_[0]})
            print(self.weight)
            bias_df = pd.DataFrame({"feature": ["bias"], "weight": self.log_reg.intercept_})
            print(bias_df)
            self.weight = pd.concat([bias_df, self.weight], ignore_index=True)
        else:
            # 多分类的权重
            intercept_reshaped = self.log_reg.intercept_.reshape(1, -1)  # (C,) -> (1,C)
            coef_with_bias = np.vstack([intercept_reshaped, np.array(self.log_reg.coef_).T])  # (1,C) + (D,C) -> (D+1,C)
            features_with_bias = ["bias"] + X.columns.tolist()  # (D,) -> (D,) -> (D+1,)
            print(intercept_reshaped.shape)
            print(np.array(self.log_reg.coef_).T.shape)
            print(coef_with_bias.shape)
            print(features_with_bias)
            # self.weight的shape是(D+1,C)，加上 index(features_with_bias) 和 columns(self.log_reg.classes_)
            self.weight = pd.DataFrame(coef_with_bias, index=features_with_bias, columns=self.log_reg.classes_)
        ToMd.text_to_md("逻辑回归权重", md_flag)
        ToMd.df_to_md(self.weight, md_flag, md_index=True)

        print("分类报告：")
        classify_report = classification_report(y, self.y_pred)
        print(classify_report)
        print(f"混淆矩阵 -- labels = {labels}：")
        conf_matrix = confusion_matrix(y, self.y_pred, labels=labels)
        print(conf_matrix)

        ToMd.text_to_md(f"LogisticRegression 训练集准确率: {self.ACC}", md_flag, md_color='blue')
        ToMd.df_to_md(pd.DataFrame(classification_report(y, self.y_pred, output_dict=True)).transpose(), md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(conf_matrix, index=labels, columns=labels), md_flag, md_index=True)

        # 计算部分统计量
        self.residuals = y - self.y_pred
        self.MAE = np.mean(np.abs(self.residuals))
        self.MSE = np.mean(self.residuals ** 2)
        self.RMSE = math.sqrt(self.MSE)

        if detailed:
            # 线性：计算R²和调整后的R²
            self.__check_linearity(X, y, self.log_reg, md_flag)
            # 同方差性: 使用Breusch-Pagan检验
            self.__check_homoscedasticity(X, y, md_flag)
            # 正态性: 使用Shapiro-Wilk检验和D'Agostino's K²检验
            self.__check_normality(self.residuals, md_flag)
            # 自相关性: 使用Durbin-Watson检验
            self.__check_autocorrelation(self.residuals, md_flag)
            # 多重共线性: 使用VIF检测
            self.__check_multicollinearity(X, use_bias, md_flag)
            # 计算MCC值
            self.__check_mcc(y, self.y_pred, md_flag)
            # 计算汉明损失
            self.__check_hamming_loss(y, self.y_pred, md_flag)
            # 计算ROC曲线和AUC值
            self.__check_auc(X, y, self.log_reg, pos_label, draw_roc, md_flag)

    def _cal_poly(self, X, y, degree=2, include_linear_bias=False, include_poly_bias=False, detailed=True, md_flag=False):
        """
        多项式回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param degree: int，多项式的次数。
        :param include_linear_bias: bool，是否包含线性偏置参数。
        :param include_poly_bias: bool，是否包含多项式偏置参数。
        :return: None
        """
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_poly_bias)
        X_poly = self.poly_features.fit_transform(X)
        self.lin_reg = LinearRegression(fit_intercept=include_linear_bias)
        self.lin_reg.fit(X_poly, y)
        print(CT("[PolynomialFeatures]多项式回归:").blue())
        print("偏置参数：", self.lin_reg.intercept_)  # b
        print("权重参数：", self.lin_reg.coef_)  # w
        print("特征名称：", self.poly_features.get_feature_names_out())

        self.y_pred = self.lin_reg.predict(X_poly)  # 预测值
        self.residuals = y - self.y_pred  # 残差
        self.MAE = np.mean(np.abs(self.residuals))
        self.MSE = np.mean(self.residuals ** 2)
        self.RMSE = math.sqrt(self.MSE)

        # 权重DataFrame
        self.weight = pd.DataFrame({"feature": self.poly_features.get_feature_names_out(),
                                    "weight": self.lin_reg.coef_[0]})
        bias_df = pd.DataFrame({"feature": ["bias"], "weight": self.lin_reg.intercept_})
        self.weight = pd.concat([bias_df, self.weight], ignore_index=True)
        ToMd.text_to_md("多项式回归权重", md_flag)
        ToMd.df_to_md(self.weight, md_flag, md_index=True)

    # 绘制ROC曲线
    def __draw_roc_curve(self, fpr, tpr, draw_roc, md_flag):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {self.AUC:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        ToMd.pic_to_md(plt, md_flag, md_title="ROC")
        if draw_roc:
            plt.show()
        plt.close()

    # 线性：计算R²和调整后的R²
    def __check_linearity(self, X, y, reg, md_flag):
        """
        X,Y的关系是线性的
        R^{2}=1-\frac{\sum\left(y_{i}-\hat{y}_{i}\right)^{2}}{\sum\left(y_{i}-\bar{y}\right)^{2}}
        R_{\text{adjusted}}^2=1-\left(1-R^2\right)\cdot\frac{n-1}{n-p-1}，式中n是样本数，p是特征数。
        通常R²超过0.7被认为有较好的线性拟合。
        """
        formula1 = r"R^{2}=1-\frac{\sum\left(y_{i}-\hat{y}_{i}\right)^{2}}{\sum\left(y_{i}-\bar{y}\right)^{2}}"
        formula2 = r"R_{\text{adjusted}}^2=1-\left(1-R^2\right)\cdot\frac{n-1}{n-p-1}"

        # 线性关系检查：通过R²和调整后的R²
        self.R_squared = reg.score(X, y)
        n = X.shape[0]
        p = X.shape[1]
        self.adjusted_R_squared = 1 - (1 - self.R_squared) * ((n - 1) / (n - p - 1))

        print(f"R²: {self.R_squared}, 调整后的R²: {self.adjusted_R_squared}")
        ToMd.text_to_md(f"R²的公式是： $ {formula1} $, $ {formula2} $", md_flag)
        ToMd.text_to_md(f"R²: {self.R_squared}, 调整后的R²: {self.adjusted_R_squared}", md_flag)

        if self.R_squared > 0.7:
            print(CT("√ 线性关系良好[R²]").green())
            ToMd.text_to_md(f"√ 线性关系良好[R²]", md_flag, md_color='green')
        else:
            print(CT("× 线性关系较差[R²]").red())
            ToMd.text_to_md(f"× 线性关系较差[R²]", md_flag, md_color='red')

    # 同方差性：使用Breusch-Pagan检验
    def __check_homoscedasticity(self, X, y, md_flag):
        """
        误差项的方差不随X的不同而变化
        LM=\frac n2\cdot R_{\mathrm{aux}}^2
        P 值大于 0.05，接受同方差性假设；否则，拒绝同方差性假设。
        """
        formula = r"LM=\frac n2\cdot R_{\mathrm{aux}}^2"

        model = sm.OLS(y, sm.add_constant(X)).fit()  # 拟合最小二乘模型
        test = het_breuschpagan(model.resid, model.model.exog)
        self.breusch_pagan_pvalue = test[1]
        print(f"Breusch-Pagan P值（同方差性）: {self.breusch_pagan_pvalue}")

        ToMd.text_to_md(f"同方差性的公式是： $ {formula} $", md_flag)
        ToMd.text_to_md(f"Breusch-Pagan P值（同方差性）: {self.breusch_pagan_pvalue}", md_flag)

        if self.breusch_pagan_pvalue > 0.05:
            print(CT("√ 同方差性假设成立[Breusch-Pagan]").green())
            ToMd.text_to_md(f"√ 同方差性假设成立[Breusch-Pagan]", md_flag, md_color='green')
        else:
            print(CT("× 存在异方差性[Breusch-Pagan]").red())
            ToMd.text_to_md(f"× 存在异方差性[Breusch-Pagan]", md_flag, md_color='red')

    # 正态性：使用Shapiro-Wilk检验和D'Agostino's K²检验
    def __check_normality(self, residuals, md_flag):
        """
        误差项的分布应该是正态分布的
        Shapiro-Wilk: W=\frac{\left(\sum a_ix_{(i)}\right)^2}{\sum\left(x_i-\bar{x}\right)^2}，
        其中a_i是常数，x_{(i)}是排序后的数据。
        D'Agostino's K²：K^2=\frac{(\text{Skewness}^2+\text{Kurtosis}^2)}2
        :return:
        """
        formula1 = r"Shapiro-Wilk: W=\frac{\left(\sum a_ix_{(i)}\right)^2}{\sum\left(x_i-\bar{x}\right)^2}"
        formula2 = r"D'Agostino's K²：K^2=\frac{(\text{Skewness}^2+\text{Kurtosis}^2)}2"

        self.shapiro_pvalue = shapiro(residuals)[1]
        print(f"Shapiro-Wilk P值（正态性）: {self.shapiro_pvalue}")
        ToMd.text_to_md(f"正态性的公式是： $ {formula1} $, $ {formula2} $", md_flag)
        ToMd.text_to_md(f"Shapiro-Wilk P值（正态性）: {self.shapiro_pvalue}", md_flag)

        if self.shapiro_pvalue > 0.05:
            print(CT("√ 残差正态性假设成立[Shapiro-Wilk]").green())
            ToMd.text_to_md("√ 残差正态性假设成立[Shapiro-Wilk]", md_flag, md_color='green')
        else:
            print(CT("× 残差不符合正态性假设[Shapiro-Wilk]").red())
            ToMd.text_to_md("× 残差不符合正态性假设[Shapiro-Wilk]", md_flag, md_color='red')

        dagostino_statistic, dagostino_pvalue = normaltest(residuals)
        self.dagostino_pvalue = dagostino_pvalue
        print(f"D'Agostino's K² P值（正态性）: {self.dagostino_pvalue}")
        ToMd.text_to_md(f"D'Agostino's K² P值（正态性）: {self.dagostino_pvalue}", md_flag)
        if self.dagostino_pvalue > 0.05:
            print(CT("√ 残差正态性假设成立[D'Agostino's K²]").green())
            ToMd.text_to_md("√ 残差正态性假设成立[D'Agostino's K²]", md_flag, md_color='green')
        else:
            print(CT("× 残差不符合正态性假设[D'Agostino's K²]").red())
            ToMd.text_to_md("× 残差不符合正态性假设[D'Agostino's K²]", md_flag, md_color='red')

    # 自相关性：使用Durbin-Watson检验
    def __check_autocorrelation(self, residuals, md_flag):
        """
        残差不应随时间或其他自变量的顺序发生系统性变化
        d=\frac{\sum_{t=2}^n(e_t-e_{t-1})^2}{\sum_{t=1}^ne_t^2}
        计算相邻残差的差值平方和。
        值接近 2，表示没有自相关性；接近 0，表示存在正自相关；接近 4，表示存在负自相关。
        """
        formula = r"d=\frac{\sum_{t=2}^n(e_t-e_{t-1})^2}{\sum_{t=1}^ne_t^2}"

        self.durbin_watson_statistic = sm.stats.stattools.durbin_watson(residuals)
        print(f"Durbin-Watson 统计量（自相关性）: {self.durbin_watson_statistic}")
        ToMd.text_to_md(f"自相关性的公式是： $ {formula} $", md_flag)
        ToMd.text_to_md(f"Durbin-Watson 统计量（自相关性）: {self.durbin_watson_statistic}", md_flag)
        if 1.5 < self.durbin_watson_statistic < 2.5:
            print(CT("√ 没有显著的自相关性[Durbin-Watson]").green())
            ToMd.text_to_md("√ 没有显著的自相关性[Durbin-Watson]", md_flag, md_color='green')
        else:
            print(CT("× 存在自相关性[Durbin-Watson]").red())
            ToMd.text_to_md("× 存在自相关性[Durbin-Watson]", md_flag, md_color='red')

    # 多重共线性：使用VIF检测
    def __check_multicollinearity(self, X, use_bias, md_flag):
        """
        多个自变量之间不应存在线性关系
        VIF_i=\frac1{1-R_i^2}
        VIF 值小于 5，一般认为没有多重共线性问题；大于 10，表明存在严重的多重共线性问题。
        """
        formula = r"VIF_i=\frac1{1-R_i^2}"

        if use_bias:
            X_const = sm.add_constant(X)
        else:
            X_const = X

        vif_data = pd.DataFrame()
        vif_data["feature"] = X_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        self.VIF = vif_data
        print(f"VIF（多重共线性）:")
        print(self.VIF)
        ToMd.text_to_md(f"VIF（多重共线性）的公式是： $ {formula} $", md_flag)
        ToMd.df_to_md(self.VIF, md_flag, md_index=True)

        vif_no_const = self.VIF[self.VIF["feature"] != "const"]
        if all(vif_no_const["VIF"] < 5):
            print(CT("√ 没有严重的多重共线性问题[VIF]").green())
            ToMd.text_to_md("√ 没有严重的多重共线性问题[VIF]", md_flag, md_color='green')
        else:
            print(CT("× 存在多重共线性问题[VIF]").red())
            ToMd.text_to_md("× 存在多重共线性问题[VIF]", md_flag, md_color='red')

    # MCC：计算MCC值
    def __check_mcc(self, y_true, y_pred, md_flag):
        """
        计算MCC值
        MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP) \times (TP + FN) \times (TN + FP) \times (TN + FN)}}
        值接近 1 表示分类性能好，值接近 -1 表示分类性能差。
        """
        formula = r"MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP) \times (TP + FN) \times (TN + FP) \times (TN + FN)}}"

        self.mcc = matthews_corrcoef(y_true, y_pred)
        print(f"Matthews 相关系数（MCC）: {self.mcc}")
        ToMd.text_to_md(f"MCC的公式是： $ {formula} $", md_flag)
        ToMd.text_to_md(f"Matthews 相关系数（MCC）: {self.mcc}", md_flag)
        if self.mcc > 0.5:
            print(CT("√ 分类性能良好[MCC]").green())
            ToMd.text_to_md("√ 分类性能良好[MCC]", md_flag, md_color='green')
        elif self.mcc > 0:
            print(CT("△ 分类性能一般[MCC]").red())
            ToMd.text_to_md("△ 分类性能一般[MCC]", md_flag, md_color='red')
        else:
            print(CT("× 分类性能极差[MCC]").red())
            ToMd.text_to_md("× 分类性能极差[MCC]", md_flag, md_color='red')

    # 汉明损失：Hamming Loss
    def __check_hamming_loss(self, y_true, y_pred, md_flag):
        """
        计算汉明损失
        Hamming Loss = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{L} \sum_{j=1}^{L} \mathbb{I} \left (y_{ij} \ne \hat{y}_{ij} \right)，
        其中N是样本数量，L是标签数量，\mathbb{I} 是指示函数。
        汉明损失表示标签预测错误的比例，值越小越好。
        """
        formula = r"Hamming Loss = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{L} \sum_{j=1}^{L} \mathbb{I} \left (y_{ij} \ne \hat{y}_{ij} \right)"
        self.hamming_loss = hamming_loss(y_true, y_pred)
        print(f"汉明损失（Hamming Loss）: {self.hamming_loss}")

        ToMd.text_to_md(f"MCC的公式是： $ {formula} $", md_flag)
        ToMd.text_to_md(f"汉明损失（Hamming Loss）: {self.hamming_loss}", md_flag)
        if self.hamming_loss < 0.1:
            print(CT("√ 汉明损失较低[Hamming Loss]").green())
            ToMd.text_to_md("√ 汉明损失较低[Hamming Loss]", md_flag, md_color='green')
        else:
            print(CT("× 汉明损失较高[Hamming Loss]").red())
            ToMd.text_to_md("× 汉明损失较高[Hamming Loss]", md_flag, md_color='red')

    # AUC：计算ROC曲线和AUC值
    def __check_auc(self, X, y, log_reg, pos_label, draw_roc, md_flag):
        y_pred_prob = log_reg.predict_proba(X)[:, 1]  # 计算预测概率
        # 计算ROC曲线和AUC值
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob, pos_label=pos_label)
        self.AUC = auc(fpr, tpr)
        self.__draw_roc_curve(fpr, tpr, draw_roc, md_flag)
        print(f"ROC曲线下的面积（AUC）: {self.AUC}")
        ToMd.text_to_md(f"ROC曲线下的面积（AUC）: {self.AUC}", md_flag)
        if self.AUC > 0.8:
            print(CT("√ AUC值较高[AUC]").green())
            ToMd.text_to_md("√ AUC值较高[AUC]", md_flag, md_color='green')
        else:
            print(CT("× AUC值较低[AUC]").red())
            ToMd.text_to_md("× AUC值较低[AUC]", md_flag, md_color='red')


class SVM(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.svc = None  # 支持向量机分类模型
        self.svr = None  # 支持向量机回归模型

        self.y_pred = None  # 预测值，np.ndarray类型
        self.residuals = None  # 残差，np.ndarray类型

    def cal_svc(self, X_name, y_name, draw_svc=False, md_flag=False, **kwargs):
        """
        支持向量机
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_svc: bool，是否绘制SVM的支持向量与contourf。
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
        y = pd.Series(self.df[y_name], name=y_name)  # 我也不知道为什么不指定name就没name了，不过也只影响绘图
        self._cal_svc(X, y, draw_svc, md_flag, **kwargs)

    def cal_svr(self, X_name, y_name, draw_svr=False, md_flag=False, **kwargs):
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
        y = pd.Series(self.df[y_name], name=y_name)  # 我也不知道为什么不指定name就没name了，不过也只影响绘图
        self._cal_svr(X, y, draw_svr, md_flag, **kwargs)

    def _cal_svc(self, X, y, draw_svc, md_flag, **kwargs):
        """
        支持向量机
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_svc: bool，是否绘制SVM的支持向量与contourf。
        :param kwargs: 其他参数。
        :return: None
        """
        self.svc = SVC(**kwargs)
        self.svc.fit(X, y)
        print(CT("支持向量机-分类:").blue())
        # print("支持向量：", self.svc.support_vectors_)
        # print("支持向量的索引：", self.svc.support_)
        print("支持向量的个数：", self.svc.n_support_)
        # 如果是线性核函数，可以输出权重参数
        if self.svc.kernel == "linear":
            print("权重参数：", self.svc.coef_)
        print("偏置参数：", self.svc.intercept_)
        print("类别：", self.svc.classes_)
        self.ACC = self.svc.score(X, y)
        print("准确率：", self.ACC)
        ToMd.text_to_md(f"SVC 准确率: {self.ACC}", md_flag, md_color='blue')
        self.y_pred = self.svc.predict(X)  # 预测值
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')  # F1分数
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")
        if draw_svc:
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
            Z_contourf = self.svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx, yy, Z_contourf, cmap=ListedColormap(['#FF1493', '#66ccff']), alpha=0.5)
            # 2.绘制决策边界
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            Z_contour = self.svc.decision_function(xy).reshape(xx.shape)
            ax.contour(xx, yy, Z_contour, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
            # 3.绘制散点图
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
            # 4.绘制支持向量
            sv = self.svc.support_vectors_
            plt.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
            plt.title("SVM Classification")
            if hasattr(X, "columns"):
                # 如果含有中文，设置字体为宋体
                if self.has_chinese(X.columns[0]):
                    plt.xlabel(X.columns[0], fontproperties='SimSun')
                else:
                    plt.xlabel(X.columns[0])
                if self.has_chinese(X.columns[1]):
                    plt.ylabel(X.columns[1], fontproperties='SimSun')
                else:
                    plt.ylabel(X.columns[1])
            else:
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
            plt.show()
            plt.close()

    def _cal_svr(self, X, y, draw_svr, md_flag, **kwargs):
        """
        支持向量回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_svr: bool，是否绘制SVR的拟合曲线。
        :param kwargs: 其他参数。
        :return: None
        """
        self.svr = SVR(**kwargs)
        self.svr.fit(X, y)
        print(CT("支持向量机-回归:").blue())
        # print("支持向量：", model.support_vectors_)
        # print("支持向量的索引：", model.support_)
        print("支持向量的个数：", self.svr.n_support_)
        # 如果是线性核函数，可以输出权重参数
        if self.svr.kernel == "linear":
            print("权重参数：", self.svr.coef_)
        print("偏置参数：", self.svr.intercept_)
        print("R²分数：", self.svr.score(X, y))  # R^2分数越接近1，表示模型拟合得越好
        self.y_pred = self.svr.predict(X)  # 预测值
        self.residuals = y - self.y_pred  # 残差
        self.MAE = np.mean(np.abs(self.residuals))
        self.MSE = np.mean(self.residuals ** 2)
        self.RMSE = math.sqrt(self.MSE)

        ToMd.text_to_md(f"SVR R²分数: {self.svr.score(X, y)}", md_flag, md_color='blue')

        if draw_svr:
            # 绘制SVR的拟合曲线
            # 如果X超过了一维，就只绘制第一个特征
            if X.shape[1] > 1:
                fw(self._cal_svr, "X的维度大于1，无法绘制SVM-Regression")
                return None
            plt.scatter(X, y, c='b')
            plt.plot(X, self.svr.predict(X), c='r', label='SVR')
            plt.title("SVR Regression")
            if hasattr(X, "columns"):
                if self.has_chinese(X.columns[0]):
                    plt.xlabel(X.columns[0], fontproperties='SimSun')
                else:
                    plt.xlabel(X.columns[0])
            if hasattr(y, "name"):
                if self.has_chinese(y.name):
                    plt.ylabel(y.name, fontproperties='SimSun')
                else:
                    plt.ylabel(y.name)
            plt.legend()
            plt.show()
            plt.close()


class Tree(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.tree = None
        self.y_pred = None
        # 按特征重要性降序排列的特征
        self.feature_by_importance = None

    def cal_tree(self, X_name, y_name, draw_tree=False, pos_label=1, **kwargs):
        """
        决策树
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_tree: bool，是否绘制决策树。
        :param pos_label: int，正类别标签。
        :param kwargs: 其他参数。包括：
            criterion: Any = "gini",              # 选择特征的标准，包括{"gini", "entropy", "log_loss"}，对应CART、C4.5
            splitter: Any = "best",               # 选择分裂节点的策略，包括{"best", "random"}
            max_depth: Any = None,                # 树的最大深度
            min_samples_split: Any = 2,           # 内部节点再划分所需最小样本数
            min_samples_leaf: Any = 1,            # 叶子节点最少样本数
            min_weight_fraction_leaf: Any = 0.0,  # 叶子节点的样本权重和的最小加权分数
            max_features: Any = None,             # 每次分裂的最大特征数
            random_state: Any = None,             # 随机种子
            max_leaf_nodes: Any = None,           # 最大叶子节点数
            min_impurity_decrease: Any = 0.0,     # 分裂节点的最小不纯度
            class_weight: Any = None,             # 类别权重
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_tree(X, y, draw_tree, pos_label, **kwargs)

    def cal_random_forest(self, X_name, y_name, pos_label=1, **kwargs):
        """
        随机森林
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param pos_label: int，正类别标签。
        :param kwargs: 其他参数。包括：
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_random_forest(X, y, pos_label, **kwargs)

    def _cal_tree(self, X, y, draw_tree, pos_label, md_flag=False, **kwargs):
        """
        决策树
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_tree: bool，是否绘制决策树。
        :param kwargs: 其他参数。
        :return: None
        """
        neg_label = [i for i in set(y) if i != pos_label]
        labels = [pos_label] + neg_label

        self.tree = tree.DecisionTreeClassifier(**kwargs)
        self.tree.fit(X, y)
        print(CT("决策树:").blue())
        print("特征重要性：", self.tree.feature_importances_)
        self.ACC = self.tree.score(X, y)
        print("准确率：", self.ACC)
        self.y_pred = self.tree.predict(X)
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')  # F1分数
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")

        print("分类报告：")
        classify_report = classification_report(y, self.y_pred)
        print(classify_report)
        print(f"混淆矩阵 -- labels = {labels}：")
        conf_matrix = confusion_matrix(y, self.y_pred, labels=labels)
        print(conf_matrix)

        ToMd.text_to_md(f"DecisionTreeClassifier 训练集准确率: {self.ACC}", md_flag, md_color='blue')
        ToMd.df_to_md(pd.DataFrame(classification_report(y, self.y_pred, output_dict=True)).transpose(), md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(conf_matrix, index=labels, columns=labels), md_flag, md_index=True)
        # 绘制决策树
        plt.figure()
        # 如果X.columns.tolist()有中文，就设置字体是宋体
        if hasattr(X, "columns"):
            for col in X.columns:
                if self.has_chinese(col):
                    plt.rcParams['font.sans-serif'] = ['SimSun']
                    break
        tree.plot_tree(self.tree, filled=True,
                       feature_names=X.columns.tolist(), class_names=[str(i) for i in self.tree.classes_])
        ToMd.pic_to_md(plt, md_flag, md_title="DecisionTreeClassifier")
        if draw_tree:
            plt.show()
        plt.close()

    def _cal_random_forest(self, X, y, pos_label, md_flag=False, **kwargs):
        """
        随机森林
        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        neg_label = [i for i in set(y) if i != pos_label]
        labels = [pos_label] + neg_label

        self.tree = RandomForestClassifier(**kwargs)
        self.tree.fit(X, y)
        print(CT("随机森林:").blue())
        print("特征重要性：", self.tree.feature_importances_)
        print("特征重要性降序排序是：")
        feature_importance = pd.DataFrame({"feature": X.columns, "importance": self.tree.feature_importances_})
        feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        print(feature_importance)
        self.feature_by_importance = feature_importance["feature"].values.tolist()
        self.ACC = self.tree.score(X, y)
        print("准确率：", self.ACC)
        self.y_pred = self.tree.predict(X)
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')  # F1分数
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")

        print("分类报告：")
        classify_report = classification_report(y, self.y_pred)
        print(classify_report)
        print(f"混淆矩阵 -- labels = {labels}：")
        conf_matrix = confusion_matrix(y, self.y_pred, labels=labels)
        print(conf_matrix)

        ToMd.text_to_md(f"RandomForestClassifier 训练集准确率: {self.ACC}", md_flag, md_color='blue')
        ToMd.df_to_md(feature_importance, md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(classification_report(y, self.y_pred, output_dict=True)).transpose(), md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(conf_matrix, index=labels, columns=labels), md_flag, md_index=True)


class GBDT(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.gbdt = None
        self.y_pred = None
        self.residuals = None
        self.feature_by_importance = None

    def cal_gbdtC(self, X_name, y_name, draw_gbdt=False, pos_label=1, **kwargs):
        """
        GBDT分类
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_gbdt: bool，是否绘制GBDT的拟合曲线。
        :param pos_label: int，正类别标签。
        :param kwargs: 其他参数。包括：
            loss: Any = "deviance",               # 损失函数，包括{"deviance", "exponential"}
            learning_rate: Any = 0.1,             # 学习率
            n_estimators: Any = 100,              # 树的数量
            subsample: Any = 1.0,                 # 子采样
            criterion: Any = "friedman_mse",      # 评估节点分裂的标准
            min_samples_split: Any = 2,           # 内部节点再划分所需最小样本数
            min_samples_leaf: Any = 1,            # 叶子节点最少样本数
            min_weight_fraction_leaf: Any = 0.0,  # 叶子节点的样本权重和的最小加权分数
            max_depth: Any = 3,                   # 树的最大深度
            min_impurity_decrease: Any = 0.0,     # 分裂节点的最小不纯度
            max_features: Any = None,             # 每次分裂的最大特征数
            max_leaf_nodes: Any = None,           # 最大叶子节点数
            random_state: Any = None,             # 随机种子
            verbose: Any = 0,                     # 详细输出
            warm_start: Any = False,              # 是否热启动
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_gbdtC(X, y, draw_gbdt, pos_label, **kwargs)

    def cal_gbdtR(self, X_name, y_name, draw_gbdt=False, **kwargs):
        """
        GBDT回归
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_gbdt: bool，是否绘制GBDT的拟合曲线。
        :param kwargs: 其他参数。包括：
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_gbdtR(X, y, draw_gbdt, **kwargs)

    def _cal_gbdtC(self, X, y, draw_gbdt, pos_label, md_flag=False, **kwargs):
        """
        GBDT分类
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_gbdt: bool，是否绘制GBDT的拟合曲线。
        :param kwargs: 其他参数。
        :return: None
        """
        neg_label = [i for i in set(y) if i != pos_label]
        labels = [pos_label] + neg_label

        self.gbdt = GradientBoostingClassifier(**kwargs)
        self.gbdt.fit(X, y)
        print(CT("GBDT-分类:").blue())

        print("特征重要性：", self.gbdt.feature_importances_)
        feature_importance = pd.DataFrame({"feature": X.columns, "importance": self.gbdt.feature_importances_})
        feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        print(feature_importance)
        self.feature_by_importance = feature_importance["feature"].values.tolist()
        print(self.feature_by_importance)
        self.ACC = self.gbdt.score(X, y)
        print("准确率：", self.ACC)
        self.y_pred = self.gbdt.predict(X)
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')  # F1分数
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")

        print("分类报告：")
        classify_report = classification_report(y, self.y_pred)
        print(classify_report)
        print(f"混淆矩阵 -- labels = {labels}：")
        conf_matrix = confusion_matrix(y, self.y_pred, labels=labels)
        print(conf_matrix)

        ToMd.text_to_md(f"GradientBoostingClassifier 训练集准确率: {self.ACC}", md_flag, md_color='blue')
        ToMd.df_to_md(feature_importance, md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(classification_report(y, self.y_pred, output_dict=True)).transpose(), md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(conf_matrix, index=labels, columns=labels), md_flag, md_index=True)
        if draw_gbdt:
            # 绘制GBDT的拟合曲线
            if X.shape[1] > 1:
                fw(self._cal_gbdtC, "X的维度大于1，无法绘制GBDT-Classification")
                return None
            plt.figure()
            plt.scatter(X, y, c='b')
            plt.plot(X, self.gbdt.predict(X), c='r', label='GBDT')
            plt.title("GBDT Classification")
            if hasattr(X, "columns"):
                if self.has_chinese(X.columns[0]):
                    plt.xlabel(X.columns[0], fontproperties='SimSun')
                else:
                    plt.xlabel(X.columns[0])
            if hasattr(y, "name"):
                if self.has_chinese(y.name):
                    plt.ylabel(y.name, fontproperties='SimSun')
                else:
                    plt.ylabel(y.name)
            plt.show()
            plt.close()

    def _cal_gbdtR(self, X, y, draw_gbdt, md_flag=False, **kwargs):
        """
        GBDT回归
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_gbdt: bool，是否绘制GBDT的拟合曲线。
        :param kwargs: 其他参数。
        :return: None
        """
        self.gbdt = GradientBoostingRegressor(**kwargs)
        self.gbdt.fit(X, y)
        print(CT("GBDT-回归:").blue())
        print("特征重要性：", self.gbdt.feature_importances_)
        print("R²分数：", self.gbdt.score(X, y))
        self.y_pred = self.gbdt.predict(X)
        self.residuals = y - self.y_pred
        self.MAE = np.mean(np.abs(self.residuals))
        self.MSE = np.mean(self.residuals ** 2)
        self.RMSE = math.sqrt(self.MSE)

        ToMd.text_to_md(f"GradientBoostingRegressor R²分数: {self.gbdt.score(X, y)}", md_flag, md_color='blue')
        if draw_gbdt:
            # 绘制GBDT的拟合曲线
            if X.shape[1] > 1:
                fw(self._cal_gbdtC, "X的维度大于1，无法绘制GBDT-Classification")
                return None
            plt.scatter(X, y, c='b')
            plt.plot(X, self.y_pred, c='r', label='GBDT')
            plt.title("GBDT Regression")
            if hasattr(X, "columns"):
                if self.has_chinese(X.columns[0]):
                    plt.xlabel(X.columns[0], fontproperties='SimSun')
                else:
                    plt.xlabel(X.columns[0])
            if hasattr(y, "name"):
                if self.has_chinese(y.name):
                    plt.ylabel(y.name, fontproperties='SimSun')
                else:
                    plt.ylabel(y.name)
            plt.show()
            plt.close()


class KNN(CalData):
    def __init__(self, df):
        super().__init__(df)

    def cal_knnC(self, X_name, y_name, k):
        """
        knn分类
        :param X_name:
        :param y_name:
        :param k:
        :return:
        """
        X = self.df[X_name]
        y = pd.Series(self.df[y_name], name=y_name)
        self._cal_knnC(X, y, k)

    def cal_knnR(self, X_name, y_name, k):
        """
        knn回归
        :param X_name:
        :param y_name:
        :param k:
        :return:
        """
        X = self.df[X_name]
        y = pd.Series(self.df[y_name], name=y_name)
        self._cal_knnR(X, y, k)

    def _cal_knnC(self, X, y, k):
        """
        knn分类
        :param X:
        :param y:
        :param k:
        :return:
        """
        self.knnC = KNeighborsClassifier(n_neighbors=k)
        self.knnC.fit(X, y)

    def _cal_knnR(self, X, y, k):
        """
        knn分类
        :param X:
        :param y:
        :param k:
        :return:
        """
        self.knnR = KNeighborsRegressor(n_neighbors=k)
        self.knnR.fit(X, y)


class NaiveBayes(CalData):
    def __init__(self, df):
        super().__init__(df)

    def cal_naive_bayes(self, X_name, y_name, draw_nb=False, pos_label=1, **kwargs):
        """
        朴素贝叶斯，用于分类
        原理是基于贝叶斯定理和特征条件独立假设，即每个特征在给定类别下是独立的。
        公式：P(y|X) = P(y) * P(X|y) / P(X)
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param draw_nb: bool，是否绘制朴素贝叶斯的拟合曲线。
        :param pos_label: int，正类别标签。
        :param kwargs: 其他参数。包括：
            priors: Any = None,                   # 先验概率
            var_smoothing: Any = 1e-9             # 方差平滑
        """
        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_naive_bayes(X, y, draw_nb, pos_label, **kwargs)

    def _cal_naive_bayes(self, X, y, draw_nb, pos_label, md_flag=False, **kwargs):
        """
        朴素贝叶斯
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param draw_nb: bool，是否绘制朴素贝叶斯的拟合曲线。
        :param kwargs: 其他参数。
        :return: None
        """
        neg_label = [i for i in set(y) if i != pos_label]
        labels = [pos_label] + neg_label

        self.nb = GaussianNB(**kwargs)
        self.nb.fit(X, y)
        print(CT("朴素贝叶斯:").blue())
        self.ACC = self.nb.score(X, y)
        print("准确率：", self.ACC)
        self.y_pred = self.nb.predict(X)
        self.F1_weight = f1_score(y, self.y_pred, average='weighted')
        self.F1_unweighted = f1_score(y, self.y_pred, average="macro")

        print("分类报告：")
        classify_report = classification_report(y, self.y_pred)
        print(classify_report)
        print(f"混淆矩阵 -- labels = {labels}：")
        conf_matrix = confusion_matrix(y, self.y_pred, labels=labels)
        print(conf_matrix)

        ToMd.text_to_md(f"GaussianNB 训练集准确率: {self.ACC}", md_flag, md_color='blue')
        ToMd.df_to_md(pd.DataFrame(classification_report(y, self.y_pred, output_dict=True)).transpose(), md_flag, md_index=True)
        ToMd.df_to_md(pd.DataFrame(conf_matrix, index=labels, columns=labels), md_flag, md_index=True)


class CrossValidation(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.cv = None
        self.avg_score = None

    def cal_cross_validation(self, model, X_name, y_name, cv=5, scoring="accuracy", **kwargs):
        """
        交叉验证
        :param model: sklearn的模型。
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param cv: int，交叉验证的折数，默认是5。
        :param scoring: str，评分标准，有{"accuracy", "precision", "recall", "f1", "roc_auc"}等。
        :param kwargs: 其他参数。
        :return: None
        """

        X = self.df[X_name]
        y = self.df[y_name].values
        self._cal_cross_validation(model, X, y, cv, scoring, **kwargs)

    def _cal_cross_validation(self, model, X, y, cv, scoring, **kwargs):
        """
        交叉验证
        :param model: sklearn的模型。
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param cv: int，交叉验证的折数，默认是5。
        :param scoring: str，评分标准，有{"accuracy", "precision", "recall", "f1", "roc_auc"}等。
        :param kwargs: 其他参数。
        :return: None
        """
        self.cv = cross_val_score(model, X, y, cv=cv, scoring=scoring, **kwargs)
        print(CT("交叉验证:").blue())
        print("交叉验证结果：", self.cv)
        self.avg_score = np.mean(self.cv)
        print(f"平均{scoring}：", self.avg_score)
        print(f"{scoring}的标准差：", np.std(self.cv))


# 降维
class DimReduction(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.df_DR = None  # 降维后的数据
        self.pca = None
        self.lda = None
        self.tsne = None

    def cal_pca(self, X_name, y_name, n_components=2, draw_DR=False, ax=None, **kwargs):
        """
        主成分分析（PCA）
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param n_components: int，降维后的维度，默认是2。
        :param draw_DR: bool，是否绘制降维后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_pca(X, y, n_components, draw_DR, ax, **kwargs)

    def cal_lda(self, X_name, y_name, n_components=2, draw_DR=False, ax=None, **kwargs):
        """
        线性判别分析（LDA）
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param n_components: int，降维后的维度，默认是2。
        :param draw_DR: bool，是否绘制降维后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_lda(X, y, n_components, draw_DR, ax, **kwargs)

    def cal_tsne(self, X_name, y_name, n_components=2, draw_DR=False, ax=None, **kwargs):
        """
        t-SNE
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param n_components: int，降维后的维度，默认是2。
        :param draw_DR: bool，是否绘制降维后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_tsne(X, y, n_components, draw_DR, ax, **kwargs)

    def _cal_pca(self, X, y, n_components, draw_DR, ax, **kwargs):
        """
        主成分分析（PCA）
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param n_components: int，降维后的维度，默认是2。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.pca = PCA(n_components=n_components, **kwargs)
        X_DR = self.pca.fit_transform(X)
        self.df_DR = pd.concat([pd.DataFrame(X_DR), pd.Series(y, name=y.name)], axis=1)

        if draw_DR:
            draw_scatter(X_DR[:, 0], X_DR[:, 1], show_plt=show_plt, ax=ax,
                         if_colorful=True, c=y, title="PCA", x_label="PC1", y_label="PC2", cmap="viridis")

    def _cal_lda(self, X, y, n_components, draw_DR, ax, **kwargs):
        """
        线性判别分析（LDA）
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param n_components: int，降维后的维度，默认是2。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.lda = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
        X_DR = self.lda.fit_transform(X, y)
        self.df_DR = pd.concat([pd.DataFrame(X_DR), pd.Series(y, name=y.name)], axis=1)

        if draw_DR:
            draw_scatter(X_DR[:, 0], X_DR[:, 1], show_plt=show_plt, ax=ax,
                         if_colorful=True, c=y, title="LDA", x_label="LD1", y_label="LD2", cmap="viridis")

    def _cal_tsne(self, X, y, n_components, draw_DR, ax, **kwargs):
        """
        t-SNE
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param n_components: int，降维后的维度，默认是2。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.tsne = TSNE(n_components=n_components, **kwargs)
        X_DR = self.tsne.fit_transform(X)
        self.df_DR = pd.concat([pd.DataFrame(X_DR), pd.Series(y, name=y.name)], axis=1)

        if draw_DR:
            draw_scatter(X_DR[:, 0], X_DR[:, 1], show_plt=show_plt, ax=ax,
                         if_colorful=True, c=y, title="t-SNE", x_label="t-SNE1", y_label="t-SNE2", cmap="viridis")


# 聚类
class Cluster(CalData):
    def __init__(self, df):
        super().__init__(df)
        self.df_cluster = None

        self.kmeans = None
        self.agg = None
        self.dbscan = None

        self.silhouette_score = None
        self.adjusted_rand_score = None
        self.mutual_info_score = None

    def cal_kmeans(self, X_name, y_name, n_clusters=2, draw_cluster=False, ax=None, **kwargs):
        """
        KMeans
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param n_clusters: int，簇的数量，默认是2。
        :param draw_cluster: bool，是否绘制聚类后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_kmeans(X, y, n_clusters, draw_cluster, ax, **kwargs)

    def cal_agg(self, X_name, y_name, n_clusters=2, draw_cluster=False, ax=None, **kwargs):
        """
        层次聚类(Agglomerative Clustering)
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param n_clusters: int，簇的数量，默认是2。
        :param draw_cluster: bool，是否绘制聚类后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_agg(X, y, n_clusters, draw_cluster, ax, **kwargs)

    def cal_dbscan(self, X_name, y_name, eps=0.5, min_samples=5, draw_cluster=False, ax=None, **kwargs):
        """
        DBSCAN
        :param X_name: str或list，输入特征的列名。
        :param y_name: str，输出标签的列名。
        :param eps: float，两个样本被看作邻居节点的最大距离。
        :param min_samples: int，核心点的最小样本数。
        :param draw_cluster: bool，是否绘制聚类后的散点图。
        :param ax: plt.axis，绘制图的坐标轴。
        :param kwargs: 其他参数。
        :return: None
        """
        X = self.df[X_name]
        y = self.df[y_name]
        self._cal_dbscan(X, y, eps, min_samples, draw_cluster, ax, **kwargs)

    def _cal_kmeans(self, X, y, n_clusters, draw_cluster, ax, **kwargs):
        """
        KMeans
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param n_clusters: int，簇的数量，默认是2。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        y_pred = self.kmeans.fit_predict(X)
        self.df_cluster = pd.concat([pd.DataFrame(X), pd.Series(y_pred, name="cluster")], axis=1)

        self._calculate_metrics(X, y, y_pred)
        if draw_cluster:
            self._draw_cluster(X, y, y_pred, ax, "KMeans", show_plt=show_plt)

    def _cal_agg(self, X, y, n_clusters, draw_cluster, ax, **kwargs):
        """
        层次聚类(Agglomerative Clustering)
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param n_clusters: int，簇的数量，默认是2。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.agg = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        y_pred = self.agg.fit_predict(X)
        self.df_cluster = pd.concat([pd.DataFrame(X), pd.Series(y_pred, name="cluster")], axis=1)

        self._calculate_metrics(X, y, y_pred)
        if draw_cluster:
            self._draw_cluster(X, y, y_pred, ax, "Agglomerative Clustering", show_plt=show_plt)

    def _cal_dbscan(self, X, y, eps, min_samples, draw_cluster, ax, **kwargs):
        """
        DBSCAN
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param eps: float，两个样本被看作邻居节点的最大距离。
        :param min_samples: int，核心点的最小样本数。
        :param kwargs: 其他参数。
        :return: None
        """
        show_plt = kwargs.pop("show_plt", True)
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        y_pred = self.dbscan.fit_predict(X)
        self.df_cluster = pd.concat([pd.DataFrame(X), pd.Series(y_pred, name="cluster")], axis=1)

        self._calculate_metrics(X, y, y_pred)
        if draw_cluster:
            self._draw_cluster(X, y, y_pred, ax, "DBSCAN", show_plt=show_plt)

    def _calculate_metrics(self, X, y, y_pred):
        """
        计算评价指标
        s(i)=\frac{b(i)-a(i)}{\max \{a(i), b(i)\}}
        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
        I(U ; V)=\sum_{i=1}^{|U|} \sum_{j=1}^{|V|} p_{i j} \log \left(\frac{p_{i j}}{p_{i} p_{j}}\right)
        :param X: np.ndarray，输入特征。
        :param y: np.ndarray，输出标签。
        :param y_pred: np.ndarray，聚类预测标签。
        :return: None
        """
        if len(np.unique(y_pred)) > 1:
            self.silhouette_score = silhouette_score(X, y_pred)
        else:
            self.silhouette_score = "Error"
            print("聚类结果只有一个簇或所有点都被标记为噪声，无法计算 silhouette score")
        self.adjusted_rand_score = adjusted_rand_score(y, y_pred)
        self.mutual_info_score = normalized_mutual_info_score(y, y_pred)
        print(f"Silhouette Score: {self.silhouette_score}")
        print(f"Adjusted Rand Index: {self.adjusted_rand_score}")
        print(f"Mutual Information Score: {self.mutual_info_score}")

    def _draw_cluster(self, X, y, y_pred, ax, title, show_plt=True):
        """
        绘制聚类结果的等高线填充图
        :param X: np.ndarray或者pd.dataframe，输入特征。
        :param y: np.ndarray，实际的y
        :param y_pred: np.ndarray，聚类预测标签。（事实上该参数并不需要，
        :param ax: plt.axis，绘制图的坐标轴。
        :param title: str，图的标题。
        :return: None
        """
        # 检查输入特征的维度
        if X.shape[1] > 2:
            raise ValueError("输入特征的维度大于2，无法绘制聚类结果的等高线图。")

        # 将X变为numpy.array
        X = np.array(X)
        # 检查ax
        if ax is None:
            ax = plt.gca()

        # 设置边界范围的扩展比例
        padding = 0.1
        print("X.shape:", X.shape)
        print("y_pred.shape:", y_pred.shape)
        X0_min, X0_max = X[:, 0].min(), X[:, 0].max()
        X1_min, X1_max = X[:, 1].min(), X[:, 1].max()
        x_min, x_max = X0_min - padding * (X0_max - X0_min), X0_max + padding * (X0_max - X0_min)
        y_min, y_max = X1_min - padding * (X1_max - X1_min), X1_max + padding * (X1_max - X1_min)

        # 创建网格
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # 根据聚类方法预测网格点的聚类标签
        if hasattr(self, 'kmeans') and self.kmeans is not None:
            Z_contourf = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        elif hasattr(self, 'agg') and self.agg is not None:
            Z_contourf = self.agg.fit_predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        elif hasattr(self, 'dbscan') and self.dbscan is not None:
            Z_contourf = self.dbscan.fit_predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        else:
            raise ValueError("未找到有效的聚类方法。")

        # 绘制等高线填充图
        cmap_light = ListedColormap(['#FF1493', '#66ccff'])
        ax.contourf(xx, yy, Z_contourf, cmap=cmap_light, alpha=0.5)

        # 绘制决策边界
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        if hasattr(self, 'kmeans') and self.kmeans is not None:
            Z_contour = self.kmeans.predict(xy).reshape(xx.shape)
        elif hasattr(self, 'agg') and self.agg is not None:
            Z_contour = self.agg.fit_predict(xy).reshape(xx.shape)
        elif hasattr(self, 'dbscan') and self.dbscan is not None:
            Z_contour = self.dbscan.fit_predict(xy).reshape(xx.shape)
        else:
            raise ValueError("未找到有效的聚类方法。")

        ax.contour(xx, yy, Z_contour, colors='k', levels=np.unique(Z_contour), alpha=0.8, linestyles=['--', '-', '--'])

        # 绘制散点图
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')

        # 添加图例
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)

        # 设置图的标题和网格线
        ax.set_title(title)
        ax.grid(True)
        if show_plt:
            plt.show()
        plt.close()

# 以下是一些废弃的函数，请不要使用
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



