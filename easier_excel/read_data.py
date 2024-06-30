import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy import stats
from scipy.interpolate import interp1d, interp2d, lagrange, RectBivariateSpline, griddata, Rbf  # interp2d已被弃用
from sklearn.experimental import enable_iterative_imputer  # 为了使用IterativeImputer，需要导入这个
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer

from easier_excel.utils import DFUtils
from easier_tools.Colorful_Console import func_warning as func_w
from easier_tools.Colorful_Console import func_error as func_e
from easier_tools.Colorful_Console import ColoredText as CT

def set_pd_option(max_show=True, float_type=True, decimal_places=2, reset_all=False, reset_display=False):
    """
    该设置为全局设置。
    如果需要部分代码块设置，则需要类似于这样的代码：
        with pd.option_context("display.max_rows", None, "display.max_columns", None, 'display.expand_frame_repr', False):
            print(pd.get_option("display.max_rows"))
            print(pd.get_option("display.max_columns"))
            ...  # 只在with内的代码设置
        # with外的代码还是原来的设置
    :param max_show:
    :param float_type:
    :param decimal_places:
    :param reset_all:
    :param reset_display:
    """
    if max_show:
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.expand_frame_repr', False)  # 不允许水平拓展
        pd.set_option('display.max_rows', None)  # 显示所有行
        pd.set_option('display.width', None)  # 不换行
    else:
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")
        pd.reset_option("display.max_rows")
        pd.reset_option("display.width")
    if float_type:
        str_decimal_places = '%.' + str(decimal_places) + 'f'
        pd.set_option('display.float_format', lambda x: str_decimal_places % x)  # 根据参数设置浮点数输出格式
        # pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 设置浮点数输出格式
        # pd.options.display.float_format = '{:.2f}'.format  # 也可以这样
    else:
        pd.reset_option('display.float_format')
    if reset_all:
        pd.reset_option('all')
    if reset_display:
        pd.reset_option("^display")  # ^表示以某个字符开始，在这里表示以display开始全部重置

def read_df(path):
    """
    读入数据，支持.csv和.xlsx
    :param path:路径。依据路径结尾的类型来判断是csv还是xlsx
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    elif path.endswith('.sav'):
        df = pd.read_spss(path)
    else:
        raise ValueError("不支持的数据类型，目前只支持.csv, .xls, .xlsx")
    return df


class desc_df(DFUtils):
    def __init__(self, df):
        """
        初始化
        :param df: 可以传入df或者df.copy()
        """
        super().__init__(df)
        self.numeric_stats = None  # 数据描述
        self.missing_info = None  # 缺失值信息

    def show_df(self, head_n=0, tail_n=0, show_shape=True, show_columns=True, show_dtypes=True, dtypes_T=False):
        """
        查看数据，请传入DataFrame数据类型的数据。
        一些可能用到的，用于查阅：
            printf(df.info())  # 查看数据的基本信息
            print(df['离散型数据的属性名称'].value_counts())
            print(df['连续性数据的属性名称'].describe())
        :param head_n: 查看前head_n行数据
        :param tail_n: 查看末tail_n行数据
        :param show_shape: 查看数据大小
        :param show_columns: 是否查看表首行(属性名称)
        :param show_dtypes: 是否查看data数据类型
        :param dtypes_T: 是否转置dtypes(默认不转置)(dtypes的数据类型是Series)
        """
        self.shape = self.df.shape
        self.columns = self.df.columns
        self.dtypes = self.df.dtypes
        if head_n:
            print(CT("前{}行数据:".format(head_n)).blue())
            print(self.df.head(head_n))  # 查看前head_n行数据
        if tail_n:
            print(CT("末{}行数据:".format(tail_n)).blue())
            print(self.df.tail(tail_n))  # 查看末tail_n行数据
        if show_shape:
            print(CT("Shape: ").blue() + str(self.shape))  # 数据大小
        if show_columns:
            print(CT("属性名称:").blue())
            print(self.columns)  # 查看表首行(属性名称)
        if show_dtypes:
            print(CT("数据类型:").blue())
            if dtypes_T:
                print(self.dtypes.to_frame().T)
            else:
                print(self.dtypes)

    def describe_df(self, show_stats=True, stats_T=True, stats_detailed=False, show_nan=True, show_nan_heatmap=False):
        """
        输出数据的基本统计信息，以及检测缺失值。
        [使用方法]:
            desc = read_data.desc_df(df)
            desc.describe_df(show_stats=True, stats_T=True, stats_detailed=False, show_nan=True, show_nan_heatmap=False)
        [Tips]:
            1.如果要把缺失值比例还原成数值，比如3.7%变成0.037，可以使用下面这行代码：
                self.missing_info['缺失值比例'] = self.missing_info['缺失值比例'].apply(lambda x: float(x[:-1]) / 100)
            2.有时为了更新missing_info，需要使用下面这行代码：
                self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)
        :param show_stats: 描述性统计信息。显示DataFrame中数值列的描述性统计信息，如均值、标准差、最大值、最小值等
        :param stats_T: 是否转置输出stats。转置时，每一列对应一个属性(如count)。
        :param stats_detailed: 是否输出stats的详细信息。False时，只输出 'count', 'mean', 'min', 'max' 这四项。
        :param show_nan: 缺失值检测。检查DataFrame中是否存在缺失值，并显示每列缺失值的数量或比例。
        :param show_nan_heatmap: 是否画缺失值热力图
        """
        if stats_T:
            self.numeric_stats = self.df.describe().T
        else:
            self.numeric_stats = self.df.describe()

        # 如果不需要详细的统计信息，则选择特定的列
        if not stats_detailed:
            if stats_T:
                self.numeric_stats = self.numeric_stats[['count', 'mean', 'min', 'max']]
            else:
                self.numeric_stats = self.numeric_stats.loc[['count', 'mean', 'min', 'max']]

        if show_stats:
            print(CT("描述性统计信息:\n").blue(), self.numeric_stats)

        missing_values = self.df.isnull().sum()
        # total_rows = self.df.shape[0]  # 行数，相当于len(self.df)
        # missing_percentage = missing_values / total_rows
        missing_percentage = self.df.isnull().mean()
        self.missing_info = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失值比例': missing_percentage
        })
        self.missing_info['缺失值比例'] = self.missing_info['缺失值比例'].apply(lambda x: '{:.1%}'.format(x))

        if show_nan:
            print(CT("缺失值检测:\n").blue(), self.missing_info)
        if show_nan_heatmap:
            # 黑色是缺失值，白色是非缺失值
            sns.heatmap(self.df.isna(), cmap='gray_r', cbar_kws={"orientation": "vertical"}, vmin=0, vmax=1)
            plt.show()
            plt.close()
            # 还可以使用import missingno
            # missingno.matrix(self.df)

    def draw_heatmap(self, scale=False, xticklabels=None):
        """
        画热力图
        :param scale: 是否标准化
        :param xticklabels: x轴标签，None代表使用df的columns
        """
        if xticklabels is None:
            xticklabels = list(self.df.columns)
        if scale:
            df = self.df.copy()
            df = (df - df.mean()) / df.std()
            sns.heatmap(df, cmap='RdYlBu_r', xticklabels=xticklabels, cbar_kws={"orientation": "vertical"})
        else:
            sns.heatmap(self.df, cmap='RdYlBu_r', xticklabels=xticklabels, cbar_kws={"orientation": "vertical"})
        plt.show()
        plt.close()

    def delete_missing_values(self, axis=0, how='any', inplace=True):
        """
        删除缺失值
        [Waring]:
            该操作会直接在原数据上进行修改
        :param axis: 0表示删除行，1表示删除列
        :param how: any表示只要有缺失值就删除，all表示全部是缺失值才删除
        :param inplace: 是否在原数据上进行修改，不建议设置为False(设置为False时返回删除缺失值后的df)
        :return: 删除缺失值后的df(这在选用inplace=False时有用)
        """
        temp_df = self.df.dropna(axis=axis, how=how, inplace=inplace)
        # 下面这行是为了更新self.missing_info，便于在外部调用delete_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)
        return temp_df  # 返回删除缺失值后的df(这在选用inplace=False时有用)

    def fill_missing_values(self, fill_type='mean', **kwargs):
        """
        填充缺失值
        [Warning]:
            该操作会直接在原数据上进行修改，也就是修改self.df
        [使用方法]:
            desc = read_data.desc_df(df)
            desc.fill_missing_values(fill_type=114514)  # 实际填充的时候可别逸一时误一世了
        [Tips]:
            1.对于数值型数据，目前支持的填充类型有：
                SimpleImputer: 'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)'。
                KNNImputer: 'knn'。
                IterativeImputer: 'rf'。
              请注意这个不适合于时间序列的数据，时间序列的请使用插值法，比如本文件中的interpolate_data类。
            2.对于其他类型的数据，目前只支持的填充类型有：
                填充 'nan'，也就是输出时显示NaN（其实还是缺失值~~）
            3.如需删除缺失值请使用：delete_missing_values。
            4.文档：https://scikit-learn.org/stable/modules/impute.html
        :param fill_type: 填补类型，支持 'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)', 'knn', 'rf'
        :param kwargs: 一些填充类型的参数。如
            n_neighbors(fill_type='knn'时生效),
            random_state, max_iter(fill_type='rf'时生效)
        """
        n_neighbors = kwargs.get('n_neighbors', 5)
        random_state = kwargs.get('random_state', 42)
        max_iter = kwargs.get('max_iter', 20)

        # 处理数值型数据
        if isinstance(fill_type, (int, float)):
            imputer = SimpleImputer(strategy='constant', fill_value=fill_type)
        elif fill_type in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=fill_type)
        elif fill_type == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif fill_type == 'rf':
            imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state), max_iter=max_iter)
        else:
            func_w(self.fill_missing_values,
                   warning_text=f"不支持的fill_type格式'{fill_type}'，这里默认采用0填充，如有需要，请自行更改为正确的格式",
                   modify_tip="请检查fill_type是否正确")
            imputer = SimpleImputer(strategy='constant', fill_value=0)  # 默认采用0填充
        filled_numeric_df = pd.DataFrame(imputer.fit_transform(self.df_numeric), columns=self.df_numeric.columns)

        # 处理非数值型数据
        filled_non_numeric_df = self.df_non_numeric.fillna(float('nan'))  # NaN填充

        # 合并
        self.df = pd.concat([filled_numeric_df, filled_non_numeric_df], axis=1)

        # 下面这段代码以前写的，仅作为保存与参考
        # missing_cols = self.missing_info[self.missing_info['缺失值数量'] != 0].index  # 缺失值数量不为 0 的属性
        # # print(missing_cols)
        # for col in missing_cols:
        #     dtype = self.df[col].dtype
        #     if dtype == 'object':
        #         self.df[col].fillna('nan', inplace=True)  # 用 'nan' 填充缺失值
        #     else:  # 如果是数值类型
        #         mean_value = self.df[col].mean()
        #         self.df[col].fillna(mean_value, inplace=True)  # 用均值填充缺失值

        # 下面这一行是为了更新self.missing_info，便于在外部调用fill_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)

    def process_outlier(self, method='IQR', process_type='delete', show_info=False):
        """
        处理异常值
        [Warning]:
            该操作会直接在原数据上进行修改，也就是修改self.df
        [Tips]:
            1.IQR:
                四分位间距(interquartile range) IQR=Q3-Q1，在[Q1-1.5*IQR, Q3+1.5*IQR]之外的数据被认为是异常值。
                不过在Sklearn绘制箱型图的时候，左右边界是[Q1-1.5*IQR, Q3+1.5*IQR]里的最远的实际的数据点，而不一定是计算得到的值。
            2.Z-score:
                |X-mu|/sigma > 3 的数据被认为是异常值。
            3.文档：
                Visualize-ML/Book6_Ch03的ipynb
                https://scikit-learn.org/stable/modules/outlier_detection.html
        :param method: 处理异常值的方法，支持 'IQR', 'Z-score'
        :param process_type: 处理异常值的类型，支持 'delete', 'fill', 'ignore'
        :param show_info: 是否输出异常值的一些信息
        """
        # 处理数值型数据
        lower_bound, upper_bound = None, None
        if method not in ['IQR', 'Z-score']:
            func_w(self.process_outlier,
                   warning_text=f"不支持的method格式'{method}'，这里默认选择IQR",
                   modify_tip="请检查method是否正确")
            method = 'IQR'
        if method == 'IQR':
            Q1 = self.df_numeric.quantile(0.25)  # 返回的是一个Series
            Q3 = self.df_numeric.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'Z-score':
            threshold = 3
            Z_scores = np.abs((self.df_numeric - self.df_numeric.mean()) / self.df_numeric.std())  # 返回的是一个DataFrame
            # 返回的是一个布尔值的DataFrame，True表示异常值
            outliers = Z_scores > threshold
            # 返回的是一个Series，~是取反，mask是将不满足条件的值替换为NaN，然后取最小值，即最小的正常值
            lower_bound = self.df_numeric.mask(~outliers).min()
            upper_bound = self.df_numeric.mask(~outliers).max()  # 返回的是一个Series，即最大的正常值
        else:
            pass

        # 异常值的条件是小于下界或者大于上界。outlier_index是一个DataFrame，True表示异常值
        outlier_index = (self.df_numeric < lower_bound) | (self.df_numeric > upper_bound)

        # todo 这里可以变成self属性值保存，然后再输出
        if show_info:
            print(CT("异常值的数量:\n").blue(), self.df_numeric[outlier_index].count())
            print(CT("异常值的比例:\n").blue(), self.df_numeric[outlier_index].count() / self.df_numeric.shape[0])
            # 输出outlier_index为True的坐标
            print(CT("异常值的坐标:\n").blue(), np.where(outlier_index))

        if process_type == 'delete':
            self.df_numeric = self.df_numeric[~outlier_index]
        elif process_type == 'fill':
            self.df_numeric[outlier_index] = np.nan
        elif process_type == 'ignore':
            pass
        else:
            func_w(self.process_outlier,
                   warning_text=f"不支持的process_type格式'{process_type}'，这里默认不做修改，也就是ignore",
                   modify_tip="请检查process_type是否正确")

        # 处理非数值型数据(这里不做处理)
        # 合并
        self.df = pd.concat([self.df_numeric, self.df_non_numeric], axis=1)

        # 下面这一行是为了更新self.missing_info，便于在外部调用fill_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)

    def transform_df(self, minmax=(0, 1)):
        """
        数据转换，包括中心化(去均值)，标准化(Z分数)，归一化(minmax)
        得到:
            self.demeaned_df: 去均值
            self.zscore_df: Z-score标准化
            self.minmax_df: 映射到[minmax[0], minmax[1]]的区间上
        [Tips]:
            1.标准化standardization: 使得处理后的数据具有固定均值0和标准差1，可以使得不同特征之间的数值尺度相同，
            避免某些特征对模型的影响过大，从而提高模型的鲁棒性和稳定性。标准化不会限制数据的范围。
            2.归一化normalization: 将数据缩放到[0,1]或[-1,1]的区间上。，可以使得不同特征的权重相同，
            避免某些特征对模型的影响过大，从而提高模型的准确性和泛化能力。归一化可使所有特征具有相似的尺度。
        """
        self.demeaned_df = self.df_numeric - self.df_numeric.mean()
        self.zscore_df = (self.df_numeric - self.df_numeric.mean()) / self.df_numeric.std()
        self.minmax_df = (minmax[1] - minmax[0]) * (self.df_numeric - self.df_numeric.min()) / \
                         (self.df_numeric.max() - self.df_numeric.min()) + minmax[0]
        if (self.df_numeric < 0).any().any():
            func_w(self.transform_df,
                   warning_text="数据中有负数项，无法进行boxcox",
                   modify_tip="请检查数据是否有负数项")
        else:
            pt = PowerTransformer(method='box-cox')
            self.boxcox_df = pt.fit_transform(self.df_numeric)
        pt = PowerTransformer(method='yeo-johnson')
        self.yeojohnson_df = pt.fit_transform(self.df_numeric)


# 一些可能的读入方法以作为记录
# df_main['index'] = range(1, df_main.shape[0] + 1)  # 但不能将index放在第一行，下面一行代码可以：
# df_main.insert(0, 'index', range(1, df_main.shape[0] + 1))
# print(df_main.iloc[5])  # 获取第6行


class interpolate_data:
    def __init__(self, x, y, z=None):
        """
        初始化。分为 datadims=1 或 datadims=2 两种情况。
        :param x: 横坐标
        :param y: 纵坐标
        :param z: None或者高度
        """
        self.x = x
        self.y = y
        self.z = z
        self.datadims = None  # 已知数据的维度
        self.x_predict = None
        self.y_predict = None
        self.z_predict = None
        self.f_predict = None  # 插值函数
        self.init_params()

    def interpolate(self, method='cubic', show_plt=True, plt_2d=True):
        """
        插值
        [使用方法-整体预测]:
        1.一维
            x_train = np.linspace(0, 6, num=7, endpoint=True)
            y_train = np.sin(x_train) + x_train/6  # 相当于加上一点点噪声，免得太规则
            methods = ['previous', 'next', 'nearest', 'linear', 'cubic']
            for kind in methods:
                interpolate_data(x_train, y_train).interpolate(method=kind)
        2.二维
          二维的时候，一般而言x,y形成的是平面网格，z是高度。但有时候不是网格也可以，比如下面的例子。
          2.0 下面两个例子都需要的公用函数与变量：
            def y_func(x1, x2):
                v = (2 * x1 + x2) * np.exp(-2 * (x1 ** 2 + x2 ** 2))
                return v
            methods = ['linear', 'cubic']
          2.1.网格
            x1_data = np.linspace(-1, 1, 5)
            x2_data = np.linspace(-1, 1, 5)
            xx1_data, xx2_data = np.meshgrid(x1_data, x2_data)
            yy_data = y_func(xx1_data, xx2_data)  # (5, 5)
            for kind in methods:
                interpolate_data(xx1_data, xx2_data, yy_data).interpolate(method=kind, show_plt=True, plt_2d=False)
          2.2 非网格
            x1_data = np.linspace(-1, 1, 5)
            x2_data = np.linspace(-1, 1, 5)
            y_data = y_func(x1_data, x2_data)  # (5,)
            for kind in methods:
                interpolate_data(x1_data, x2_data, y_data).interpolate(method=kind, show_plt=True, plt_2d=False)
          2.3 二者在哪里不同
            2.1中的数据是网格数据，是一个二维数组，而2.2中的数据是非网格数据，是一个一维数组。
            2.1中的数据像是张起来了一个平面，而2.2中的数据是一条曲线(这里是平面的对角线)。
            一般而言，2.1中的数据像是一个表格，里面存放的是z(深度/高度等)，比如国赛2023的B题。
            而2.2中的数据是每个确定的坐标对应的z值。
        [使用方法-具体值的预测]:
          对于创建的类的实例，可以使用其中的f_predict来预测具体的值，比如：
            interp = interpolate_data(xx1_data, xx2_data, yy_data)
            interp.interpolate(method=kind, show_plt=False, plt_2d=True)
            y_test = interp.f_predict(0.0032, 0.0004)  # 请注意f_predict的参数数量应该与你的预测维度相同
          [Warning]:请注意interp1d不允许预测超出已知数据范围的值，如果需要，可以使用拉格朗日插值，也就是method='lagrange'
        [todo]:
             1.将draw_data里的函数参数改为**kwargs形式，并修改这里的代码
             2.对于二维差值，还可以考虑使用griddata函数
        [Tips]:
            1.维度datadims是根据传入的x,y,z自动确定的。不传入z时维度是1，传入z时维度是2。
            2.插值和拟合有一个相同之处，它们都是根据已知数据点，构造函数，从而推断得到更多数据点。
                插值一般得到分段函数，分段函数通过所有给定的数据点。
                拟合得到的函数一般只有一个解析式，这个函数尽可能靠近样本数据点。
            3.method在一维时有：
                'previous' 是取前一个值，'next' 是取后一个值，'nearest' 是取最近的值。
                'linear' 是线性插值，'cubic' 是三次插值，'lagrange' 是拉格朗日插值。
              在二维时有：
                'multiquadric' 是多孔径插值，'inverse' 是反距离插值，'gaussian' 是高斯插值。
                'linear' 是线性插值，'cubic' 是三次插值，'quintic' 是五次插值，'thin_plate' 是薄板样条插值。
        :param method: 插值方法，有 'previous', 'next', 'nearest', 'linear', 'cubic', 'lagrange'
        :param show_plt: 是否绘制插值后的图像
        :param plt_2d: 是否绘制2D图像。只在3d预测时有效。默认为True表示绘制2D的colorbar图像，False表示绘制3D的plot_surface图像
        """
        if self.datadims == 1:
            if method in ['previous', 'next', 'nearest', 'linear', 'cubic']:
                self.f_predict = interp1d(self.x, self.y, kind=method)  # interp1d是一维插值函数
                self.y_predict = self.f_predict(self.x_predict)
            elif method == 'lagrange':
                self.f_predict = lagrange(self.x, self.y)  # 拉格朗日插值
                self.y_predict = self.f_predict(self.x_predict)
            else:
                func_e(self.interpolate,
                       error_text=f"不支持的method格式'{method}'",
                       modify_tip="请检查method是否正确")
            if show_plt:
                fig, axs = plt.subplots()
                plt.plot(self.x, self.y, 'or')
                plt.plot(self.x_predict, self.y_predict, linewidth=1.5)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(method)
                plt.show()
                plt.close()
        elif self.datadims == 2:
            if method in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']:
                self.f_predict = Rbf(self.x, self.y, self.z, function=method)  # 二维插值函数
                self.z_predict = self.f_predict(self.grid_x, self.grid_y)
            else:
                func_e(self.interpolate,
                       error_text=f"不支持的method格式'{method}'",
                       modify_tip="请检查method是否正确")
            if show_plt:
                if plt_2d:
                    fig, axs = plt.subplots()
                    plt.imshow(self.z_predict, extent=(self.x.min(), self.x.max(), self.y.min(), self.y.max()),
                               origin='lower', cmap='RdYlBu_r')
                    plt.colorbar()
                    plt.show()
                    plt.close()
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(self.x, self.y, self.z, marker='x', c='k')
                    ax.plot_surface(self.grid_x, self.grid_y, self.z_predict, cmap='RdYlBu_r')
                    plt.show()
                    plt.close()
        else:
            pass  # 不会进入该分支

    def init_params(self, x_predict_nums=500, y_predict_nums=500):
        """
        初始化一些参数
        """
        self.x_predict = np.linspace(self.x.min(), self.x.max(), x_predict_nums)  # 生成2d预测的x坐标

        self.grid_x, self.grid_y = np.mgrid[self.x.min():self.x.max():100j, self.y.min():self.y.max():100j]  # 生成3d预测所需的网格

        if self.z is None:
            self.datadims = 1
        else:
            self.datadims = 2
            # self.y_predict = np.linspace(self.y.min(), self.y.max(), y_predict_nums)


class time_data:
    def __init__(self):
        """
        参考Book6_Ch06，Book6_Ch07，Book6_Ch08
        时间序列的数据没怎么学，这里给出一个简单的时间序列数据的处理方法。
        import pandas as pd
        import matplotlib.pyplot as plt
        import pandas_datareader
        import calendar
        import seaborn as sns
        import easier_excel.read_data as read_data
        # [253 rows x 1 columns]，属性是UNRATENSA
        df = pandas_datareader.data.DataReader(['UNRATENSA'], data_source='fred', start='08-01-2000', end='08-01-2021')
        df['UNRATENSA'].plot()
        plt.show()
        df['year'] = pd.DatetimeIndex(df.index).year  # 获取年份，DatetimeIndex输入格式是时间序列，输出是年份
        df['month'] = pd.DatetimeIndex(df.index).month  # 获取月份
        df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])  # 将月份转换为英文缩写
        sns.lineplot(data=df, x="year", y="UNRATENSA", hue="month")
        plt.show()
        sns.lineplot(data=df, x="month", y="UNRATENSA", hue="year")
        plt.show()
        desc = read_data.desc_df(df)
        desc.describe_df(stats_detailed=False)
        print(desc.missing_info)
        import statsmodels.api as sm
        # 通过seasonal_decompose函数进行分解，得到res.resid残差、res.trend趋势、res.seasonal季节性、res.observed原始数据
        res = sm.tsa.seasonal_decompose(df['UNRATENSA'])
        resplot = res.plot()
        plt.show()


        import plotly.express as px
        import numpy as np
        import pandas as pd

        df = px.data.gapminder()
        df.rename(columns={"country": "country_or_territory"}, inplace=True)
        print(df.head(5))
        # 按年份和大洲分组，再对pop列求和
        df_pop_continent_over_t = df.groupby(['year', 'continent'], as_index=False).agg({'pop': 'sum'})
        print(df_pop_continent_over_t.head(5))
        fig = px.bar(df_pop_continent_over_t,
                     x='year', y='pop',
                     width=600, height=380,
                     color='continent',
                     labels={"year": "Year",
                             "pop": "Population"})
        fig.show()

        fig = px.line(df_pop_continent_over_t,
                      x='year', y='pop',
                      width=600, height=380,
                      color='continent',
                      labels={"year": "Year",
                              "pop": "Population"})
        fig.show()

        """
        pass


def read_image(img_path, gray_pic=False, show_details=False):
    """
    读取图片
    [使用示例]：
        path = '../output/arona.jpg'
        img = read_image(path, gray_pic=True, show_details=True)  # 读取为灰度图
    :param img_path: 图像路径
    :param gray_pic: 是否读取灰度图像
    :param show_details: 是否输出图片的shape以及显示图片
    :return: 图像数组，类型为np.ndarray。大小是(H, W, 3)或(H, W)
    """
    if gray_pic:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img_gbr = cv2.imread(img_path)
        img = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2RGB)
    if show_details:
        print(img.shape)
        if gray_pic:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()
        plt.close()
    return img


if __name__ == "__main__":
    pass


