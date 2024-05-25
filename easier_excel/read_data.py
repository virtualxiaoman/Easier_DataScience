import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cv2
from sklearn.experimental import enable_iterative_imputer  # 为了使用IterativeImputer，需要导入这个
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from easier_tools.Colorful_Console import func_warning as func_w
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


class desc_df:
    def __init__(self, df):
        """
        初始化
        :param df: 可以传入df或者df.copy()
        """
        self.df = df  # 传入的df数据
        self.shape = None  # df大小
        self.columns = None  # df属性
        self.dtypes = None  # 数据类型
        self.numeric_stats = None  # 数据描述
        self.missing_info = None  # 缺失值信息

        self.numeric_df = self.df.select_dtypes(include=['number'])  # 数值型数据
        self.non_numeric_df = self.df.select_dtypes(exclude=['number'])  # 非数值型数据

    def get_df(self):
        """
        如果在初始化这个类desc_df的时候传入的是df.copy()，那就不会对原来的df进行更改。
        此时如果需要获得更改后的df，请使用这个函数吧！
        [Tips]:
            该函数不如直接使用.df，这里只是为了免得以前的代码寄了，所以保留了这个函数
        """
        return self.df

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
        filled_numeric_df = pd.DataFrame(imputer.fit_transform(self.numeric_df), columns=self.numeric_df.columns)

        # 处理非数值型数据
        self.non_numeric_df = self.df.select_dtypes(exclude=['number'])
        filled_non_numeric_df = self.non_numeric_df.fillna(float('nan'))  # NaN填充

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
            Q1 = self.numeric_df.quantile(0.25)  # 返回的是一个Series
            Q3 = self.numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'Z-score':
            threshold = 3
            Z_scores = np.abs((self.numeric_df - self.numeric_df.mean()) / self.numeric_df.std())  # 返回的是一个DataFrame
            # 返回的是一个布尔值的DataFrame，True表示异常值
            outliers = Z_scores > threshold
            # 返回的是一个Series，~是取反，mask是将不满足条件的值替换为NaN，然后取最小值，即最小的正常值
            lower_bound = self.numeric_df.mask(~outliers).min()
            upper_bound = self.numeric_df.mask(~outliers).max()  # 返回的是一个Series，即最大的正常值
        else:
            pass

        # 异常值的条件是小于下界或者大于上界。outlier_index是一个DataFrame，True表示异常值
        outlier_index = (self.numeric_df < lower_bound) | (self.numeric_df > upper_bound)

        if show_info:
            print(CT("异常值的数量:\n").blue(), self.numeric_df[outlier_index].count())
            print(CT("异常值的比例:\n").blue(), self.numeric_df[outlier_index].count() / self.numeric_df.shape[0])
            # 输出outlier_index为True的坐标
            print(CT("异常值的坐标:\n").blue(), np.where(outlier_index))

        if process_type == 'delete':
            self.numeric_df = self.numeric_df[~outlier_index]
        elif process_type == 'fill':
            self.numeric_df[outlier_index] = np.nan
        elif process_type == 'ignore':
            pass
        else:
            func_w(self.process_outlier,
                   warning_text=f"不支持的process_type格式'{process_type}'，这里默认不做修改，也就是ignore",
                   modify_tip="请检查process_type是否正确")

        # 处理非数值型数据
        self.non_numeric_df = self.df.select_dtypes(exclude=['number'])
        # 合并
        self.df = pd.concat([self.numeric_df, self.non_numeric_df], axis=1)

        # 下面这一行是为了更新self.missing_info，便于在外部调用fill_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)


# 一些可能的读入方法以作为记录
# df_main['index'] = range(1, df_main.shape[0] + 1)  # 但不能将index放在第一行，下面一行代码可以：
# df_main.insert(0, 'index', range(1, df_main.shape[0] + 1))
# print(df_main.iloc[5])  # 获取第6行


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

