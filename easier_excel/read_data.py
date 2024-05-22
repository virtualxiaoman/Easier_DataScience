import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn.impute import SimpleImputer

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
    # 读取.sav格式
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

    def get_df(self):
        """
        如果在初始化这个类desc_df的时候传入的是df.copy()，那就不会对原来的df进行更改。
        此时如果需要获得更改后的df，请使用这个函数吧！
        """
        return self.df

    def show_df(self, head_n=5, tail_n=3, show_shape=True, show_columns=True, show_dtypes=True, dtypes_T=False):
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
            print(self.df.head(head_n))  # 查看前head_n行数据
        if tail_n:
            print(self.df.tail(tail_n))  # 查看末tail_n行数据
        if show_shape:
            print("Shape:", self.shape)  # 数据大小
        if show_columns:
            print("属性名称:\n", self.columns)  # 查看表首行(属性名称)
        if show_dtypes:
            if dtypes_T:
                print("数据类型:\n", self.dtypes.to_frame().T)
            else:
                print("数据类型:\n", self.dtypes)

    def describe_df(self, show_stats=True, stats_T=True, stats_detailed=False, show_nan=True, show_nan_heatmap=False):
        """
        输出数据的基本统计信息，以及检测缺失值。
        [使用方法]:
            desc = read_data.desc_df(df)
            desc.describe_df(show_stats=True, stats_T=True, stats_detailed=False, show_nan=True, show_nan_heatmap=False)
        [Tips]:
            # 如果要把缺失值比例还原成数值，比如3.7%变成0.037，可以使用下面这行代码
            self.missing_info['缺失值比例'] = self.missing_info['缺失值比例'].apply(lambda x: float(x[:-1]) / 100)
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
            print("描述性统计信息:\n", self.numeric_stats)

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
            print("\n缺失值检测:\n", self.missing_info)
        if show_nan_heatmap:
            # 黑色是缺失值，白色是非缺失值
            sns.heatmap(self.df.isna(), cmap='gray_r', cbar_kws={"orientation": "vertical"}, vmin=0, vmax=1)
            plt.show()
            plt.close()
            # 还可以使用import missingno
            # missingno.matrix(self.df)

    def delete_missing_values(self, axis=0, how='any', inplace=True):
        """
        删除缺失值
        [Waring]: 该操作会直接在原数据上进行修改
        :param axis: 0表示删除行，1表示删除列
        :param how: any表示只要有缺失值就删除，all表示全部是缺失值才删除
        :param inplace: 是否在原数据上进行修改，不建议设置为False(设置为False时返回删除缺失值后的df)
        :return: 删除缺失值后的df(这在选用inplace=False时有用)
        """
        temp_df = self.df.dropna(axis=axis, how=how, inplace=inplace)
        # 下面这行是为了更新self.missing_info，便于在外部调用delete_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)
        return temp_df  # 返回删除缺失值后的df(这在选用inplace=False时有用)

    def fill_missing_values(self, fill_type='mean'):
        """
        填充缺失值
        [Warning]:
            该操作会直接在原数据上进行修改，也就是修改self.df
        [使用方法]:
            desc = read_data.desc_df(df)
            desc.fill_missing_values(fill_type=114514)  # 实际填充的时候可别逸一时误一世了
        [Tips]:
            目前支持的填充类型有：'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)'。
            如需删除缺失值请使用delete_missing_values。
        :param fill_type: 填补类型，支持 'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)'
        """
        if isinstance(fill_type, (int, float)):
            imputer = SimpleImputer(strategy='constant', fill_value=fill_type)
        else:
            imputer = SimpleImputer(strategy=fill_type)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
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

