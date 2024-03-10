import pandas as pd


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
    else:
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")
        pd.reset_option("display.max_rows")
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
        您可能还需要：
            print(df['离散型数据的属性名称'].value_counts())
            print(df['连续性数据的属性名称'].describe())
        来查看某个属性的值的分布情况。
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

    def describe_df(self, show_stats=True, stats_T=True, stats_detailed=False, show_nan=True):
        """
        输出数据的基本统计信息，以及检测缺失值。
        :param show_stats: 描述性统计信息。显示DataFrame中数值列的描述性统计信息，如均值、标准差、最大值、最小值等
        :param stats_T: 是否转置输出stats
        :param stats_detailed: 是否输出stats的详细信息
        :param show_nan: 缺失值检测。检查DataFrame中是否存在缺失值，并显示每列缺失值的数量或比例。
        """
        if stats_T:
            self.numeric_stats = self.df.describe().T
        else:
            self.numeric_stats = self.df.describe()
        if not stats_detailed:
            self.numeric_stats = self.numeric_stats[['count', 'mean', 'min', 'max']]
        if show_stats:
            print("描述性统计信息:\n", self.numeric_stats)

        missing_values = self.df.isnull().sum()
        total_rows = self.df.shape[0]  # 行数，相当于len(self.df)
        missing_percentage = missing_values / total_rows
        self.missing_info = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失值比例': missing_percentage
        })
        self.missing_info['缺失值比例'] = self.missing_info['缺失值比例'].apply(lambda x: '{:.1%}'.format(x))
        if show_nan:
            print("\n缺失值检测:\n", self.missing_info)

    def fill_missing_values(self):
        """
        目前仅支持object用nan填补，其他的用均值填补
        """
        missing_cols = self.missing_info[self.missing_info['缺失值数量'] != 0].index  # 缺失值数量不为 0 的属性
        # print(missing_cols)
        for col in missing_cols:
            dtype = self.df[col].dtype
            if dtype == 'object':
                self.df[col].fillna('nan', inplace=True)  # 用 'nan' 填充缺失值
            else:  # 如果是数值类型
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)  # 用均值填充缺失值
        # 下面这一行是为了更新self.missing_info，便于在外部调用fill_missing_values后能直接查看修改后的self.missing_info的值
        self.describe_df(show_stats=False, stats_T=True, stats_detailed=False, show_nan=False)


# 一些可能的读入方法以作为记录
# df_main['index'] = range(1, df_main.shape[0] + 1)  # 但不能将index放在第一行，下面一行代码可以：
# df_main.insert(0, 'index', range(1, df_main.shape[0] + 1))
# print(df_main.iloc[5])  # 获取第6行
