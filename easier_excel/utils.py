# df工具

class DFUtils:
    def __init__(self, df):
        self.df = df
        self.df_numeric = self.df.select_dtypes(include=['number'])  # 数值型数据
        self.df_non_numeric = self.df.select_dtypes(exclude=['number'])  # 非数值型数据
        self.shape = None  # df大小
        self.columns = None  # df属性
        self.dtypes = None  # 数据类型

        self.init_util_params()

    def init_util_params(self):
        """
        初始化工具参数
        :return: None
        """
        self.shape = self.df.shape
        self.columns = self.df.columns
        self.dtypes = self.df.dtypes

