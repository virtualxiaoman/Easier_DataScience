# df工具
import re

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