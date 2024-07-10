# 将数据输出到md文档里，本工具仅供方便使用，实际上的格式、内容建议之后自行调整。比如序号就不容易对上。
import os
import time

class ToMd:
    path = "output/data_analysis.md"
    path_dir = os.path.dirname(path)
    pic_dir_name = "md_pic"
    pic_dir = os.path.join(path_dir, pic_dir_name).replace('\\', '/')

    def __init__(self):
        pass

    # 清空md
    def clear_md(self, clear_md=True, clear_pic=False):
        # 不建议clear_pic，因为不能确保其他md的图片不会被删除
        if clear_md:
            with open(self.path, 'w', encoding='utf-8') as file:
                file.seek(0)
                file.truncate()
        if clear_pic:
            for filename in os.listdir(self.pic_dir):
                file_path = os.path.join(self.pic_dir, filename).replace('\\', '/')
                os.remove(file_path)

    # text -> md
    def text_to_md(self, md_text=None, md_flag=False, md_bold=False, md_color=None, md_h=None, **kwargs):
        """
        text -> md
        :param md_text: 原文本
        :param md_flag: 是否追加写入
        :param md_bold: 是否加粗，使用 **{测试文本}** 这样的写法
        :param md_color: 颜色，使用 $\color{Pink} {测试文本} $ 这样的写法
        :param md_h: md_h=多少就是多少个#
        :return:
        """
        if md_flag and md_text is not None:
            if md_color:
                md_text = f"$\color{{{md_color}}} {{{md_text}}} $"  # $\color{Pink} {测试文本} $
            if md_bold and md_h is None:
                md_text = f"**{md_text}**"
            if md_h is not None and isinstance(md_h, int):
                h = "#"*md_h
                md_text = f"{h} {md_text}"
            with open(self.path, 'a', encoding='utf-8') as file:
                file.write(md_text + '\n\n')

    # df -> md
    def df_to_md(self, md_df=None, md_flag=False, md_model='a', md_index=False, **kwargs):
        if md_flag and md_df is not None:
            markdown_table = md_df.to_markdown(index=md_index)
            with open(self.path, md_model, encoding='utf-8') as file:
                file.write(markdown_table + '\n\n')

    # pic -> md
    def pic_to_md(self, plt=None, md_flag=False, md_title="pic", md_dpi=1200, md_model='a', **kwargs):
        if md_flag and plt is not None:
            # 将plt保存在self.path_dir下，文件名为plt的md_title加上时间戳
            timestamp = int(time.time())
            pic_path = os.path.join(self.pic_dir, f"{md_title}_{timestamp}.png").replace('\\', '/')
            plt.savefig(pic_path, dpi=md_dpi)
            md_pic_path = os.path.join(ToMd.pic_dir_name, f"{md_title}_{timestamp}.png").replace('\\', '/')  # 因为是md的相对路径
            # 将图片路径写入md文件
            with open(self.path, md_model, encoding='utf-8') as file:
                file.write(f"![{md_title}]({md_pic_path})\n\n")

    # 检查path是否存在
    def check_path(self):
        if not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir)
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                f.write("")

    # 更新路径
    def update_path(self, path_new=None):
        """
        可以通过此方法指定更新路径，也可以直接  ToMd.path = 新路径
        但建议无论使用的是什么方法，只要更新路径，最好调用此函数来自动check_path
        :param path_new:
        :return:
        """
        if path_new is not None:
            ToMd.path = path_new
        ToMd.path_dir = os.path.dirname(ToMd.path)
        ToMd.pic_dir = os.path.join(ToMd.path_dir, ToMd.pic_dir_name).replace('\\', '/')
        print("当前ToMd.path: ", ToMd.path)
        print("当前ToMd.path_dir: ", ToMd.path_dir)
        print("当前ToMd.pic_dir: ", ToMd.pic_dir)
        self.check_path()

