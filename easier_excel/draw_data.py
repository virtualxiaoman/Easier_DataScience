import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from easier_excel.Colorful_Console import ColoredText as CT

def plot_xy(x, y, axes=None, title="Title", label='label', color='blue', linestyle='-', x_label='x', y_label='y',
            show_plt=True, save_path=None, save_name='x-y图', font_name='SimSun', show_grid=True, show_legend=True,
            x_labelsize=12, y_labelsize=12, x_ticksize=12, y_ticksize=12, x_rotation=0, adjust_params=None,
            use_LR=False, LR_color="green", LR_linestyle='--', use_ax=False, ax=None):
    """
    绘制曲线图
    使用ax的示例1（绘制多个子图）：
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0] = xm_dd.plot_xy(x, y1, label='$\sin$', color='blue', title='$f(x) = \sin(x)$', ax=axs[0], use_ax=True, show_plt=False)
        axs[1] = xm_dd.plot_xy(x, y2, label='$\cos$', color='green', title='$f(x) = \cos(x)$', ax=axs[1], use_ax=True, show_plt=False)
        plt.show()
    使用ax的示例2（绘制在同一幅图上）：
        fig, ax = plt.subplots(figsize=(10, 6))
        axs[0] = xm_dd.plot_xy(x, y1, label='Sin', color='blue', ax=ax, use_ax=True, show_plt=False)
        axs[1] = xm_dd.plot_xy(x, y2, label='Cos', color='green', ax=ax, use_ax=True, show_plt=False)
        plt.show()
    :param x: x轴数据点
    :param y: y轴数据点
    :param axes: 坐标轴范围，比如(-1, 5, -1, 5)
    :param title: 标题
    :param label: 标签
    :param color: 线条颜色 默认blue
    :param linestyle: 线条样式 默认-
    :param x_label: x轴标签
    :param y_label: y轴标签
    :param show_plt: 是否显示
    :param save_path: 保存路径
    :param save_name: 保存的名字
    :param font_name: 默认SimSun，还能取值SimHei，KaiTi，Times New Roman，Arial等
    :param show_grid: plt.grid(show_grid)
    :param show_legend: ax.legend()
    :param x_labelsize: x轴的label的大小
    :param y_labelsize: y轴的label的大小
    :param x_ticksize: x轴的tick的大小
    :param y_ticksize: y轴的tick的大小
    :param x_rotation: x轴逆时针旋转角度
    :param adjust_params: 用于传入plt.subplots_adjust()，比如
            adjust_params = {'top': 0.93, 'bottom': 0.15, 'left': 0.09, 'right': 0.97, 'hspace': 0.2, 'wspace': 0.2}
    :param use_LR: 是否使用LinearRegression来拟合数据并绘制
    :param LR_color： 线性回归拟合直线的颜色，默认green
    :param LR_linestyle: 线性回归的线条样式 默认--
    :param use_ax: 是否是使用ax，如果是就直接返回ax，然后在调用此函数的地方绘制
    :param ax: 在use_ax的情况下应该传入的参数
    :return ax: 如果use_ax=True的话
    """
    if ax is not None:
        if not use_ax:
            print(CT("Warning in func").red(), CT("plot_xy").yellow(),
                  CT(": 在没有use_ax的情况下，是不会返回ax的，建议使用:").red(), CT("use_ax=True").pink())
    if ax is None:
        if use_ax:
            print(CT("Warning in func").red(), CT("plot_xy").yellow(),
                  CT(": 在use_ax的情况下，最好是主动传入ax，这种情况下是默认使用函数内的ax").red())
        _, ax = plt.subplots()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
    ax.plot(x, y, label=label, color=color, linestyle=linestyle)
    if use_LR:
        lin_reg = LinearRegression()
        lin_reg.fit(np.array(x).reshape(-1, 1), np.array(y))
        x_LR = np.linspace(min(x), max(x), 100).reshape(-1, 1)  # 生成预测用的 x 值
        y_LR = lin_reg.predict(x_LR)  # 生成拟合直线的预测值
        ax.plot(x_LR, y_LR, color=LR_color, linestyle=LR_linestyle,
                label='y={:.2f}x+{:.2f}'.format(lin_reg.coef_[0], lin_reg.intercept_))
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=x_labelsize)
    ax.set_ylabel(y_label, fontsize=y_labelsize)
    ax.tick_params(axis='x', rotation=x_rotation, labelsize=x_ticksize)
    ax.tick_params(axis='y', labelsize=y_ticksize)  # 相当于plt.yticks(fontsize=y_ticksize)
    ax.grid(show_grid)
    if show_legend:
        ax.legend()
    if axes is not None:
        ax.set_xlim((axes[0], axes[1]))
        ax.set_ylim((axes[2], axes[3]))
    if adjust_params is None:
        plt.tight_layout()
    else:
        plt.subplots_adjust(**adjust_params)
    if isinstance(save_path, str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 必须先保存再plt.show()，不然show会释放缓冲区里的图像
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=300)  # dpi为了调节清晰度
    if use_ax:
        return ax
    if show_plt:
        plt.show()
    plt.close()


def plot_xys(x, y_list, labels=None, colors=None, linestyles=None, axes=None):
    """
    绘制多个y
    暂不支持7个及以上的y作为输入，除非自行传入colors, linestyles
    :param x: x轴
    :param y_list: 若干个y
    :param labels: y的标签
    :param colors: 颜色list
    :param linestyles: 线条形式list
    :param axes: 坐标轴范围，比如(-1, 5, -1, 5)
    :return:
    """
    fig, ax = plt.subplots()
    lines_num = len(y_list)

    # 如果标签、颜色和线型未提供，则使用默认值
    if labels is None:
        labels = [f"Line {i}" for i in range(lines_num)]
    if colors is None:
        colors = ['blue', 'green', 'red', 'black', 'purple', 'pink', 'orange', 'cyan'][:lines_num]
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'][:lines_num]

    # 检查所有y数组的长度是否相同
    y_lengths = [len(y) for y in y_list]
    if len(set(y_lengths)) != 1:
        raise ValueError("All y arrays must have the same length")

    for i in range(lines_num):
        ax.plot(x, y_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i])

    if axes is not None:
        ax.set_xlim((axes[0], axes[1]))
        ax.set_ylim((axes[2], axes[3]))
    ax.legend()
    ax.grid(True)
    plt.show()


class draw_df:
    def __init__(self, df):
        """
        初始化
        :param df: 可以传入df或者df.copy()
        """
        self.df = df
        self.shape = self.df.shape
        self.columns = self.df.columns

    def get_df(self):
        """
        如果在初始化这个类draw_df的时候传入的是df.copy()，那就不会对原来的df进行更改。
        此时如果需要获得更改后的df，请使用这个函数吧！
        """
        return self.df

    def draw_corr(self, show_plt=True, save_path=None, font_name='SimSun', print_corr=False, title='相关系数矩阵',
                  ignore_diagonal=False, corr_threshold=0.0, original_order=False, ascending_order=False,
                  x_rotation=75, x_fontsize=12, y_fontsize=12, v_minmax=None, adjust_params=None):
        """
        绘制相关系数矩阵图，并输出相关系数
        :param show_plt: 是否plt.show()
        :param save_path: 存储路径(文件夹名称)
        :param font_name: 默认SimSun，还能取值SimHei，KaiTi，Times New Roman，Arial等
        :param print_corr: 是否输出corr
        :param title: 标题
        :param ignore_diagonal: 是否忽视与自身的相关系数
        :param corr_threshold: 相关系数阈值，会输出高于此阈值的相关系数
        :param original_order: 是否原序输出高于corr_threshold的相关系数
        :param ascending_order: 从大到小/从小到大输出高于corr_threshold的相关系数（默认从大到小)
        :param x_rotation: 逆时针旋转多少度
        :param x_fontsize: x轴的字体大小
        :param y_fontsize: y轴的字体大小
        :param v_minmax: 修改heatmap里的vmin和vmax:vmin=v_minmax[0], vmax=v_minmax[1]
        :param adjust_params: 用于传入plt.subplots_adjust
        """
        corr = self.df.corr()
        if print_corr:
            print(corr)
        if ignore_diagonal:
            np.fill_diagonal(corr.values, np.nan)  # 这段代码用于屏蔽与自身的相关系数，让图片更加清晰但是不美观
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
        if adjust_params is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(**adjust_params)
        if v_minmax is None:
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values)
        else:
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values, vmin=v_minmax[0], vmax=v_minmax[1])
        plt.xticks(rotation=x_rotation, fontsize=x_fontsize)
        plt.yticks(fontsize=y_fontsize)
        plt.title(title)
        if isinstance(save_path, str):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # 必须先保存再plt.show()，不然show会释放缓冲区里的图像
            plt.savefig(os.path.join(save_path, f"相关系数矩阵.png"), dpi=300)  # dpi为了调节清晰度
        if show_plt:
            plt.show()
        plt.close()

        if corr_threshold != 0:
            print(f"相关系数的绝对值大于{corr_threshold}的有：")
            if original_order:
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > corr_threshold:
                            print(f"{corr.columns[i]}和{corr.columns[j]}的相关系数为:{corr.iloc[i, j]}")
            else:
                upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
                sorted_corr = upper_tri.unstack().sort_values(ascending=ascending_order)
                sorted_corr = sorted_corr[abs(sorted_corr) > corr_threshold]
                for index, value in sorted_corr.items():
                    feature1, feature2 = index
                    print(f"{feature1}和{feature2}的相关系数为:{value}")

    def draw_scatter(self, x_name, y_name, target_name=None, save_path=None, show_plt=True, title=None,
                     font_name='SimSun', color='blue', colors=('red', 'green', 'blue', 'yellow', 'purple', 'orange'),
                     x_labelsize=12, y_labelsize=12, x_ticksize=12, y_ticksize=12, adjust_params=None):
        """
        以两个属性为x,y轴，以target属性作为标签，绘制散点图
        :param x_name: x轴的属性名称
        :param y_name: y轴的属性名称
        :param target_name: 目标(分类、预测...)的属性名称
        :param save_path: 图片保存的地址
        :param show_plt: 是否plt.show()
        :param title: 图片标题
        :param font_name: 默认SimSun，还能取值SimHei，KaiTi，Times New Roman，Arial等
        :param color: 单标签时候的颜色
        :param colors: 多标签时候的颜色
        :param x_labelsize: x轴的label的大小
        :param y_labelsize: y轴的label的大小
        :param x_ticksize: x轴的tick的大小
        :param y_ticksize: y轴的tick的大小
        :param adjust_params: 用于传入plt.subplots_adjust()
        """
        x_data = self.df[x_name]
        y_data = self.df[y_name]
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
        if adjust_params is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(**adjust_params)
        if target_name is not None:
            # 如果你是二分类问题，或者需要自定义颜色，可以使用下面两行代码来替代本部分代码：
            # target_colors = self.df[target_name].map({值1: 'green', 值2: 'red'})  # 根据target_name设置颜色
            # plt.scatter(x_data, y_data, c=target_colors, alpha=0.5)
            unique_values = self.df[target_name].unique()  # 获取target_name列的唯一值
            # 方法1 使用cm
            # colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_values)))  # 生成对应数量的颜色
            # target_color_mapping = dict(zip(unique_values, colors))  # 类似于这样的形式{0:'green', 1:'red'}，但这里是RGBA
            # 方法2 自定义
            colors = colors  # 指定颜色列表，根据需要调整颜色数量
            target_color_mapping = dict(zip(unique_values, colors[:len(unique_values)]))  # 根据颜色数量来映射
            # 方法1 不含标签
            # target_colors = self.df[target_name].map(target_color_mapping)  # 使用map函数根据target_name的值获取相应的颜色
            # plt.scatter(x_data, y_data, c=target_colors, alpha=0.5)
            # 方法2 含标签
            for target_value, color in target_color_mapping.items():
                plt.scatter(x_data[self.df[target_name] == target_value], y_data[self.df[target_name] == target_value],
                            c=color, label=f'{target_name}={target_value}', alpha=0.5)
        else:
            plt.scatter(x_data, y_data, c=color, label='All', alpha=0.5)  # 添加标签
        if title is None:
            plt.title(f'x:{x_name}&y:{y_name}')
        else:
            plt.title(title)
        plt.xlabel(x_name, fontsize=x_labelsize)
        plt.ylabel(y_name, fontsize=y_labelsize)
        plt.xticks(fontsize=x_ticksize)
        plt.yticks(fontsize=y_ticksize)
        plt.grid(True)
        plt.legend()
        if isinstance(save_path, str):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f"{x_name}&{y_name}.png"), dpi=300)
        if show_plt:
            plt.show()
        plt.close()

    def draw_all_scatter(self, target_name=None, save_path=None, all_scatter=False):
        # print(self.columns)
        if target_name is None:
            if all_scatter:
                for i in range(0, len(self.columns)):
                    for j in range(0, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], show_plt=False,
                                          save_path=save_path)
            else:
                for i in range(0, len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j],
                                          save_path=save_path, show_plt=False)
        else:
            if all_scatter:
                for i in range(0, len(self.columns)):
                    for j in range(0, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], target_name=target_name,
                                          save_path=save_path, show_plt=False)
            else:
                for i in range(0, len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        # j从i+1开始相当于只取上三角矩阵
                        if self.columns[i] == target_name or self.columns[j] == target_name:
                            continue  # 如果要绘制的图里的某个坐标轴是target_name属性就跳过当前循环
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], target_name=target_name,
                                          save_path=save_path, show_plt=False)

    def draw_feature_importance(self, target_name, target_change="No", print_top=5, descending_draw=True,
                                save_path=None, show_plt=True, save_name="特征重要性"):
        """
        通过传入的目标属性target_name来判断其他属性的重要性
        :param target_name: target是哪一个属性
        :param target_change: 如何制定target的判断标准，"No":按照原属性值，"mean":按照平均值为分隔标准(高于均值为1，低于均值为0)
        :param print_top: 输出前print_top个重要的特征
        :param descending_draw: 是否从大到小绘制特征曲线图
        :param save_path: 保存路径
        :param show_plt: 是否plt.show()
        :param save_name: 保存的文件名字
        """
        X = self.df.drop(target_name, axis=1)
        rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
        if target_change == 'No':
            rf_clf.fit(X, self.df[target_name])
        elif target_change == 'mean':
            target_mean = self.df[target_name].mean()
            y = (self.df[target_name] > target_mean).astype(int)  # 将大于均值的值设为1，其余设为0
            rf_clf.fit(X, y)
        # for name, score in zip(X.columns, rf_clf.feature_importances_):
        #     print(name, score)
        if not descending_draw:
            plot_xy(X.columns, rf_clf.feature_importances_, title="特征重要性", x_label='特征名称',
                    y_label='重要性程度', x_rotation=0, label="重要性")
        print(f'最重要的前{print_top}个特征与重要性程度是：')
        features_scores = [(name, score) for name, score in zip(X.columns, rf_clf.feature_importances_)]
        sorted_features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)
        top_features_scores = sorted_features_scores[:print_top]
        for name, score in top_features_scores:
            print(name, score)
        if descending_draw:
            features = [feat_score[0] for feat_score in top_features_scores]
            scores = [feat_score[1] for feat_score in top_features_scores]
            plot_xy(features, scores, title=f"以{target_name}为目标的特征重要性", x_label='特征名称', y_label='重要性程度',
                    x_rotation=0, label="重要性", save_path=save_path, show_plt=show_plt, save_name=save_name)

    def draw_density(self, feature_name, target_name=None, use_mean=False, show_plt=True, save_path=None, classify=True,
                     colors=('red', 'green', 'blue', 'yellow', 'purple', 'orange'), font_name='SimSun',
                     x_labelsize=12, y_labelsize=12, x_ticksize=12, y_ticksize=12, adjust_params=None):
        """
        绘制概率密度图
        :param feature_name: 要绘制密度的属性名称
        :param target_name: 目标(分类、预测...)的属性名称
        :param use_mean: 是否使用target的均值
        :param show_plt: plt.show()
        :param save_path: 图片保存的地址
        :param classify: 绘制密度图的时候是否将各个类别区分开来
        :param colors: 多标签时候的颜色
        :param font_name: 字体名称
        :param x_labelsize: x轴的label的大小
        :param y_labelsize: y轴的label的大小
        :param x_ticksize: x轴的tick的大小
        :param y_ticksize: y轴的tick的大小
        :param adjust_params: 用于传入plt.subplots_adjust()
        """
        if classify:
            if use_mean:
                target_mean = self.df[target_name].mean()
                y = (self.df[target_name] > target_mean).astype(int)  # 将大于均值的值设为1，其余设为0
                unique_values = y.unique()  # 获取target_name列的唯一值
                colors = colors  # 指定颜色列表，根据需要调整颜色数量
                target_color_mapping = dict(zip(unique_values, colors[:len(unique_values)]))  # 根据颜色数量来映射
                for target_value, color in target_color_mapping.items():
                    sns.kdeplot(self.df[y == target_value][feature_name],
                                label=f'{target_name}={target_value}', fill=True, color=color)
            else:
                unique_values = self.df[target_name].unique()  # 获取target_name列的唯一值
                colors = colors  # 指定颜色列表，根据需要调整颜色数量
                target_color_mapping = dict(zip(unique_values, colors[:len(unique_values)]))  # 根据颜色数量来映射
                for target_value, color in target_color_mapping.items():
                    sns.kdeplot(self.df[self.df[target_name] == target_value][feature_name],
                                label=f'{target_name}={target_value}', fill=True, color=color)
        else:
            sns.kdeplot(self.df[:][feature_name], label='ALL', fill=True, color='blue')
        if adjust_params is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(**adjust_params)
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
        plt.xlabel(feature_name, fontsize=x_labelsize)
        plt.ylabel('密度', fontsize=y_labelsize)
        plt.xticks(fontsize=x_ticksize)
        plt.yticks(fontsize=y_ticksize)
        if classify:
            plt.title(f'特征"{feature_name}"以"{target_name}"为目标的密度')
        else:
            plt.title(f'特征"{feature_name}"的密度')
        plt.legend()
        if isinstance(save_path, str):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if classify:
                plt.savefig(os.path.join(save_path, f"{feature_name}在{target_name}下的密度.png"))
            else:
                plt.savefig(os.path.join(save_path, f"{feature_name}的密度.png"))
        if show_plt:
            plt.show()
        plt.close()

    def draw_boxplot(self, feature, show_plt=True, save_path=None, font_name='SimSun', x_rotation=0, standard=False,
                     title="箱型图", x_ticksize=12, y_ticksize=12, adjust_params=None):
        if adjust_params is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(**adjust_params)
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
        if standard:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(self.df[feature])
            df_scaled = pd.DataFrame(df_scaled, columns=self.df[feature].columns)
            df_scaled.boxplot()
        else:
            self.df[feature].boxplot()
        plt.title(title)
        plt.xticks(fontsize=x_ticksize, rotation=x_rotation)
        plt.yticks(fontsize=y_ticksize)
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path, dpi=300)  # dpi为了调节清晰度
        if show_plt:
            plt.show()
        plt.close()












