import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joypy
import os

import torch

import pandas as pd
from pandas.plotting import parallel_coordinates

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from easier_excel.utils import DFUtils
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.Colorful_Console import func_warning as func_w


def set_plot_format_(ax, **kwargs):
    """
    设置图的格式
    :param ax: 传入的ax
    :param kwargs: 传入的参数，包含参数：
     font_name: 默认SimSun，还能取值SimHei，KaiTi，Times New Roman，Arial等
     title: 标题
     x_label: x轴标签
     y_label: y轴标签
     show_grid: plt.grid(show_grid)
     show_legend: ax.legend()
     x_labelsize: x轴的label的大小
     y_labelsize: y轴的label的大小
     x_ticksize: x轴的tick的大小
     y_ticksize: y轴的tick的大小
     x_rotation: x轴逆时针旋转角度
     y_rotation: y轴逆时针旋转角度
     xlim: x轴的范围
     ylim: y轴的范围
     xscale: x轴的scale
     yscale: y轴的scale
     set_aspect：设置纵横比，默认是'auto'，也可以是'equal'
    :return: plt, ax
    """
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    title = kwargs.get('title', 'Title')
    x_label = kwargs.get('x_label', 'x')
    y_label = kwargs.get('y_label', 'y')
    show_grid = kwargs.get('show_grid', True)
    show_legend = kwargs.get('show_legend', True)
    x_labelsize = kwargs.get('x_labelsize', 12)
    y_labelsize = kwargs.get('y_labelsize', 12)
    x_ticksize = kwargs.get('x_ticksize', 12)
    y_ticksize = kwargs.get('y_ticksize', 12)
    x_rotation = kwargs.get('x_rotation', 0)
    y_rotation = kwargs.get('y_rotation', 0)
    x_lim = kwargs.get('x_lim', None)
    y_lim = kwargs.get('y_lim', None)
    x_scale = kwargs.get('x_scale', None)
    y_scale = kwargs.get('y_scale', None)
    set_aspect = kwargs.get('set_aspect', 'auto')

    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=x_labelsize)
    ax.set_ylabel(y_label, fontsize=y_labelsize)

    ax.grid(show_grid)
    ax.set_aspect(set_aspect)

    if show_legend:
        if ax.get_legend() is not None:
            ax.legend()

    if x_lim is not None and len(x_lim) == 2:
        ax.set_xlim(x_lim)
    if y_lim is not None and len(y_lim) == 2:
        ax.set_ylim(y_lim)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    # 对于x和y是字符串的情况，需要设置刻度
    if x is not None and isinstance(x[0], str):
        ax.set_xticks(ticks=list(range(len(x))))  # 设置x轴刻度为索引值
        ax.set_xticklabels(labels=x, rotation=x_rotation, fontsize=x_ticksize)  # 设置x轴刻度标签为特征名称，并旋转标签
    else:
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=x_ticksize)
    if y is not None and isinstance(y[0], str):
        ax.set_yticks(ticks=list(range(len(y))))
        ax.set_yticklabels(labels=y, rotation=y_rotation, fontsize=y_ticksize)
    else:
        ax.tick_params(axis='y', rotation=y_rotation, labelsize=y_ticksize)  # 相当于plt.yticks(fontsize=y_ticksize)

    return ax


def plot_LR_(x, y, ax, LR_color, LR_linestyle, LR_digits=2):
    """
    在原图的基础上绘制线性回归拟合直线。
    :param x: x轴
    :param y: y轴(原始数据点，等待拟合)
    :param ax: ax
    :param LR_color: 线性回归直线的颜色
    :param LR_linestyle: 线性回归直线的形式
    :param LR_digits: 线性回归直线的系数保留的小数位数
    :return: 绘制了线性回归拟合直线的ax
    """
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(x).reshape(-1, 1), np.array(y))
    x_LR = np.linspace(min(x), max(x), 100).reshape(-1, 1)  # 生成预测用的 x 值
    y_LR = lin_reg.predict(x_LR)  # 生成拟合直线的预测值
    ax.plot(x_LR, y_LR, color=LR_color, linestyle=LR_linestyle,
            label=f'y={lin_reg.coef_[0]:.{LR_digits}f}x+{lin_reg.intercept_:.{LR_digits}f}')
    return ax


def save_plot(plt=None, save_path=None, save_name=None, save_dpi=300, save_format='png'):
    """
    保存图片，默认存为png格式，保存到save_path路径下的save_name.png
    :param plt: plt
    :param save_path: 保存路径
    :param save_name: 保存名称
    :param save_dpi: 保存的dpi
    :param save_format: 保存的格式，有png,jpg,svg
    :return:
    """
    if plt is None:
        raise ValueError("plt不能为None")
    supported_formats = ['png', 'svg', 'jpg']  # 可支持的文件格式
    if save_format not in supported_formats:
        func_w(save_plot,
               warning_text=f"不支持的保存格式'{save_format}'，支持的格式有：{', '.join(supported_formats)}。\n"
                            f"这里自动更改为'.png'，如有需要，请自行更改为正确的格式",
               modify_tip="请检查格式是否正确")
        save_format = 'png'
    if save_name is None or save_name == "":
        func_w(save_plot, warning_text='保存的名字不能为空，这里使用默认的名称"未命名"。，如有需要，请自行更改为正确的名称',
               modify_tip="请检查是否正确填写了参数save_name")
        save_name = "未命名"
    if isinstance(save_path, str) and isinstance(save_name, str):
        if not os.path.exists(save_path):
            func_w(save_plot, warning_text=f'路径"{save_path}"不存在，已经为你创建', modify_tip="请检查路径是否正确")
            os.makedirs(save_path)
        # 必须先保存再plt.show()，不然show会释放缓冲区里的图像
        plt.savefig(os.path.join(save_path, f"{save_name}.{save_format}"), dpi=save_dpi)  # dpi为了调节清晰度
    else:
        ValueError("save_path和save_name必须是字符串")


def plot_xy(x, y, title="Title", label='label', color='blue', linestyle='-', x_label='x', y_label='y', alpha=1,
            show_plt=True,
            use_LR=False, use_ax=False, ax=None, **kwargs):
    """
    绘制曲线图
    [使用示例]:
    使用ax的示例1（绘制多个子图）：
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0] = plot_xy(x, y1, label='$\sin$', color='blue', title='$f(x) = \sin(x)$', ax=axs[0], use_ax=True, show_plt=False)
        axs[1] = plot_xy(x, y2, label='$\cos$', color='green', title='$f(x) = \cos(x)$', ax=axs[1], use_ax=True, show_plt=False)
        plt.show()
    使用ax的示例2（绘制在同一幅图上）：
        fig, ax = plt.subplots(figsize=(10, 6))
        axs[0] = plot_xy(x, y1, label='Sin', color='blue', ax=ax, use_ax=True, show_plt=False)
        axs[1] = plot_xy(x, y2, label='Cos', color='green', ax=ax, use_ax=True, show_plt=False)
        plt.show()
    [注意事项]:
        为了便于操作，使用的是plt.rcParams['font.sans-serif'] = [font_name]，会导致全局的字体都变成这个字体。也就是如果绘制了
      多幅子图，前面的子图的字体会被新子图的字体覆盖。解决方法是单独设置，比如：
        axs[0].set_title('把标题换个字体', fontsize=14, fontname='SimSun')
      这样能够单独设置某个字体。

    :param x: x轴数据点
    :param y: y轴数据点
    :param title: 标题
    :param label: 标签
    :param color: 线条颜色 默认blue。可选['blue', 'green', 'red', 'black', 'purple', 'pink', 'orange', 'cyan']
    :param linestyle: 线条样式 默认-。可选['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    :param x_label: x轴标签
    :param y_label: y轴标签
    :param alpha: 透明度
    :param show_plt: 是否plt.show()
    :param use_LR: 是否使用LinearRegression来拟合数据并绘制
    :param use_ax: 是否是使用ax，如果是就直接返回ax，然后在调用此函数的地方绘制
    :param ax: 在use_ax的情况下应该传入的参数
    :param kwargs: 其他参数，包括：
        # 保存图片的参数
        save_path: 保存路径
        save_name: 保存的名字
        save_dpi: 保存的dpi
        # 设置图片格式的参数
        font_name: 字体，默认Times New Roman，还能取值SimSun, SimHei，KaiTi，Arial等
        show_grid: plt.grid(show_grid)
        show_legend: ax.legend()
        x_labelsize: x轴的label的大小
        y_labelsize: y轴的label的大小
        x_ticksize: x轴的tick的大小
        y_ticksize: y轴的tick的大小
        x_rotation: x轴逆时针旋转角度
        y_rotation: y轴逆时针旋转角度
        x_lim: x轴的范围，比如(-1, 5)
        y_lim: y轴的范围，比如(-1, 5)
        x_scale: {"linear线性", "log对数", "symlog包含正负的对数", "logit使用逻辑回归", ...}或ScaleBase对象
        y_scale: 同xscale
        adjust_params: 用于传入plt.subplots_adjust()，比如
            adjust_params = {'top': 0.93, 'bottom': 0.15, 'left': 0.09, 'right': 0.97, 'hspace': 0.2, 'wspace': 0.2}
        # 线性回归的参数
        LR_color： 线性回归拟合直线的颜色，默认green
        LR_linestyle: 线性回归的线条样式 默认--
        LR_digits: 线性回归的系数保留的小数位数，默认2
    :return ax: 如果use_ax=True的话
    """
    save_path = kwargs.get('save_path', None)
    save_name = kwargs.get('save_name', 'x-y图')
    save_dpi = kwargs.get('save_dpi', 300)
    font_name = kwargs.get('font_name', 'Times New Roman')
    show_grid = kwargs.get('show_grid', True)
    show_legend = kwargs.get('show_legend', True)
    x_labelsize = kwargs.get('x_labelsize', 12)
    y_labelsize = kwargs.get('y_labelsize', 12)
    x_ticksize = kwargs.get('x_ticksize', 12)
    y_ticksize = kwargs.get('y_ticksize', 12)
    x_rotation = kwargs.get('x_rotation', 0)
    y_rotation = kwargs.get('y_rotation', 0)
    x_lim = kwargs.get('x_lim', None)
    y_lim = kwargs.get('y_lim', None)
    x_scale = kwargs.get('x_scale', None)
    y_scale = kwargs.get('y_scale', None)
    adjust_params = kwargs.get('adjust_params', None)
    LR_color = kwargs.get('LR_color', 'green')
    LR_linestyle = kwargs.get('LR_linestyle', '--')
    LR_digits = kwargs.get('LR_digits', 2)
    # 检查是否使用ax
    if ax is not None:
        if not use_ax:
            func_w(func=plot_xy, warning_text="在没有use_ax的情况下，是不会返回ax的", modify_tip="使用use_ax=True")
    elif ax is None:
        if use_ax:
            func_w(func=plot_xy, warning_text="在use_ax的情况下，最好是主动传入ax，这种情况下是默认使用函数内的ax",
                   modify_tip="传入自己的ax")
        _, ax = plt.subplots()
    # 绘图
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha)
    if use_LR:
        plot_LR_(x=x, y=y, ax=ax, LR_color=LR_color, LR_linestyle=LR_linestyle, LR_digits=LR_digits)
    ax = set_plot_format_(
        x=x, y=y, ax=ax, font_name=font_name, title=title, x_label=x_label, y_label=y_label,
        show_grid=show_grid, show_legend=show_legend,
        x_labelsize=x_labelsize, y_labelsize=y_labelsize, x_ticksize=x_ticksize, y_ticksize=y_ticksize,
        x_rotation=x_rotation, y_rotation=y_rotation, x_lim=x_lim, y_lim=y_lim, x_scale=x_scale, y_scale=y_scale,
        adjust_params=adjust_params)
    if adjust_params is None:
        plt.tight_layout()
    else:
        plt.subplots_adjust(**adjust_params)
    # 保存/显示
    if isinstance(save_path, str) and isinstance(save_name, str):
        save_plot(plt, save_path, save_name, save_dpi)
    if use_ax:
        return ax
    if show_plt:
        plt.show()
    plt.close()


def plot_xys(x, y_list, labels=None, colors=None, linestyles=None, alpha=1, axes=None):
    """
    在同一幅图上绘制多个y。另外，使用plot_xy更便于自定义，plot_xys只是一个简化版。
    暂不支持9个及以上的y直接作为输入，除非自行传入colors, linestyles以与y匹配
    使用示例：
        plot_xys(x, [y1, y2, y3], labels=['Sin', 'Cos', 'Tan'], colors=['blue', 'green', 'red'], axes=(-5, 5, -5, 5))
    :param x: x轴
    :param y_list: 若干个y
    :param labels: y的标签
    :param colors: 颜色list
    :param linestyles: 线条形式list
    :param alpha: 透明度
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
        # 输出各个y的length
        print(CT("Error in func").red(), CT("plot_xys").yellow(), CT("。 请注意你的y的长度是否一致：").red())
        for i in range(lines_num):
            print(CT(f"y{i}的长度为{y_lengths[i]}").pink())
        raise ValueError("All y arrays must have the same length")

    for i in range(lines_num):
        ax.plot(x, y_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i], alpha=alpha)

    if axes is not None:
        ax.set_xlim((axes[0], axes[1]))
        ax.set_ylim((axes[2], axes[3]))
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_f_and_df(x, y=None, y_func=None, use_ax=False, ax=None, just_f=False, just_df=False, font_name='Times New Roman',
                  x_label=r'$x$', y_label_f=r'$f(x)$', y_label_df=r'$\frac{df}{dx}$', label_f=r'$f(x)$',
                  label_df=r'$\frac{df}{dx}$', save_path=None, save_name=None):
    """
    绘制f与其导函数。
    use_ax=False的示例：
        xm_dd.plot_f_and_df(x, y_func=f, use_ax=False)
    使用ax的示例：
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        axs = xm_dd.plot_f_and_df(x, y_func=f, use_ax=True, ax=axs)
        plt.show()
    :param x: 传入x坐标的值
    :param y: 传入y的值
    :param y_func: 传入函数f(x)，通过函数f计算y
    :param use_ax: 是否使用ax
    :param ax: 如果use_ax，就需要传入你的ax
    :param just_f: 是否只绘制f(x)
    :param just_df: 是否只绘制df(x)
    :param font_name: 字体，默认TNR
    :param x_label: x轴的标签
    :param y_label_f: f(x)的y轴标签
    :param y_label_df: df(x)的y轴标签
    :param label_f: f(x)的label
    :param label_df: df(x)的label
    :param save_path: 保存的路径
    :param save_name: 保存的名字
    :return: ax(如果use_ax的话)
    """
    if y is None:
        y = torch.zeros_like(x)
        for i, x_i in enumerate(x):
            y[i] = y_func(x_i)
    y.backward(torch.ones_like(y))
    dy = x.grad
    if not use_ax:
        fig, axs = plt.subplots(1, 2)
        axs[0] = plot_xy(x.detach().numpy(), y.detach().numpy(), use_ax=True, show_plt=False, label=label_f,
                         title=label_f, ax=axs[0], x_label=x_label, y_label=y_label_f, font_name=font_name)
        axs[1] = plot_xy(x.detach().numpy(), dy.detach().numpy(), use_ax=True, show_plt=False, label=label_df,
                         title=label_df, ax=axs[1], x_label=x_label, y_label=y_label_df, font_name=font_name)
        if isinstance(save_path, str) and isinstance(save_name, str):
            save_plot(plt, save_path, save_name, save_dpi=300)
        plt.show()
    else:
        if ax is None:
            print(CT("Warning in func").red(), CT("plot_f_and_df").yellow(),
                  CT(": 在use_ax的情况下，应该主动传入ax，这种情况下不会继续为你绘图了").red())
        else:
            if just_f:
                ax = plot_xy(x.detach().numpy(), y.detach().numpy(), use_ax=True, show_plt=False, label=label_f,
                             title=label_f, x_label=x_label, y_label=y_label_f, ax=ax, font_name=font_name)
                return ax
            if just_df:
                ax = plot_xy(x.detach().numpy(), dy.detach().numpy(), use_ax=True, show_plt=False, label=label_df,
                             title=label_df, x_label=x_label, y_label=y_label_df, ax=ax, font_name=font_name)
                return ax
            ax[0] = plot_xy(x.detach().numpy(), y.detach().numpy(), use_ax=True, show_plt=False, label=label_f,
                            title=label_f, x_label=x_label, y_label=y_label_f, ax=ax[0], font_name=font_name)
            ax[1] = plot_xy(x.detach().numpy(), dy.detach().numpy(), use_ax=True, show_plt=False, label=label_df,
                            title=label_df, x_label=x_label, y_label=y_label_df, ax=ax[1], font_name=font_name)
            return ax


def draw_density(x, ax=None, show_plt=True, **kwargs):
    """
    绘制密度图
    :param x: 传入的数据
    :param ax: 传入的ax
    :param show_plt: 是否plt.show()
    :return: None
    """
    kwargs.setdefault('title', 'Probability Density')
    kwargs.setdefault('x_label', 'Value')
    kwargs.setdefault('y_label', 'Density')
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(x, fill=True, ax=ax, common_norm=False, linewidth=1, palette="viridis")
    ax = set_plot_format_(ax=ax, **kwargs)
    if show_plt:
        plt.show()
        plt.close()
    return ax


def draw_scatter(x, y, ax=None, show_plt=True,
                 if_colorful=False, c=None, cmap='viridis', norm=None,
                 **kwargs):
    """
    绘制散点图
    [使用方法]:
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1  # 半径
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        c = np.arange(len(x))
        draw_scatter(x, y, show_plt=True, if_colorful=True, c=c, norm=None, set_aspect='equal')
    :param x: 传入的x数据
    :param y: 传入的y数据
    :param ax: 传入的ax
    :param show_plt: 是否plt.show()
    :param if_colorful: 是否依据c与cmap绘制颜色
    :param c: 颜色，可以指定为y，或者其他形式(如 'blue', 'green', 'red', 'black', 'purple', 'pink', 'orange', 'cyan')
    :param cmap: 颜色映射，可选值有：'viridis', 'RdYlBu', 'coolwarm', 'spring', 'summer', 'autumn', 'winter' 等
    :param norm: 归一化，比如norm=plt.Normalize(y.min(), y.max())
    :return: None
    """
    kwargs.setdefault('title', 'Scatter')
    kwargs.setdefault('x_label', 'x')
    kwargs.setdefault('y_label', 'y')
    label = kwargs.get('label', None)
    if ax is None:
        ax = plt.gca()
    if if_colorful:
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, label=label)
    else:
        ax.scatter(x, y, c=c, label=label)
    ax = set_plot_format_(ax=ax, **kwargs)
    if show_plt:
        plt.show()
        plt.close()
    return ax

def pair_feature_plot(df, kde_hist=True, diag=None, kind=None):  # todo 要在draw_df里也加上这个函数
    """
    绘制特征对的散点图
    [使用示例]:
        pair_feature_plot(df, kde_hist=False, diag='hist', kind='reg')
    :param df: 数据集
    :param kde_hist: 是否这样绘制: upper:scatter, lower:kde, diag:hist
    :param diag: [kde_hist=False时有效]对角线的图形类型，可取值为 'auto', 'hist', 'kde', None
    :param kind: [kde_hist=False时有效]非对角线的图形类型， 可取值为 'scatter', 'kde', 'hist', 'reg'
    :return:
    """
    sns.set(style="ticks")  # 设置风格为ticks，即：坐标轴上有刻度
    if kde_hist:
        if diag is not None or kind is not None:
            func_w(warning_text='kde_hist=True时，diag和kind参数无效', func=pair_feature_plot, modify_tip='修改为kde_hist=False')
        # 设置右上角和左下角的对角线为散点图和概率密度曲面图
        g = sns.PairGrid(df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_diag(sns.histplot, kde_kws={'color': 'k'})
    else:
        sns.pairplot(df, diag_kind=diag, kind=kind)  # 对角线为直方图/概率密度曲线，非对角线为散点图

    plt.tight_layout()
    plt.show()
    plt.close()


class draw_df(DFUtils):
    def __init__(self, df):
        """
        初始化
        :param df: 可以传入df或者df.copy()
        """
        super().__init__(df)

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

    def draw_scatter(self, x_name, y_name, target_name=None, show_plt=True, font_name='Times New Roman',
                     color='blue', colors=('red', 'green', 'blue', 'yellow', 'purple', 'orange'),
                     **kwargs):
        """
        以两个属性为x,y轴，以target属性作为标签，绘制散点图
        :param x_name: x轴的属性名称
        :param y_name: y轴的属性名称
        :param target_name: 目标(分类、预测...)的属性名称
        :param show_plt: 是否plt.show()
        :param font_name: 默认Times New Roman，还能取值SimSun, SimHei，KaiTi，Arial等
        :param color: 单标签时候的颜色
        :param colors: 多标签时候的颜色
        :param kwargs: 其他参数，其中重要的包括：
            title: 图片标题
            alpha: 透明度
            save_path: 图片保存的地址
            save_name: 图片保存的名字
            save_dpi: 图片保存的dpi
            adjust_params: 用于传入plt.subplots_adjust()
        """
        title = kwargs.get('title', f"Scatter: {x_name}&{y_name}")
        alpha = kwargs.get('alpha', 0.5)
        x_label = kwargs.get('x_label', x_name)
        y_label = kwargs.get('y_label', y_name)
        adjust_params = kwargs.get('adjust_params', None)
        save_path = kwargs.get('save_path', None)
        save_name = kwargs.get('save_name', f"Scatter--{x_name}&{y_name}")
        save_dpi = kwargs.get('save_dpi', 300)

        _, ax = plt.subplots()
        x_data = self.df[x_name]
        y_data = self.df[y_name]

        # todo 颜色指定不够好，比如RdYlBu_r，还有unique_values应该从小到大排序
        if target_name is not None:
            # 如果你是二分类问题，或者需要自定义颜色，可以使用下面两行代码来替代本部分代码：
            # target_colors = self.df[target_name].map({值1: 'green', 值2: 'red'})  # 根据target_name设置颜色
            # plt.scatter(x_data, y_data, c=target_colors, alpha=0.5)
            unique_values = self.df[target_name].unique()  # 获取target_name列的唯一值
            if len(unique_values) > len(colors):
                ValueError(f"颜色数量不足，需要{len(unique_values)}种颜色，但只提供了{len(colors)}种，请自行重写这个代码"
                           f"另外也应该注意类别是否是离散型的，如果是连续型的，就不应该使用这个函数。")
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
                ax = draw_scatter(
                    x_data[self.df[target_name] == target_value], y_data[self.df[target_name] == target_value], ax=ax,
                    c=color, show_plt=show_plt, label=f'{target_name}={target_value}',
                    alpha=alpha, title=title, x_label=x_label, y_label=y_label, show_legend=False,
                    **kwargs)
            ax.legend()
        else:
            ax = draw_scatter(x=x_data, y=y_data, ax=ax, show_plt=show_plt, c=color,
                              alpha=alpha, title=title, x_label=x_label, y_label=y_label, **kwargs)  # 添加标签

        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 为了显示负号
        if adjust_params is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(**adjust_params)
        if save_path is not None:
            save_plot(plt, save_path, save_name, save_dpi=save_dpi)
        if show_plt:
            plt.show()
        plt.close()

    def draw_all_scatter(self, target_name=None, save_path=None, all_scatter=False, **kwargs):
        # print(self.columns)
        if target_name is None:
            if all_scatter:
                for i in range(0, len(self.columns)):
                    for j in range(0, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], show_plt=False,
                                          save_path=save_path, **kwargs)
            else:
                for i in range(0, len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j],
                                          save_path=save_path, show_plt=False, **kwargs)
        else:
            if all_scatter:
                for i in range(0, len(self.columns)):
                    for j in range(0, len(self.columns)):
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], target_name=target_name,
                                          save_path=save_path, show_plt=False, **kwargs)
            else:
                for i in range(0, len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        # j从i+1开始相当于只取上三角矩阵
                        if self.columns[i] == target_name or self.columns[j] == target_name:
                            continue  # 如果要绘制的图里的某个坐标轴是target_name属性就跳过当前循环
                        self.draw_scatter(x_name=self.columns[i], y_name=self.columns[j], target_name=target_name,
                                          save_path=save_path, show_plt=False, **kwargs)

    def draw_feature_importance(self, target_name, feature_name=None, target_change="No", print_top=5, descending_draw=True,
                                save_path=None, show_plt=True, save_name="特征重要性", **kwargs):
        """
        通过传入的目标属性target_name来判断其他属性的重要性
        :param target_name: target是哪一个属性
        :param feature_name: 传入的特征名称，默认为None，即所有的特征
        :param target_change: 如何制定target的判断标准，"No":按照原属性值，"mean":按照平均值为分隔标准(高于均值为1，低于均值为0)
        :param print_top: 输出前print_top个重要的特征
        :param descending_draw: 是否从大到小绘制特征曲线图
        :param save_path: 保存路径
        :param show_plt: 是否plt.show()
        :param save_name: 保存的文件名字
        """
        if feature_name is None:
            X = self.df.drop(target_name, axis=1)
        else:
            X = self.df[feature_name]
        rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
        y = self.df[target_name]
        # 如果target_change是mean，就将y转换为0和1，0代表小于均值，1代表大于均值
        if target_change == 'mean':
            y = (y > self.df[target_name].mean()).astype(int)  # 将大于均值的值设为1，其余设为0
        rf_clf.fit(X, y)

        # 非降序绘制（按照特征的顺序）
        if not descending_draw:
            plot_xy(list(X.columns), rf_clf.feature_importances_, title="特征重要性", x_label='特征名称',
                    y_label='重要性程度', x_rotation=0, label="重要性")
        # 降序绘制（默认，按照特征的重要性程度）
        features_scores = {name: score for name, score in zip(X.columns, rf_clf.feature_importances_)}
        sorted_features_scores = sorted(features_scores.items(), key=lambda x: x[1], reverse=True)
        top_features_scores = sorted_features_scores[:print_top]
        print(f'最重要的前{print_top}个特征与重要性程度是：')
        for name, score in top_features_scores:
            print(f"{name}: {score:.4f}")
        if descending_draw:
            features, scores = zip(*top_features_scores)
            # features = [feat_score[0] for feat_score in top_features_scores]
            # scores = [feat_score[1] for feat_score in top_features_scores]
            print(features)
            plot_xy(features, scores, title=f"Feature Importance with Target {target_name}",
                    x_label='Feature Name', y_label='Importance', label="Importance",
                    x_rotation=45, save_path=save_path, show_plt=show_plt, save_name=save_name, **kwargs)

    def draw_density(self, feature_name, target_name=None, use_mean=False, show_plt=True, save_path=None, classify=True,
                     colors=('red', 'green', 'blue', 'yellow', 'purple', 'orange'), font_name='SimSun',
                     x_labelsize=12, y_labelsize=12, x_ticksize=12, y_ticksize=12, adjust_params=None):
        """
        绘制概率密度图
        :param feature_name: 要绘制密度的属性名称
        :param target_name: 目标(分类、预测...)的属性名称
        :param use_mean: 对于连续值的分类密度绘制，是否使用target的均值
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
            sns.kdeplot(self.df[:][feature_name], label=feature_name, fill=True, color='blue')
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


if __name__ == '__main__':
    pass
    # todo 将山脊图、PCP合并到draw_df里，然后删除下面的代码。另外可以试着重构draw_df里的函数，分为描述性绘图和计算型绘图
    # df, y = load_iris_df()
    # df['species'] = y
    # joypy.joyplot(df, ylim='own', by='species', figsize=(8, 6), alpha=0.6, color=['r', 'g', 'b', 'y'])
    # # joypy.joyplot(df, ylim='own')
    # plt.show()
    # plt.close()
    # # 绘制平行坐标图 (Parallel Coordinate Plot, PCP)
    # parallel_coordinates(df, 'species', color=['r', 'g', 'b', 'y'])
    # plt.show()
    # plt.close()










