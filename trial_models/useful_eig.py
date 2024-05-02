import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easier_excel.draw_data import draw_scatter, plot_xy
from easier_excel.read_data import desc_df

def show_transform(x, y, T, c, T_text='T'):
    """
    显示变换前后的散点图
    :param x: x坐标
    :param y: y坐标
    :param T: 变换矩阵，仅支持2x2
    :param c: 颜色
    :param T_text: 变换矩阵的文本
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    adjust_params = {'top': 0.9, 'bottom': 0.1, 'left': 0.09, 'right': 0.97, 'hspace': 0.2, 'wspace': 0.2}
    plt.subplots_adjust(**adjust_params)
    plt.rc('text', usetex=True)

    axes[0] = draw_scatter(x, y, ax=axes[0], show_plt=False,
                           if_colorful=True, c=c, norm=None, set_aspect='equal', title='Original: $x$')

    eig_vals, eig_vecs = np.linalg.eig(T)
    print('特征值：', eig_vals, '\n特征向量：', eig_vecs)
    # 如果特征值不是实数，那么eig_text就给出原文本
    if not np.isreal(eig_vals).all():
        eig_text = r'$\lambda_1=%.2f+(%.2fj)$' % (eig_vals[0].real, eig_vals[0].imag) + '\n' + \
                   r'$\lambda_2=%.2f+(%.2fj)$' % (eig_vals[1].real, eig_vals[1].imag)
    else:
        eig_text = r'$\lambda_1=%.2f, \lambda_2=%.2f$' % (eig_vals[0], eig_vals[1])
    T_text += '\n\n' + '%.2f   %.2f\n%.2f   %.2f' % (T[0, 0], T[0, 1], T[1, 0], T[1, 1])
    axes[1].quiver(-1, 0, 2, 0, scale_units='xy', angles='xy', scale=1, color='r', headwidth=5)
    axes[1].text(0, 0.1, T_text, fontsize=12, ha='center')
    axes[1].text(0, -0.2, eig_text, fontsize=12, ha='center')
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].axis('off')

    points = np.row_stack((x, y))
    T_points = np.dot(T, points)
    axes[2] = draw_scatter(T_points[0, :], T_points[1, :], ax=axes[2], show_plt=False,
                           if_colorful=True, c=c, norm=None, set_aspect='equal', title='Transform: $Tx$')

    plt.show()
    plt.close()


def eig_decomposition(x, y, T, add_base=False, special_show=False):
    """
    :param add_base: 是否绘制基向量
    :param special_show: 是否特殊显示，为简便，仅支持np.array([[1.5, -np.sqrt(3)/2], [-np.sqrt(3)/2, 2.5]])的输入
    特征值分解
    """
    # 对T进行特征值分解T= PDP^(-1)
    eigenvalues, eigenvectors = np.linalg.eig(T)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    print("P^(-1):\n", P_inv)
    print("P:\n", P)
    print("D:\n", D)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    if add_base:
        # 绘制基向量，使用直线绘制(先绘制是因为后面的有set_aspect='equal'，懒得这里也设置set_aspect='equal' 了)
        # 基底[1, 0]和[0, 1]
        axes[0] = plot_xy(x=np.linspace(0, 1, 100), y=np.linspace(0, 0, 100), ax=axes[0], show_plt=False, color='red',
                          use_ax=True)
        axes[0] = plot_xy(x=np.linspace(0, 0, 100), y=np.linspace(0, 1, 100), ax=axes[0], show_plt=False, color='red',
                          use_ax=True)
        # P^(-1)基底[-0.8660254, 0.5]和[-0.5, -0.8660254]
        # D基底[-0.8660254, 0.5]和[-0.5, -0.8660254]
        # P基底[-0.8660254, -0.5]和[0.5, -0.8660254]
        axes[1] = plot_xy(x=np.linspace(0, P_inv[0][0], 100), y=np.linspace(0, P_inv[1][0], 100), ax=axes[1],
                          show_plt=False, color='red', use_ax=True)
        axes[1] = plot_xy(x=np.linspace(0, P_inv[0][1], 100), y=np.linspace(0, P_inv[1][1], 100), ax=axes[1],
                          show_plt=False, color='red', use_ax=True)
        axes[2] = plot_xy(x=np.linspace(0, np.dot(D, P_inv)[0][0], 100), y=np.linspace(0, np.dot(D, P_inv)[1][0], 100),
                          ax=axes[2], show_plt=False, color='red', use_ax=True)
        axes[2] = plot_xy(x=np.linspace(0, np.dot(D, P_inv)[0][1], 100), y=np.linspace(0, np.dot(D, P_inv)[1][1], 100),
                          ax=axes[2],show_plt=False, color='red', use_ax=True)
        axes[3] = plot_xy(x=np.linspace(0, np.dot(np.dot(P, D), P_inv)[0][0], 100), y=np.linspace(0, np.dot(np.dot(P, D), P_inv)[1][0], 100),
                          ax=axes[3], show_plt=False, color='red', use_ax=True)
        axes[3] = plot_xy(x=np.linspace(0, np.dot(np.dot(P, D), P_inv)[0][1], 100), y=np.linspace(0, np.dot(np.dot(P, D), P_inv)[1][1], 100),
                          ax=axes[3], show_plt=False, color='red', use_ax=True)

    # 依次绘制原图x,p^(-1)x, Dp^(-1)x,PDP^(-1)x
    axes[0] = draw_scatter(x, y, ax=axes[0], show_plt=False, if_colorful=True, c=c, norm=None,
                           set_aspect='equal', title='Original: $x$')
    points = np.row_stack((x, y))
    P_inv_points = np.dot(P_inv, points)
    axes[1] = draw_scatter(P_inv_points[0, :], P_inv_points[1, :], ax=axes[1], show_plt=False,
                           if_colorful=True, c=c, norm=None, set_aspect='equal', title='Transform1: $P^{-1}x$')
    D_P_inv_points = np.dot(D, P_inv_points)
    axes[2] = draw_scatter(D_P_inv_points[0, :], D_P_inv_points[1, :], ax=axes[2], show_plt=False,
                           if_colorful=True, c=c, norm=None, set_aspect='equal', title='Transform2: $DP^{-1}x$')
    P_D_P_inv_points = np.dot(P, D_P_inv_points)
    axes[3] = draw_scatter(P_D_P_inv_points[0, :], P_D_P_inv_points[1, :], ax=axes[3], show_plt=False,
                           if_colorful=True, c=c, norm=None, set_aspect='equal', title='Transform3: $PDP^{-1}x$')
    if special_show:
        axes[1].set_title(r"Transform1: $P^{-1}x \left ( \theta = \frac{5}{6}\pi \right )$")
        axes[2].set_title(r"Transform2: $DP^{-1}x \left ( \Lambda = 1,3 \right )$")
        axes[3].set_title(r"Transform3: $PDP^{-1}x \left ( \theta = \frac{7}{6}\pi \right )$")

    plt.show()
    plt.close()


# 画个圆
theta = np.linspace(0, 2 * np.pi, 100)
r = 1  # 半径
x = r * np.cos(theta)
y = r * np.sin(theta)
c = np.arange(len(x))

# # 缩放
# T_S = np.diag([1, 2])
# show_transform(x, y, T_S, c, T_text='Scale')
#
# # 旋转
# sita = 0.5 * np.pi
# T_R = np.array([[np.cos(sita), -np.sin(sita)], [np.sin(sita), np.cos(sita)]])
# show_transform(x, y, T_R, c, T_text='Rotate')
#
# # 等比缩放+旋转
# a = 1
# b = np.sqrt(3)
# T_SeR = np.array([[a, -b], [b, a]])
# show_transform(x, y, T_SeR, c, T_text='Scale(equally) and Rotate')

# # 缩放+旋转
# sita = 0.25 * np.pi
# T_R = np.array([[np.cos(sita), -np.sin(sita)], [np.sin(sita), np.cos(sita)]])
# T_S = np.diag([2, 1])
# T_SR = np.dot(T_R, T_S)  # sita=0.25*np.pi时是[[sqrt(2), -sqrt(2)/2], [sqrt(2), sqrt(2)/2]]
# show_transform(x, y, T_SR, c, T_text='Scale and Rotate')


# # 特征值分解 以对称矩阵为例
# T_SM = np.array([[1.5, -np.sqrt(3)/2], [-np.sqrt(3)/2, 2.5]])
# show_transform(x, y, T_SM, c, T_text='Symmetric Matrix')
# eig_decomposition(x, y, T_SM, add_base=True, special_show=True)

# 对8*8随机矩阵特征值分解
T = np.random.rand(8, 8)
T = np.dot(T, T.T)  # 只是为了输出的是实数
eig_vals, eig_vecs = np.linalg.eig(T)
print('特征值：', eig_vals)
print('特征向量：', eig_vecs)
# 将特征值按照从大到小排序，同时也将特征向量按照特征值的顺序排序。可以发现特征值只有少部分很大，大部分都很小
idx = np.argsort(-eig_vals)
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# 将diag(eig_vals)变成dataframe
eig_vals = np.diag(eig_vals)
df_val = pd.DataFrame(eig_vals)
df_vec = pd.DataFrame(eig_vecs)
dd_val = desc_df(df_val)
dd_vec = desc_df(df_vec)
dd_val.draw_heatmap(scale=False)
dd_vec.draw_heatmap(scale=False)

