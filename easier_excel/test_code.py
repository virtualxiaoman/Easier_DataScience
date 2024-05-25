import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import easier_excel.read_data as read_data
import easier_excel.draw_data as draw_data
import easier_excel.cal_data as cal_data
from easier_excel.draw_data import plot_xy, _save_plot
from scipy import stats

read_data.set_pd_option(max_show=True, float_type=True)
#
# path = '../input/CharacterData.xlsx'
# df = xm_rd.read_df(path)
#
# x = stats.norm.rvs(size=100)
# xm_cd.cal_skew_kurtosis(x)
# # x = pd.DataFrame(x, columns=['x'])
# # draw_df = xm_dd.draw_df(x)
# # draw_df.draw_density(target_name=None, classify=False, feature_name='x')

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([0.1, 0.14, -0.19, 0.116, 0.125])
# x2 = np.array([1, 3, 5, 7, 9])
# y2 = np.array([1, 9, 25, 49, 81])
# fig, axs = plt.subplots(1, 2, figsize=(10, 6))
# axs[0] = plot_xy(x, y, use_ax=True, show_plt=False, ax=axs[0], title='aaa',
#                  font_name='SimSun')
# axs[0].set_title('把标题换个字体', fontsize=14, fontname='SimSun')
# axs[1] = plot_xy(x2, y2, use_ax=True, show_plt=True, ax=axs[1], title='我超这是中文',
#                  font_name='KaiTi')
#
# # for ahh in ['png', 'svg', 'jpg', 'ahh']:
# #     _save_plot(fig, save_path='output/no', save_dpi=1200, save_format=ahh)
# #     break
# plt.show()
# font_list = plt.rcParams['font.sans-serif']
# print("当前使用的字体列表：", font_list)

# path = "D:\HP\Desktop\杂七杂八数据集\B站科普短视频\数据.sav"
# df = read_data.read_df(path)
# # print(df.head())
# # print(df.info())
# # 将随机的列与行的数据变成nan
# df = df.mask(np.random.random(df.shape) < 0.1)  # mask函数是将符合条件的数据变成nan
# # print(df.head())
#
# desc = read_data.desc_df(df)
# desc.fill_missing_values(fill_type=114514)  # 实际填充的时候可别逸一时误一世了
# desc.describe_df(show_stats=True, stats_T=False)
#
# print("----")
# from easier_tools.print_variables import print_variables_function
# print_variables_function(desc.describe_df, show_stats=True, stats_T=False)


# 生成一组对数正态分布的数据
x = stats.lognorm.rvs(0.5, size=1000)
# 绘制直方图
plt.hist(x, bins=50, density=True, alpha=0.6, color='g')
plt.show()
# 对x进行z-score标准化
x_zscore = stats.zscore(x)
# 绘制直方图
plt.hist(x_zscore, bins=50, density=True, alpha=0.6, color='g')
plt.show()





