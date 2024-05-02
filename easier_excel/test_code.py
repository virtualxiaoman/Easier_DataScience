import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import easier_excel.read_data as xm_rd
import easier_excel.draw_data as xm_dd
import easier_excel.cal_data as xm_cd
from easier_excel.draw_data import plot_xy, _save_plot
from scipy import stats

# xm_rd.set_pd_option(max_show=True, float_type=True, decimal_places=2)
#
# path = '../input/CharacterData.xlsx'
# df = xm_rd.read_df(path)
#
# x = stats.norm.rvs(size=100)
# xm_cd.cal_skew_kurtosis(x)
# # x = pd.DataFrame(x, columns=['x'])
# # draw_df = xm_dd.draw_df(x)
# # draw_df.draw_density(target_name=None, classify=False, feature_name='x')

x = np.array([1, 2, 3, 4, 5])
y = np.array([0.1, 0.14, -0.19, 0.116, 0.125])
x2 = np.array([1, 3, 5, 7, 9])
y2 = np.array([1, 9, 25, 49, 81])
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0] = plot_xy(x, y, use_ax=True, show_plt=False, ax=axs[0], title='aaa',
                 font_name='SimSun')
axs[0].set_title('把标题换个字体', fontsize=14, fontname='SimSun')
axs[1] = plot_xy(x2, y2, use_ax=True, show_plt=True, ax=axs[1], title='我超这是中文',
                 font_name='KaiTi')

# for ahh in ['png', 'svg', 'jpg', 'ahh']:
#     _save_plot(fig, save_path='output/no', save_dpi=1200, save_format=ahh)
#     break
plt.show()
font_list = plt.rcParams['font.sans-serif']
print("当前使用的字体列表：", font_list)

