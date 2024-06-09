# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# import easier_excel.read_data as read_data
# import easier_excel.draw_data as draw_data
# import easier_excel.cal_data as cal_data
# from easier_excel.draw_data import plot_xy, _save_plot
# from scipy import stats

# read_data.set_pd_option(max_show=True, float_type=True)
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


# # 生成一组对数正态分布的数据
# x = stats.lognorm.rvs(0.5, size=1000)
# # 绘制直方图
# plt.hist(x, bins=50, density=True, alpha=0.6, color='g')
# plt.show()
# # 对x进行z-score标准化
# x_zscore = stats.zscore(x)
# # 绘制直方图
# plt.hist(x_zscore, bins=50, density=True, alpha=0.6, color='g')
# plt.show()


#
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# import numpy as np
# from easier_excel.read_data import interpolate_data
#
# # x_train = np.linspace(0, 6, num=7, endpoint=True)
# # y_train = np.sin(x_train) + x_train/6  # 相当于加上噪声
# #
# # x_test = np.linspace(0, 6, num=500, endpoint=True)
# #
# # methods = ['linear', 'nearest', 'cubic', 'previous', 'lagrange']
# # for kind in methods:
# #     interpolate_data(x_train, y_train).interpolate(method=kind, show_plt=False)
# from scipy.interpolate import griddata, interp2d, Rbf, RectBivariateSpline, SmoothBivariateSpline, RegularGridInterpolator
#
# x_train = np.linspace(0, 6, num=7, endpoint=True)
# y_train = np.sin(x_train) + x_train / 6  # 相当于加上噪声
# methods = ['lagrange']
# for kind in methods:
#     interp = interpolate_data(x_train, y_train)
#     interp.interpolate(method=kind)
#     print(interp.f_predict(-0.5))
#
# def y_func(x1, x2):
#     v = (2 * x1 + x2) * np.exp(-2 * (x1 ** 2 + x2 ** 2))
#     return v
# methods = ['linear', 'cubic']
#
# x1_data = np.linspace(-1, 1, 5)
# x2_data = np.linspace(-1, 1, 5)
# y_data = y_func(x1_data, x2_data)  # (5,)
# for kind in methods:
#     interp = interpolate_data(x1_data, x2_data, y_data)
#     interp.interpolate(method=kind, show_plt=False, plt_2d=True)
#     y_test = interp.f_predict(-1.1232, 2.0004)
#     print(y_test)
#
# x1_data = np.linspace(-1, 1, 5)
# x2_data = np.linspace(-1, 1, 5)
# xx1_data, xx2_data = np.meshgrid(x1_data, x2_data)
# yy_data = y_func(xx1_data, xx2_data)  # (5, 5)
# for kind in methods:
#     interp = interpolate_data(xx1_data, xx2_data, yy_data)
#     interp.interpolate(method=kind, show_plt=False, plt_2d=True)
#     y_test = interp.f_predict(0.0032, 0.0004)
#     print(y_test)
# exit(1)
# import numpy as np
#
#
# def func(x, y):
#     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
#
#
# grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:, 0], points[:, 1])
# print(points.shape, values.shape)  # (1000, 2) (1000,)
# print(grid_x.shape, grid_y.shape)  # (100, 200) (100, 200)
# print((grid_x, grid_y))  # (2, 100, 200)
#
# from scipy.interpolate import griddata
#
# grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
# import matplotlib.pyplot as plt
#
# plt.subplot(221)
# plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')
# plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
# plt.title('Original')
# plt.subplot(222)
# plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
# plt.title('Nearest')
# plt.subplot(223)
# plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
# plt.title('Linear')
# plt.subplot(224)
# plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
# plt.title('Cubic')
# plt.gcf().set_size_inches(6, 6)
# plt.show()
#
# import plotly.express as px
# import numpy as np
# import pandas as pd
#
# df = px.data.gapminder()
# df.rename(columns={"country": "country_or_territory"}, inplace=True)
# print(df.head(5))
# # 按年份和大洲分组，再对pop列求和
# df_pop_continent_over_t = df.groupby(['year', 'continent'], as_index=False).agg({'pop': 'sum'})
# print(df_pop_continent_over_t.head(5))
# fig = px.bar(df_pop_continent_over_t,
#              x='year', y='pop',
#              width=600, height=380,
#              color='continent',
#              labels={"year": "Year",
#                      "pop": "Population"})
# fig.show()
#
# fig = px.line(df_pop_continent_over_t,
#               x='year', y='pop',
#               width=600, height=380,
#               color='continent',
#               labels={"year": "Year",
#                       "pop": "Population"})
# fig.show()
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(24).reshape(8, 3)
print(X)
poly = PolynomialFeatures(3)
print(poly.fit_transform(X))
print(poly.fit_transform(X).shape)
print(poly.get_feature_names_out())
