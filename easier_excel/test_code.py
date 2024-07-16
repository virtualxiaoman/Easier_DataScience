# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# import easier_excel.read_data as read_data
# import easier_excel.draw_data as draw_data
# import easier_excel.cal_data as cal_data
# from easier_excel.draw_data import plot_xy, save_plot
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
# #     save_plot(fig, save_path='output/no', save_dpi=1200, save_format=ahh)
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
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# X = np.arange(24).reshape(8, 3)
# print(X)
# poly = PolynomialFeatures(3)
# print(poly.fit_transform(X))
# print(poly.fit_transform(X).shape)
# print(poly.get_feature_names_out())

# import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from imblearn.over_sampling import SMOTE
#
# class DataPreprocessing:
#     def __init__(self, df_numeric):
#         self.df_numeric = df_numeric
#
#     def transform_df(self, minmax=(0, 1), target=None):
#         self.demeaned_df = self.df_numeric - self.df_numeric.mean()
#
#         scaler = StandardScaler()
#         self.zscore_df = pd.DataFrame(scaler.fit_transform(self.df_numeric), columns=self.df_numeric.columns, index=self.df_numeric.index)
#
#         minmax_scaler = MinMaxScaler(feature_range=minmax)
#         self.minmax_df = pd.DataFrame(minmax_scaler.fit_transform(self.df_numeric), columns=self.df_numeric.columns, index=self.df_numeric.index)
#
#         if target is not None:
#             smote = SMOTE(random_state=42)
#             X = self.df_numeric.drop([target], axis=1)
#             y = self.df_numeric[target]
#             X_smote, y_smote = smote.fit_resample(X, y)
#             self.smote_df = pd.concat([X_smote, y_smote], axis=1)
#             self.smote_df.index = range(len(self.smote_df))  # 重置索引
#
# # 示例使用
# df_numeric = pd.DataFrame({
#     'feature1': [1, 2, 3, 4, 5, 1000],
#     'feature2': [10, 20, 30, 40, 50, -1000],
#     'u_score': [0, 0, 1, 0, 1, 1]
# })
#
# preprocessor = DataPreprocessing(df_numeric)
# preprocessor.transform_df(target='u_score')
#
# # 获取处理后的数据
# df_minmax = preprocessor.minmax_df
#
# df_zscore = preprocessor.zscore_df
# df_smote = preprocessor.smote_df
#
# # 确保 'u_score' 列正确赋值回去
# df_minmax['u_score'] = df_numeric['u_score'].values
# df_zscore['u_score'] = df_numeric['u_score'].values
#
# print("Min-Max Scaled DataFrame:")
# print(df_minmax)
# print("Z-score Standardized DataFrame:")
# print(df_zscore)
# print("SMOTE Resampled DataFrame:")
# print(df_smote)

# import pandas as pd
# from sklearn.feature_selection import VarianceThreshold
#
# # 示例数据
# data = {
#     'feature1': [1, 2, 3, 4, 5],
#     'feature2': [2.1, 3.4, 1.2, 4.3, 5.1],
#     'feature3': [0, 0, 0, 0, 0],  # 常数特征
#     'feature4': [1, 1, 1, 0, 0],  # 低方差特征
#     'target': [0, 1, 0, 1, 0]
# }
# df = pd.DataFrame(data)
# X = df.drop(columns=['target'])
#
# # 计算每个特征的方差
# variances = X.var()
# print("每个特征的方差：")
# print(variances)
#
# # 设置合理的阈值
# threshold = 0.1
#
# # 使用VarianceThreshold进行特征选择
# selector = VarianceThreshold(threshold=threshold)
# X_new = selector.fit_transform(X)
#
# selected_features = X.columns[selector.get_support()]
# print("选择的特征是：")
# print(selected_features)
#
# # 输出选择后的DataFrame
# df_new = pd.DataFrame(X_new, columns=selected_features)
# print(df_new)

# import cProfile
#
# cProfile.run('np.std(np.random.rand(100000000))')

# from itertools import combinations
#
# users = ['A', 'B', 'C', 'D']
# pairs = list(combinations(users, 2))
#
# print(pairs)
#
# import networkx as nx
# from networkx.algorithms import community
#
# # 构建商品图
# G = nx.Graph()
#
# # 假设 swing_scores 是商品对及其 Swing 分数的列表，例如 [(item1, item2, weight), ...]
# swing_scores = [
#     ('A', 'B', 0.9), ('A', 'C', 0.8), ('B', 'C', 0.8),
#     ('C', 'D', 0.8), ('D', 'E', 0.2), ('E', 'A', 0.1)
# ]
#
# # 添加边和权重
# for item1, item2, weight in swing_scores:
#     G.add_edge(item1, item2, weight=weight)
#
# # 执行标签传播算法进行聚类
# communities = community.label_propagation_communities(G)
#
# # 将商品分配到聚类
# item_to_community = {}
# for community_id, community_items in enumerate(communities):
#     for item in community_items:
#         item_to_community[item] = community_id
#
# # 计算聚类层面的相似度 s2(i,j)
# def calculate_s2(item1, item2):
#     if item_to_community[item1] == item_to_community[item2]:
#         return 1.0  # 相同聚类中的商品相似度为 1
#     else:
#         return 0.0  # 不同聚类中的商品相似度为 0
#
# # 示例计算
# item1 = 'A'
# item2 = 'B'
# s2_value = calculate_s2(item1, item2)
# print(f's2({item1},{item2}) = {s2_value}')
#
# item1 = 'A'
# item2 = 'E'
# s2_value = calculate_s2(item1, item2)
# print(f's2({item1},{item2}) = {s2_value}')


import numpy as np

# 定义矩阵 R 和 Q
P = np.array([[0.1, 0.9, 0.6],
              [0.8, 0.5, 0.4]])

Q = np.array([[0.7, 0.2, 0.3, 0.4],
              [0.1, 0.6, 0.9, 0.2],
              [0.5, 0.8, 0.4, 0.1]])

# 计算矩阵乘法 R * Q
result = np.dot(P, Q)

# 输出结果
print("结果矩阵 P * Q:")
print(result)
