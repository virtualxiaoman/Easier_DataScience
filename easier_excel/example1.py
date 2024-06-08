from sklearn.cluster import KMeans
import easier_excel.read_data as rd
import easier_excel.draw_data as dd
import easier_excel.cal_data as cd

rd.set_pd_option(max_show=True, float_type=True, decimal_places=2)

path = '../input/CharacterData.xlsx'
df = rd.read_df(path)

print("----------读取数据信息----------")
desc_df = rd.desc_df(df)
desc_df.show_df(head_n=5, tail_n=0, show_columns=False, show_dtypes=False)  # "称号","备注"这两行有缺失值
df_4 = df[df['星级'] == 4]
df_5 = df[df['星级'] == 5]
print("全部数据", end='')
desc_df.describe_df(stats_detailed=False)
print("星级=4的", end='')
rd.desc_df(df_4).describe_df(stats_detailed=False, show_nan=False)
print("星级=5的", end='')
rd.desc_df(df_5).describe_df(stats_detailed=False, show_nan=False)


print("----------处理异常值与缺失值----------")
# 缺失值处理(因为数值型数据无缺失值，这里其实可以不处理)
# desc_df.fill_missing_values(fill_type='mean')  # 均值填补
print(desc_df.missing_info)
# 异常值处理(因为这里都是真实的数据，所以不删除，只是看看策划搞了些什么玩意出来)
desc_df.process_outlier(method='IQR', show_info=True, process_type='ignore')
print("异常值：", end='')
print(desc_df.df.iloc[48:49])  # (第48行)荒泷一斗的防御力为异常值


print("----------数据处理(回归)----------")
cal_df = cd.Linear(df)
cal_df.cal_linear(["星级", "生命值", "攻击力", "防御力"], '模')
cal_df.cal_logistic(["生命值", "攻击力", "防御力"], '星级', pos_label=5)

exit(1)
print("----------数据分析(绘图)----------")  # 绘图部分代码正在重构中
df_main = df[['星级', '生命值', '攻击力', '防御力']].copy()
draw_df = dd.draw_df(df_main)
adjust_params = {'top': 0.93, 'bottom': 0.15, 'left': 0.09, 'right': 0.97, 'hspace': 0.2, 'wspace': 0.2}
draw_df.draw_corr(save_path='../output/easier_excel', v_minmax=(-1, 1), adjust_params=adjust_params, show_plt=False)
draw_df.draw_scatter('生命值', '攻击力', target_name='星级', save_path='../output/easier_excel/scatter_all', show_plt=False)
draw_df.draw_all_scatter(target_name='星级', save_path='../output/easier_excel/scatter_effective')
draw_df.draw_all_scatter(target_name='星级', save_path='../output/easier_excel/scatter_all', all_scatter=True)
draw_df.draw_feature_importance(target_name='星级', save_path='../output/easier_excel', show_plt=False)
for feature_name in ['生命值', '攻击力', '防御力']:
    draw_df.draw_density(target_name="星级", feature_name=feature_name, show_plt=False, save_path='../output/easier_excel/density')
    draw_df.draw_density(target_name="星级", feature_name=feature_name, show_plt=False,
                         save_path='../output/easier_excel/density', classify=False)

# 目前暂时没加入聚类算法
df_feature = df_main.copy().drop(columns=['星级'])
kmeans_c2 = KMeans(n_clusters=2, n_init=10)
kmeans_c2.fit(df_feature)
centers = kmeans_c2.cluster_centers_
print("聚类中心：", centers)  # [[10417.12217391   236.39369565   653.4026087 ] [13148.92261905   257.995        744.7802381 ]]


