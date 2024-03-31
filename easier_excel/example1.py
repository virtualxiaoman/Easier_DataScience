from sklearn.cluster import KMeans
import easier_excel.read_data as xm_rd
import easier_excel.draw_data as xm_dd

xm_rd.set_pd_option(max_show=True, float_type=True, decimal_places=2)

path = '../input/CharacterData.xlsx'
df = xm_rd.read_df(path)

desc_df = xm_rd.desc_df(df)
desc_df.show_df(head_n=0, tail_n=0, show_columns=False, show_dtypes=False)
df_4 = df[df['星级'] == 4]
df_5 = df[df['星级'] == 5]
desc_df.describe_df(stats_detailed=False)
xm_rd.desc_df(df_4).describe_df(stats_detailed=False, show_nan=False)
xm_rd.desc_df(df_5).describe_df(stats_detailed=False, show_nan=False)
desc_df.fill_missing_values()
# print(desc_df.missing_info)

print('--------------')

df_main = df[['星级', '生命值', '攻击力', '防御力']].copy()
draw_df = xm_dd.draw_df(df_main)
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


