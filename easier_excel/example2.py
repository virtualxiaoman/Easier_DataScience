import pandas as pd
import easier_excel.read_data as xm_rd
import easier_excel.draw_data as xm_dd

# path = "../input/hsy/成绩总表.xlsx"
# df = xm_rd.read_df(path)
# df.loc[df['班级'] == '高三（25）班', '班级'] = '25'
# df.loc[df['班级'] == '高三（19）班', '班级'] = '19'
# df = df[df['班级'].isin(['19', '25'])]
# selected_columns = ['姓名', '班级', '语文', '数学', '英语', '物理', '化学']
# df = df[selected_columns]
# df = df.sort_index()
# output_path = "../input/hsy/t8.xlsx"
# df.to_excel(output_path, index=False)

path = "../input/hsy/t8.xlsx"
df = xm_rd.read_df(path).drop('姓名', axis=1)
df_19 = df[df['班级'] == 19]
df_25 = df[df['班级'] == 25]
desc_df = xm_rd.desc_df(df)
desc_df.describe_df(stats_detailed=False)
desc_df.fill_missing_values()
xm_rd.desc_df(df_19).describe_df(stats_detailed=False, show_nan=False)
xm_rd.desc_df(df_25).describe_df(stats_detailed=False, show_nan=False)
draw_df = xm_dd.draw_df(df)
draw_df_19 = xm_dd.draw_df(df_19)
draw_df_25 = xm_dd.draw_df(df_25)
for draw_df_i, num in zip([draw_df, draw_df_19, draw_df_25], ['all', '19', '25']):
    adjust_params = {'top': 0.93, 'bottom': 0.15, 'left': 0.09, 'right': 0.97, 'hspace': 0.2, 'wspace': 0.2}
    draw_df_i.draw_corr(save_path=f'../output/hsy/{num}', v_minmax=(-1, 1), adjust_params=adjust_params, show_plt=False)

draw_df.draw_all_scatter(target_name='班级', save_path='../output/hsy/scatter_effective')
draw_df.draw_feature_importance(target_name='班级', save_path='../output/hsy', show_plt=False)
for feature_name in ['语文', '数学', '英语', '物理', '化学']:
    draw_df.draw_density(target_name="班级", feature_name=feature_name, show_plt=False, save_path='../output/hsy/density')


