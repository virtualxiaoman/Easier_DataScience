# bilibili Recommend System 手动实现b站的推荐系统(非官方)

from easier_excel import read_data
from easier_excel import cal_data
from easier_excel import draw_data
df_origin = read_data.read_df("input/history_xm.xlsx")
# view_percent这一列是以百分比形式展示的，需要转换为数值型数据
df_origin['view_percent'] = df_origin['view_percent'].str.rstrip('%').astype('float') / 100.0
# 将弹幕、评论、点赞、投币、收藏、分享这六列的数据转化为比例
df_origin['dm_rate'] = df_origin['dm'] / df_origin['view']
df_origin['reply_rate'] = df_origin['reply'] / df_origin['view']
df_origin['like_rate'] = df_origin['like'] / df_origin['view']
df_origin['coin_rate'] = df_origin['coin'] / df_origin['view']
df_origin['fav_rate'] = df_origin['fav'] / df_origin['view']
df_origin['share_rate'] = df_origin['share'] / df_origin['view']
# time这一列是时间戳，数值较大。减去最小值，使时间戳从0开始
df_origin['time'] = df_origin['time'] - df_origin['time'].min()

desc_df = read_data.desc_df(df_origin)
desc_df.show_df(head_n=5, tail_n=0, show_columns=False, show_dtypes=False)
desc_df.describe_df(stats_detailed=False)

df_num = df_origin.select_dtypes(include=['number'])
desc_df = read_data.desc_df(df_num)
# desc_df.draw_heatmap(scale=True)

cal_df = cal_data.Linear(df_num)
cal_df.cal_linear(["view", "dm", "reply", "like", "coin", "fav", "share", "tid", "up_follow", "up_followers"], 'u_score')
cal_df = cal_data.SVM(df_num)
cal_df.cal_svr(["view", "dm", "reply", "like", "coin", "fav", "share", "tid", "up_follow", "up_followers"], 'u_score',
               draw_svr=False, kernel='rbf')

draw_df = draw_data.draw_df(df_num)
# draw_df.draw_corr(v_minmax=(-1, 1))
# 在df_num中去掉u_like, u_coin, u_fav这三列，因为这三列是计算u_score的依据
# feature_name_main = [col for col in df_num.columns if col not in ['u_like', 'u_coin', 'u_fav', 'u_score']]
# draw_df.draw_feature_importance(target_name='u_score', feature_name=feature_name_main,
#                                 descending_draw=True, print_top=10)
draw_df.draw_all_scatter(target_name='u_score', save_path='output/bilibili_RS/scatter_all')

