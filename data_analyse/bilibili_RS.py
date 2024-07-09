# Hands On bilibili Recommend System 动手实现b站的推荐系统(非官方)
import pandas as pd
from sklearn.model_selection import train_test_split

from easier_excel import read_data, cal_data, draw_data
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.to_md import ToMd
ToMd.path = "output/bilibili_RS/Bili_RS.md"  # 更改输出路径
ToMd = ToMd()
ToMd.update_path()  # 这是更改path需要做的必要更新
ToMd.clear_md(auto_clear=True)  # 清空前如有需要，务必备份

# 设置pandas显示选项
read_data.set_pd_option(max_show=True, float_type=True, decimal_places=4)

print(CT("----------读取数据----------").pink())
path = "input/history_xm.xlsx"
df_origin = read_data.read_df("input/history_xm.xlsx")

print(CT("----------增加新列----------").pink())
# view_percent这一列是以百分比形式展示的，需要转换为数值型数据
df_origin['view_percent'] = df_origin['view_percent'].str.rstrip('%').astype('float') / 100.0
# 将弹幕、评论、点赞、投币、收藏、分享这六列的数据转化为比例
df_origin['dm_rate'] = df_origin['dm'] / df_origin['view']
df_origin['reply_rate'] = df_origin['reply'] / df_origin['view']
df_origin['like_rate'] = df_origin['like'] / df_origin['view']
df_origin['coin_rate'] = df_origin['coin'] / df_origin['view']
df_origin['fav_rate'] = df_origin['fav'] / df_origin['view']
df_origin['share_rate'] = df_origin['share'] / df_origin['view']
# time这一列是时间戳，数值较大。这里简单处理：减去最小值，使时间戳从0开始
df_origin['time'] = df_origin['time'] - df_origin['time'].min()

print(CT("----------数据信息----------").pink())
ToMd.text_to_md(md_text="1. 数据信息", md_flag=True, md_color="pink", md_h=1)
# 按照u_score这一列的值来查看df_origin的情况
u_score_counts = df_origin['u_score'].value_counts().sort_index()
print(u_score_counts)

desc_df = read_data.desc_df(df_origin)
desc_df.show_df(head_n=0, tail_n=0, show_columns=False, show_dtypes=False, md_flag=True)
desc_df.describe_df(stats_detailed=False, md_flag=True)

print(CT("----------处理异常值与缺失值----------").pink())
ToMd.text_to_md(md_text="2. 处理异常值与缺失值", md_flag=True, md_color="pink", md_h=1)
# 首先处理特殊的情况。观察describe_df发现存在like_rate>1的情况，是BV12r421T73c，估计是她买量买错了，所以把一行删了
print(CT("like_rate>1的行：").red())
print(df_origin[df_origin['like_rate'] > 1])
df_origin = df_origin[df_origin['like_rate'] <= 1]  # 这个顺带把缺失值的行也删了
desc_df = read_data.desc_df(df_origin)
desc_df.describe_df(stats_detailed=False)  # 发现已经没有缺失值了
desc_df.process_outlier(method='IQR', show_info=True, process_type='ignore', md_flag=True)

print(CT("----------选择数值型数据----------").pink())
ToMd.text_to_md(md_text="3. 线性回归等机器学习算法", md_flag=True, md_color="pink", md_h=1)
df_num = df_origin.select_dtypes(include=['number'])
print(df_num.columns)
# 输出：['progress', 'duration', 'view_percent', 'view_time', 'u_like', 'u_coin', 'u_fav', 'u_score',
# 'view', 'dm', 'reply', 'time', 'like', 'coin', 'fav', 'share', 'tid', 'up_follow', 'up_followers',
# 'dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate']
feature_all = df_num.columns.tolist()
# 猜想feature_guess是下面这些
feature_guess = ['view_percent', 'view', 'dm', 'reply', 'time', 'like', 'coin', 'fav', 'share', 'tid', 'up_follow',
                 'up_followers', 'dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate']
# desc_df = read_data.desc_df(df_num[feature_all])
# desc_df.draw_heatmap(scale=True, v_minmax=(-5, 5))  # 画热力图，浅略查看特征之间的相关性
# desc_df = read_data.desc_df(df_num[feature_guess])
# desc_df.draw_heatmap(scale=True, v_minmax=(-5, 5))  # 画热力图，浅略查看特征之间的相关性

print(CT("----------数据处理(回归)[尝试线性回归]----------").pink())
ToMd.text_to_md(md_text="3.1 尝试一些属性(feature-guess)进行线性回归", md_flag=True, md_color="blue", md_h=2)
cal_df = cal_data.Linear(df_num[feature_guess + ['u_score']])  # 取feature_guess与u_score进行线性回归
cal_df.cal_linear(feature_guess, 'u_score', md_flag=True)
ToMd.df_to_md(cal_df.weight, md_flag=True, md_index=True)
ToMd.text_to_md(md_text="通过输出的结果（如VIF），可以看出view、like等这样的特征之间有很强的相关性，显然是偏好依附模型造成的。"
                        "下面使用随机森林来查看特征重要性", md_flag=True)

print(CT("----------数据处理(回归)[使用随机森林]----------").pink())
ToMd.text_to_md(md_text="3.2 尝试随机森林", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用全部数据训练", md_flag=True)
cal_df = cal_data.Tree(df_num[feature_guess + ['u_score']])
cal_df.cal_random_forest(feature_guess, 'u_score', pos_label=5, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
# 发现随机森林的准确率100%了，因此先划分训练集和测试集，再使用重要特征进行线性回归
ToMd.text_to_md(md_text="划分训练集与验证集训练", md_flag=True)
X_train, X_test, y_train, y_test = train_test_split(df_num[feature_guess], df_num['u_score'],
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_guess, 'u_score', pos_label=5, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(f"随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"随机森林准确率：{cal_df.tree.score(X_test, y_test)}", md_flag=True)


print(CT("----------数据处理(回归)[使用重要特征]----------").pink())
ToMd.text_to_md(md_text="3.3 尝试线性回归", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-RF[:10]", md_flag=True)
feature_RF = cal_df.feature_by_importance.tolist()
print(feature_RF)
cal_df = cal_data.Linear(df_num[feature_RF[:10] + ['u_score']])
cal_df.cal_linear(feature_RF[:10], 'u_score', md_flag=True)
print(cal_df.weight)
# 事实上feature_RF[:10]得出的结果并不是最优的，因为单独的RF的结果不一定能直接用于线性回归
# 可以尝试依据特征的方差、相关性等进行特征选择，或者使用其他的特征选择方法
# 下面是自主选择的特征
ToMd.text_to_md(md_text="使用feature-important[:10]", md_flag=True)
feature_important = ['dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate',
                     'up_follow', 'view_percent']
cal_df = cal_data.Linear(df_num[feature_important + ['u_score']])
cal_df.cal_logistic(feature_important, 'u_score', pos_label=5, md_flag=True)
print(cal_df.weight)


exit(111)
draw_df = draw_data.draw_df(df_num)
# draw_df.draw_corr(v_minmax=(-1, 1))
# 在df_num中去掉u_like, u_coin, u_fav这三列，因为这三列是计算u_score的依据
# feature_name_main = [col for col in df_num.columns if col not in ['u_like', 'u_coin', 'u_fav', 'u_score']]
# draw_df.draw_feature_importance(target_name='u_score', feature_name=feature_name_main,
#                                 descending_draw=True, print_top=10)
draw_df.draw_all_scatter(target_name='u_score', save_path='output/bilibili_RS/scatter_all')

