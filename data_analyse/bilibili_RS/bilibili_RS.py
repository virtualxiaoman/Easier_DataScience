# Hands On bilibili Recommend System 动手实现b站的推荐系统(非官方)
# 这是第一部分，主要是研究数据的一些基础特征，以及尝试一些简单的机器学习算法，看看准确率什么的
# 这部分可能比较乱，建议看文档

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

from easier_excel import read_data, cal_data, draw_data
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.to_md import ToMd
ToMd.path = "../output/bilibili_RS/Bili_RS.md"  # 更改输出路径
ToMd = ToMd()
ToMd.update_path()  # 这是更改path需要做的必要更新
ToMd.clear_md()  # 清空前如有需要，务必备份，默认不清空

model_path = "../output/bilibili_RS/model"

# 设置pandas显示选项
read_data.set_pd_option(max_show=True, float_type=True, decimal_places=4)

print(CT("----------读取数据----------").pink())
path = "../input/history_xm.xlsx"
df_origin = read_data.read_df("../input/history_xm.xlsx")

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

desc_df = read_data.DescDF(df_origin)
desc_df.show_df(head_n=0, tail_n=0, show_columns=False, show_dtypes=False, md_flag=True)
desc_df.describe_df(stats_detailed=False, md_flag=True)

print(CT("----------处理异常值与缺失值----------").pink())
ToMd.text_to_md(md_text="2. 处理异常值与缺失值", md_flag=True, md_color="pink", md_h=1)
# 首先处理特殊的情况。观察describe_df发现存在like_rate>1的情况，是BV12r421T73c，估计是她买量买错了，所以把一行删了
print(CT("like_rate>1的行：").red())
print(df_origin[df_origin['like_rate'] > 1])
df_origin = df_origin[df_origin['like_rate'] <= 1]  # 这个顺带把缺失值的行也删了
desc_df = read_data.DescDF(df_origin)
desc_df.describe_df(stats_detailed=False)  # 发现已经没有缺失值了
desc_df.process_outlier(method='IQR', show_info=True, process_type='ignore', md_flag=True)  # 暂不处理异常值，因为这里都是真实数据
desc_df.show_df(head_n=0, tail_n=0, show_columns=False, show_dtypes=False, md_flag=True)  # 该行只是为了写入到md文件
desc_df.describe_df(stats_detailed=False, show_stats=False, show_nan=False, md_flag=True)  # 该行只是为了写入到md文件

print(CT("----------选择数值型数据----------").pink())
ToMd.text_to_md(md_text="3. 线性回归等机器学习算法", md_flag=True, md_color="pink", md_h=1)
df_num = df_origin.select_dtypes(include=['number'])
print(df_num.columns)
# 输出：['progress', 'duration', 'view_percent', 'view_time', 'u_like', 'u_coin', 'u_fav', 'u_score',
# 'view', 'dm', 'reply', 'time', 'like', 'coin', 'fav', 'share', 'tid', 'up_follow', 'up_followers',
# 'dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate']
feature_all = df_num.columns.tolist()
# 在feature_all中，删除'u_like', 'u_coin', 'u_fav'，因为这三列构成了u_score
feature_main = ['progress', 'duration', 'view_percent', 'view_time', 'view', 'dm', 'reply', 'time', 'like', 'coin',
                'fav', 'share', 'tid', 'up_follow', 'up_followers', 'dm_rate',  'reply_rate', 'like_rate',
                'coin_rate', 'fav_rate', 'share_rate']
# 猜想feature_guess是下面这些
feature_guess = ['view_percent', 'view', 'dm', 'reply', 'time', 'like', 'coin', 'fav', 'share', 'tid', 'up_follow',
                 'up_followers', 'dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate']
# 猜想重要特征feature_important是下面这些
feature_important = ['dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate',
                     'up_follow', 'view_percent']
# DescDF = read_data.DescDF(df_num[feature_all])
# DescDF.draw_heatmap(scale=True, v_minmax=(-5, 5))  # 画热力图，浅略查看特征之间的相关性
# DescDF = read_data.DescDF(df_num[feature_guess])
# DescDF.draw_heatmap(scale=True, v_minmax=(-5, 5))  # 画热力图，浅略查看特征之间的相关性

print(CT("----------数据处理(回归)[尝试线性回归]----------").pink())
ToMd.text_to_md(md_text="3.1 尝试一些属性(feature-guess)进行线性回归", md_flag=True, md_color="blue", md_h=2)
cal_df = cal_data.Linear(df_num[feature_guess + ['u_score']])  # 取feature_guess与u_score进行线性回归
cal_df.cal_linear(feature_guess, 'u_score', md_flag=True)
ToMd.df_to_md(cal_df.weight, md_flag=True, md_index=True)
ToMd.text_to_md(md_text="通过输出的结果（如VIF），可以看出view、like等这样的特征之间有很强的相关性，显然是偏好依附模型造成的。"
                        "下面使用随机森林来查看特征重要性:", md_flag=True)

print(CT("----------数据处理(回归)[使用随机森林]----------").pink())
ToMd.text_to_md(md_text="3.2 尝试随机森林", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用全部数据训练，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(df_num[feature_guess + ['u_score']])
cal_df.cal_random_forest(feature_guess, 'u_score', pos_label=5, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
# 发现随机森林的准确率100%了，因此先划分训练集和测试集，再使用重要特征进行线性回归
ToMd.text_to_md(md_text="划分训练集与验证集训练，模型RF", md_flag=True, md_bold=True, md_h=3)
X_train, X_test, y_train, y_test = train_test_split(df_num[feature_guess], df_num['u_score'],
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_guess, 'u_score', pos_label=5, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}", md_flag=True, md_color="orange")


print(CT("----------数据处理(回归)[使用重要特征]----------").pink())
ToMd.text_to_md(md_text="3.3 尝试线性回归", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-RF[:10]，模型linear", md_flag=True, md_h=3)
feature_RF = cal_df.feature_by_importance
print(feature_RF)
cal_df = cal_data.Linear(df_num[feature_RF[:10] + ['u_score']])
cal_df.cal_linear(feature_RF[:10], 'u_score', md_flag=True, md_h=3)
print(cal_df.weight)
# 事实上feature_RF[:10]得出的结果并不是最优的，因为单独的RF的结果不一定能直接用于线性回归
# 可以尝试依据特征的方差、相关性等进行特征选择，或者使用其他的特征选择方法
# feature_important是自主选择的特征
ToMd.text_to_md(md_text="使用feature-important，模型logistic", md_flag=True, md_h=3)
cal_df = cal_data.Linear(df_num[feature_important + ['u_score']])
cal_df.cal_logistic(feature_important, 'u_score', pos_label=5, md_flag=True)
print(cal_df.weight)

print(CT("----------数据处理(GBDT)----------").pink())
ToMd.text_to_md(md_text="3.4 尝试GBDT", md_flag=True, md_color="blue", md_h=2)
cal_df = cal_data.GBDT(df_num[feature_guess + ['u_score']])
cal_df.cal_gbdtC(feature_guess, 'u_score', pos_label=5, md_flag=True)
print(cal_df.feature_by_importance)
# 选择GBDT的前10个特征进行新的GBDT
feature_GBDT = cal_df.feature_by_importance
ToMd.text_to_md(md_text="使用feature-GBDT[:10]，模型GBDT", md_flag=True, md_h=3)
cal_df = cal_data.GBDT(df_num[feature_GBDT[:10] + ['u_score']])
cal_df.cal_gbdtC(feature_GBDT[:10], 'u_score', pos_label=5, md_flag=True)
# 下面是自主选择的特征
ToMd.text_to_md(md_text="使用feature-important，模型GBDT", md_flag=True, md_h=3)
cal_df = cal_data.GBDT(df_num[feature_important + ['u_score']])
cal_df.cal_gbdtC(feature_important, 'u_score', pos_label=5, md_flag=True)
print(cal_df.feature_by_importance)
# 划分训练集和测试集，再使用feature_guess进行GBDT
ToMd.text_to_md(md_text="划分训练集与验证集训练，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(train_dataset)
cal_df.cal_gbdtC(feature_guess, 'u_score', pos_label=5, md_flag=True,
                 n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}", md_flag=True, md_color="orange")

# 观察上述结果，发现基本都是过拟合了，因此需要进一步处理
print(CT("----------[过拟合处理]----------").pink())
ToMd.text_to_md(md_text="4. 过拟合处理", md_flag=True, md_color="pink", md_h=1)
# 首先，推荐系统可以简单地认为要么用户看，要么用户不看，因此可以将u_score二值化，u_score>=3的为1，否则为0
df_num['u_score'] = df_num['u_score'].apply(lambda x: 1 if x >= 3 else 0)
# 然后，再次使用RF与GBDT，看看效果
ToMd.text_to_md(md_text="4.1 二值化u-score后再次使用RF", md_flag=True, md_color="blue", md_h=2)
# 这里先使用feature_guess，再使用feature_important
ToMd.text_to_md(md_text="使用feature-guess，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(df_num[feature_guess + ['u_score']])
cal_df.cal_random_forest(feature_guess, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(cal_df.feature_by_importance)
ToMd.text_to_md(md_text="使用feature-important，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(df_num[feature_important + ['u_score']])
cal_df.cal_random_forest(feature_important, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(cal_df.feature_by_importance)
# 划分训练集和测试集，再使用feature_important进行RF
ToMd.text_to_md(md_text="划分训练集与验证集训练，模型RF", md_flag=True, md_bold=True, md_h=3)
X_train, X_test, y_train, y_test = train_test_split(df_num[feature_important], df_num['u_score'],
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_important, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}", md_flag=True, md_color="orange")

ToMd.text_to_md(md_text="4.2 二值化u-score后再次使用GBDT", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-guess，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(df_num[feature_guess + ['u_score']])
cal_df.cal_gbdtC(feature_guess, 'u_score', pos_label=1, md_flag=True)
print(cal_df.feature_by_importance)
ToMd.text_to_md(md_text="使用feature-important，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(df_num[feature_important + ['u_score']])
cal_df.cal_gbdtC(feature_important, 'u_score', pos_label=1, md_flag=True)
print(cal_df.feature_by_importance)
# 划分训练集和测试集，再使用feature_important进行GBDT
ToMd.text_to_md(md_text="划分训练集与验证集训练，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(train_dataset)
cal_df.cal_gbdtC(feature_important, 'u_score', pos_label=1, md_flag=True)
print(f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}", md_flag=True, md_color="orange")
# 查看验证集中预测为1实际为0与预测为0实际为1的情况
df_temp_1to0 = df_origin.loc[test_dataset[(cal_df.gbdt.predict(X_test) == 1) & (y_test == 0)].index]
df_temp_0to1 = df_origin.loc[test_dataset[(cal_df.gbdt.predict(X_test) == 0) & (y_test == 1)].index]
ToMd.text_to_md(md_text="gbdt模型预测为1实际为0的情况,对应的原始数据：", md_flag=True, md_color="red")
ToMd.df_to_md(df_temp_1to0, md_flag=True, md_index=True)
ToMd.text_to_md(md_text="gbdt模型预测为0实际为1的情况,对应的原始数据：", md_flag=True, md_color="red")
ToMd.df_to_md(df_temp_0to1, md_flag=True, md_index=True)

ToMd.text_to_md(md_text="4.3 二值化u-score后再次使用逻辑回归", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important，模型logistic", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Linear(df_num[feature_important + ['u_score']])
cal_df.cal_logistic(feature_important, 'u_score', pos_label=1, md_flag=True)
print(cal_df.weight)
# 注意到模型把1全部预测为了0，所以需要进一步处理，可以尝试重采样
# 重采样的方法有很多，这里使用SMOTE

print(CT("----------[重采样]----------").pink())
ToMd.text_to_md(md_text="5. 重采样", md_flag=True, md_color="pink", md_h=1)
ToMd.text_to_md(md_text="使用SMOTE", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text=r"公式 $ x_{\text {new }}=x_{\text {original }}+\lambda \times\left(x_{\text {neighbor }}-x_{\text {original }}\right) $", md_flag=True)
# 首先我们还是使用df_num的全部特征
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_num[feature_main], df_num['u_score'])
print(CT("重采样后的数据：").red())
print(y_resampled.value_counts())
# 划分训练集和测试集，再使用feature_main进行RF看看特征重要性
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test
ToMd.text_to_md(md_text="5.0 重采样后，使用全部特征再次使用RF", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature_main，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_main, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(cal_df.feature_by_importance)
print(f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}",
                md_flag=True, md_color="orange")
# 保存模型，事实上，这个效果是最好的，原因可能是feature_main中包含了所有主要的特征
with open(f"{model_path}/RF_model_all.pkl", 'wb') as file:
    pickle.dump(cal_df.tree, file)
# 根据特征重要性，可以发现view_time、view_percent、progress、duration等特征对于u_score的影响不大，况且这是看了才会有的数据。
# 另外up_follow、dm_rate、reply_rate也不是很重要，可能是因为这些不一定是正反馈
# 其中fav_rate、coin_rate、like_rate、up_followers、view_time、time、coin、tid、fav、share_rate的重要性较高，
# 因为view_time、time其实没有太多实际意义（因为视频发布的时间不是用户的选择），所以只在原来的基础上加上up_followers、coin、fav
# 因为tid可以在频繁项集等算法中使用，因此这里不考虑
feature_important_after_Smote = ['fav_rate', 'coin_rate', 'like_rate', 'up_followers', 'coin', 'fav', 'share_rate']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_num[feature_important], df_num['u_score'])
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
print(CT("重采样后的数据：").pink())
print(y_resampled.value_counts())
# 划分训练集和测试集，再使用feature_important进行RF，GBDT，逻辑回归
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test
ToMd.text_to_md(md_text="5.1 重采样后再次使用RF", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_important, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(cal_df.feature_by_importance)
print(f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}", md_flag=True, md_color="orange")
# # 查看验证集中预测为1实际为0与预测为0实际为1的情况
# df_temp_1to0 = df_resampled.loc[test_dataset[(cal_df.tree.predict(X_test) == 1) & (y_test == 0)].index]
# df_temp_0to1 = df_resampled.loc[test_dataset[(cal_df.tree.predict(X_test) == 0) & (y_test == 1)].index]
# ToMd.text_to_md(md_text="rf模型预测为1实际为0的情况,对应的原始数据：", md_flag=True, md_color="red")
# ToMd.df_to_md(df_temp_1to0, md_flag=True, md_index=True)
# ToMd.text_to_md(md_text="rf模型预测为0实际为1的情况,对应的原始数据：", md_flag=True, md_color="red")
# ToMd.df_to_md(df_temp_0to1, md_flag=True, md_index=True)
# 保存模型
with open(f"{model_path}/RF_model_feature_important.pkl", 'wb') as file:
    pickle.dump(cal_df.tree, file)

ToMd.text_to_md(md_text="5.2 重采样后再次使用GBDT", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(train_dataset)
cal_df.cal_gbdtC(feature_important, 'u_score', pos_label=1, md_flag=True)
print(cal_df.feature_by_importance)
print(f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}", md_flag=True, md_color="orange")
# # 查看验证集中预测为1实际为0与预测为0实际为1的情况
# df_temp_1to0 = df_resampled.loc[test_dataset[(cal_df.gbdt.predict(X_test) == 1) & (y_test == 0)].index]
# df_temp_0to1 = df_resampled.loc[test_dataset[(cal_df.gbdt.predict(X_test) == 0) & (y_test == 1)].index]
# ToMd.text_to_md(md_text="gbdt模型预测为1实际为0的情况,对应的原始数据：", md_flag=True, md_color="red")
# ToMd.df_to_md(df_temp_1to0, md_flag=True, md_index=True)
# ToMd.text_to_md(md_text="gbdt模型预测为0实际为1的情况,对应的原始数据：", md_flag=True, md_color="red")
# ToMd.df_to_md(df_temp_0to1, md_flag=True, md_index=True)
# 保存模型
with open(f"{model_path}/GBDT_model_feature_important.pkl", 'wb') as file:
    pickle.dump(cal_df.gbdt, file)

ToMd.text_to_md(md_text="5.3 重采样后再次使用逻辑回归", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important，模型logistic", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Linear(train_dataset)
cal_df.cal_logistic(feature_important, 'u_score', pos_label=1, md_flag=True)
print(cal_df.weight)
print(f"划分训练集与验证集训练的逻辑回归准确率：{cal_df.log_reg.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的逻辑回归准确率：{cal_df.log_reg.score(X_test, y_test)}", md_flag=True, md_color="orange")

ToMd.text_to_md(md_text="5.4 重采样后再次使用SVM", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important，模型SVM", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.SVM(train_dataset)
cal_df.cal_svc(feature_important, 'u_score', md_flag=True)
print(f"划分训练集与验证集训练的SVM准确率：{cal_df.svc.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的SVM准确率：{cal_df.svc.score(X_test, y_test)}", md_flag=True, md_color="orange")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_num[feature_important_after_Smote], df_num['u_score'])
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
print(CT("重采样后的数据：").pink())
print(y_resampled.value_counts())
# 划分训练集和测试集，再使用feature_important_after_Smote进行RF，GBDT，逻辑回归，SVM
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test

ToMd.text_to_md(md_text="5.5 RF，但是改为feature-important-after-Smote", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important-after-Smote，模型RF", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.Tree(train_dataset)
cal_df.cal_random_forest(feature_important_after_Smote, 'u_score', pos_label=1, md_flag=True,
                         n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=3)
print(cal_df.feature_by_importance)
print(f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的随机森林准确率：{cal_df.tree.score(X_test, y_test)}", md_flag=True, md_color="orange")
# 保存模型
with open(f"{model_path}/RF_model_feature_important_after_Smote.pkl", 'wb') as file:
    pickle.dump(cal_df.tree, file)

ToMd.text_to_md(md_text="5.6 GBDT，但是改为feature-important-after-Smote", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important-after-Smote，模型GBDT", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.GBDT(train_dataset)
cal_df.cal_gbdtC(feature_important_after_Smote, 'u_score', pos_label=1, md_flag=True)
print(cal_df.feature_by_importance)
print(f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的gbdt准确率：{cal_df.gbdt.score(X_test, y_test)}", md_flag=True, md_color="orange")
# 保存模型
with open(f"{model_path}/GBDT_model_feature_important_after_Smote.pkl", 'wb') as file:
    pickle.dump(cal_df.gbdt, file)

ToMd.text_to_md(md_text="5.7 SVM，但是改为feature-important-after-Smote", md_flag=True, md_color="blue", md_h=2)
ToMd.text_to_md(md_text="使用feature-important-after-Smote，模型SVM", md_flag=True, md_bold=True, md_h=3)
cal_df = cal_data.SVM(train_dataset)
cal_df.cal_svc(feature_important_after_Smote, 'u_score', md_flag=True)
print(f"划分训练集与验证集训练的SVM准确率：{cal_df.svc.score(X_test, y_test)}")
ToMd.text_to_md(md_text=f"划分训练集与验证集训练的SVM准确率：{cal_df.svc.score(X_test, y_test)}", md_flag=True, md_color="orange")
# 保存模型
with open(f"{model_path}/SVM_model_feature_important_after_Smote.pkl", 'wb') as file:
    pickle.dump(cal_df.svc, file)



exit(111)
draw_df = draw_data.DrawDF(df_num)
# DrawDF.draw_corr(v_minmax=(-1, 1))
# 在df_num中去掉u_like, u_coin, u_fav这三列，因为这三列是计算u_score的依据
# feature_name_main = [col for col in df_num.columns if col not in ['u_like', 'u_coin', 'u_fav', 'u_score']]
# DrawDF.draw_feature_importance(target_name='u_score', feature_name=feature_name_main,
#                                 descending_draw=True, print_top=10)
draw_df.draw_all_scatter(target_name='u_score', save_path='../output/bilibili_RS/scatter_all')

