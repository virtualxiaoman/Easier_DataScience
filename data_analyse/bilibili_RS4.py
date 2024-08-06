# Hands On bilibili Recommend System 动手实现b站的推荐系统(非官方)
# 这是第四部分，主要是使用关联规则

import pandas as pd

from collections import Counter
import itertools
import ast
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from easier_excel import read_data, cal_data, draw_data
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.to_md import ToMd

ToMd.path = "output/bilibili_RS/Bili_RS_4.md"  # 更改输出路径
ToMd = ToMd()
ToMd.update_path()  # 这是更改path需要做的必要更新
ToMd.clear_md()  # 清空前如有需要，务必备份

model_path = "output/bilibili_RS/model"

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

print(CT("----------数据处理----------").pink())
df_origin = df_origin[df_origin['like_rate'] <= 1]   # 顺带删除缺失值
# 将u_score二值化，u_score>=3的为1，否则为0
df_origin['u_score'] = df_origin['u_score'].apply(lambda x: 1 if x >= 3 else 0)
df_origin['u_score'] = df_origin['u_score'].astype('int')
df_origin['up_follow'] = df_origin['up_follow'].astype('int')
df_origin['tid'] = df_origin['tid'].astype('int')

desc_df = read_data.DescDF(df_origin)
desc_df.draw_hist()
exit(11)
print(CT("----------统计----------").pink())
vc_uscore = df_origin['u_score'].value_counts().reset_index()
vc_uscore.columns = ['u_score', 'count']
vc_upname = df_origin['up_name'].value_counts().reset_index()
vc_upname.columns = ['up_name', 'count']
vc_upname = vc_upname[vc_upname['count'] > 1]  # 只保留出现次数大于1的
vc_tid = df_origin['tid'].value_counts().reset_index()
vc_tid.columns = ['tid', 'count']
vc_tid['tid'] = vc_tid['tid'].astype('int')  # 因为tid这一列其实是int，但是存储的是float，所以需要转换
vc_upfollow = df_origin['up_follow'].value_counts().reset_index()
vc_upfollow.columns = ['up_follow', 'count']
# 将每个样本的tag合并成一个列表
tag_list = []
for tags in df_origin['tag']:
    tags_list = ast.literal_eval(tags)  # 将字符串转换为列表
    for tag in tags_list:
        tag_list.append(tag)
tag_counter = Counter(tag_list)
vc_tags = pd.DataFrame(list(tag_counter.items()), columns=['tag', 'count'])
vc_tags = vc_tags.sort_values(by='count', ascending=False)
vc_tags = vc_tags[vc_tags['count'] > 3]  # 只保留出现次数大于3的

ToMd.text_to_md("标签统计", md_flag=True, md_h=1)
ToMd.text_to_md("u_score统计", md_flag=True, md_h=2)
ToMd.df_to_md(vc_uscore, md_flag=True)
ToMd.text_to_md("up_name统计(>1)", md_flag=True, md_h=2)
ToMd.df_to_md(vc_upname, md_flag=True)
ToMd.text_to_md("tid统计", md_flag=True, md_h=2)
ToMd.df_to_md(vc_tid, md_flag=True)
ToMd.text_to_md("up_follow统计", md_flag=True, md_h=2)
ToMd.df_to_md(vc_upfollow, md_flag=True)
ToMd.text_to_md("tag统计(>3)", md_flag=True, md_h=2)
ToMd.df_to_md(vc_tags, md_flag=True)

print(CT("----------关联规则----------").pink())
print(df_origin.head())
# 首先将数据转换为适合关联规则挖掘的形式
# 将每个样本的u_score变成"u_score1"和"u_score0"这两个值，u_score1表示原来的u_score等于1，u_score0表示原来的u_score等于0
# 将每个样本的tag变成list，然后提取出每个tag
# 将每个样本的up_follow变成"up_follow1"和"up_follow0"这两个值，up_follow1表示原来的up_follow等于1，up_follow0表示原来的up_follow等于0
# 将每个样本的tid变成"tid_"加上原来的tid的形式
df_rule = pd.DataFrame()
# get_dummies
df_uscore = pd.get_dummies(df_origin['u_score'], prefix='u_score')
df_upfollow = pd.get_dummies(df_origin['up_follow'], prefix='up_follow')
df_tid = pd.get_dummies(df_origin['tid'], prefix='tid')

tag_list = []
for tags in df_origin['tag']:
    tags_list = ast.literal_eval(tags)  # 将df_origin['tag'](字符串类型)转换为列表
    for tag in tags_list:
        tag_list.append(tag)
tag_counter = Counter(tag_list)
tag_list = [tag for tag, count in tag_counter.items() if count > 3]  # 防止太多了后续计算太慢，只保留出现次数大于3的
tag_list = list(set(tag_list))
# df_tag的列是tag_list，对于df_tag每一行的每个tag，去查找df_origin['tag']中是否包含该tag，如果包含则为该tag1，否则为0
df_tag = pd.DataFrame(columns=tag_list)
# 遍历df_origin的每一行，检查每个标签是否出现在该行的'tag'列中，并将结果存储在df_tag中
for index, row in df_origin.iterrows():
    tags = ast.literal_eval(row['tag'])
    for tag in tag_list:
        if tag in tags:
            df_tag.loc[index, tag] = True
        else:
            df_tag.loc[index, tag] = False

print(df_tag.head())
# 查看df_tag第一行为1的列
print(df_tag.loc[0, df_tag.loc[0] == 1])
df_rule = pd.concat([df_uscore, df_upfollow, df_tid, df_tag], axis=1)
print(df_rule.head())

# 保存df_rule以便后续使用
df_rule.to_excel("output/bilibili_RS/model/df_rule.xlsx", index=False)

df_rule = pd.read_excel("output/bilibili_RS/model/df_rule.xlsx")
# print(df_rule.head())
# 使用mlxtend库进行关联规则挖掘

# 挖掘频繁项集
frequent_itemsets = apriori(df_rule, min_support=0.05, use_colnames=True)
# 根据频繁项集计算关联规则
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values(by='lift', ascending=False)
# print(rules)
# 保存关联规则
rules.to_excel("output/bilibili_RS/model/apriori_rules.xlsx", index=False)
ToMd.text_to_md("关联规则", md_flag=True, md_h=1)
ToMd.df_to_md(rules, md_flag=True)

# 只保留antecedents和consequents中包含'u_score_1'的规则
df_u_score_1 = rules[rules['antecedents'].apply(lambda x: 'u_score_1' in x) | rules['consequents'].apply(lambda x: 'u_score_1' in x)]
df_u_score_1 = df_u_score_1.sort_values(by='lift', ascending=False)
# 因为得到的frozenset({'u_score_1', 'up_follow_0'})这种形式不太好看，所以将其转换为字符串
df_u_score_1['antecedents'] = df_u_score_1['antecedents'].apply(lambda x: list(x))
df_u_score_1['consequents'] = df_u_score_1['consequents'].apply(lambda x: list(x))
# 去除列表的[]
df_u_score_1['antecedents'] = df_u_score_1['antecedents'].apply(lambda x: str(x)[1:-1])
df_u_score_1['consequents'] = df_u_score_1['consequents'].apply(lambda x: str(x)[1:-1])
# 去除引号
df_u_score_1['antecedents'] = df_u_score_1['antecedents'].apply(lambda x: x.replace("'", ""))
df_u_score_1['consequents'] = df_u_score_1['consequents'].apply(lambda x: x.replace("'", ""))

print(df_u_score_1)
df_u_score_1.to_excel("output/bilibili_RS/model/apriori_rules_u_score_1.xlsx", index=False)
ToMd.text_to_md("关联规则(u_score_1)", md_flag=True, md_h=2)
ToMd.df_to_md(df_u_score_1, md_flag=True)

# 只保留antecedents和consequents中包含'u_score_0'的规则
df_u_score_0 = rules[rules['antecedents'].apply(lambda x: 'u_score_0' in x) | rules['consequents'].apply(lambda x: 'u_score_0' in x)]
df_u_score_0 = df_u_score_0.sort_values(by='lift', ascending=False)
# 因为得到的frozenset({'u_score_1', 'up_follow_0'})这种形式不太好看，所以将其转换为字符串
df_u_score_0['antecedents'] = df_u_score_0['antecedents'].apply(lambda x: list(x))
df_u_score_0['consequents'] = df_u_score_0['consequents'].apply(lambda x: list(x))
# 去除列表的[]
df_u_score_0['antecedents'] = df_u_score_0['antecedents'].apply(lambda x: str(x)[1:-1])
df_u_score_0['consequents'] = df_u_score_0['consequents'].apply(lambda x: str(x)[1:-1])
# 去除引号
df_u_score_0['antecedents'] = df_u_score_0['antecedents'].apply(lambda x: x.replace("'", ""))
df_u_score_0['consequents'] = df_u_score_0['consequents'].apply(lambda x: x.replace("'", ""))
print(df_u_score_0)
df_u_score_0.to_excel("output/bilibili_RS/model/apriori_rules_u_score_0.xlsx", index=False)
ToMd.text_to_md("关联规则(u_score_0)", md_flag=True, md_h=2)
ToMd.df_to_md(df_u_score_0, md_flag=True)

