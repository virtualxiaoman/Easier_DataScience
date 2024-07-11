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

