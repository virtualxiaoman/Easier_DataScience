# Hands On bilibili Recommend System 动手实现b站的推荐系统(非官方)
# 这是第二部分，主要是更进一步提升模型准确率，以及观察什么样的更改会对模型准确率产生影响

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

from easier_excel import read_data, cal_data, draw_data
from easier_tools.Colorful_Console import ColoredText as CT
from easier_tools.to_md import ToMd

ToMd.path = "output/bilibili_RS/Bili_RS_2.md"  # 更改输出路径
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

# 只保留是数值的列
desc_df_o = read_data.DescDF(df_origin.select_dtypes(include=['number']).copy())
print(desc_df_o.shape)  # (1111, 25)
print(df_origin['u_score'].value_counts())

desc_df_o.transform_df(target='u_score')
df_origin_minmax = desc_df_o.minmax_df  # 归一化
df_origin_minmax['u_score'] = df_origin['u_score']
df_origin_zscore = desc_df_o.zscore_df  # 标准化
df_origin_zscore['u_score'] = df_origin['u_score']
df_origin_smote = desc_df_o.smote_df  # SMOTE


desc_df_o.process_outlier(method='IQR', show_info=False, process_type='delete', IQR_Q1=0.2, IQR_Q3=0.95, IQR_k=3)
desc_df_o.delete_missing_values()
# print(desc_df_o.missing_info)
print(desc_df_o.shape)  # (984, 25)
print(df_origin.shape)  # (1111, 29)

df_deleteoutlier = desc_df_o.df.copy()
# temp = read_data.DescDF(df_deleteoutlier)
# temp.update_desc_df()
# print(temp.shape)  # (984, 25)
# print(temp.missing_info)
print(df_deleteoutlier['u_score'].value_counts())  # 0: 895, 1: 89
desc_df_d = read_data.DescDF(df_deleteoutlier)
desc_df_d.transform_df(target='u_score')
df_deleteoutlier_minmax = desc_df_d.minmax_df  # 归一化
df_deleteoutlier_minmax['u_score'] = df_deleteoutlier['u_score'].values
df_deleteoutlier_zscore = desc_df_d.zscore_df  # 标准化
df_deleteoutlier_zscore['u_score'] = df_deleteoutlier['u_score'].values
df_deleteoutlier_smote = desc_df_d.smote_df  # SMOTE

# temp = read_data.DescDF(df_deleteoutlier_minmax)
# temp.update_desc_df()
# print(temp.shape)  # (984, 25)
# print(temp.missing_info)
# print(df_deleteoutlier_minmax['u_score'].value_counts())  # 0: 770, 1: 83
# temp = read_data.DescDF(df_deleteoutlier_zscore)
# temp.update_desc_df()
# print(temp.shape)  # (984, 25)
# print(temp.missing_info)
# print(df_deleteoutlier_zscore['u_score'].value_counts())  # 0: 770, 1: 83

print(CT("----------数据处理----------").pink())
# 沿用part1的分析，
feature_main = ['progress', 'duration', 'view_percent', 'view_time', 'view', 'dm', 'reply', 'time', 'like', 'coin',
                'fav', 'share', 'tid', 'up_follow', 'up_followers', 'dm_rate',  'reply_rate', 'like_rate',
                'coin_rate', 'fav_rate', 'share_rate']
feature_guess = ['view_percent', 'view', 'dm', 'reply', 'time', 'like', 'coin', 'fav', 'share', 'tid', 'up_follow',
                 'up_followers', 'dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate']
feature_important = ['dm_rate',  'reply_rate', 'like_rate', 'coin_rate', 'fav_rate', 'share_rate',
                     'up_follow', 'view_percent']
target = ['u_score']

features = [feature_main, feature_guess, feature_important]
dfs = [df_origin, df_origin_minmax, df_origin_zscore, df_origin_smote,
       df_deleteoutlier, df_deleteoutlier_minmax, df_deleteoutlier_zscore]
# 首先寻找一种模型使得准确率较高
# 模型选择cal_data里的cal_logistic、cal_svc、cal_random_forest、cal_gbdtC
# 初始化一个7*4的表格，用于存放不同数据处理方式和不同模型下的准确率
df_acc = pd.DataFrame(columns=['origin', 'minmax', 'zscore', 'smote',
                               'deleteoutlier', 'deleteoutlier_minmax', 'deleteoutlier_zscore'],
                      index=['logistic', 'svc', 'random_forest', 'gbdtC'])
df_f1_weight = pd.DataFrame(columns=['origin', 'minmax', 'zscore', 'smote',
                                     'deleteoutlier', 'deleteoutlier_minmax', 'deleteoutlier_zscore'],
                            index=['logistic', 'svc', 'random_forest', 'gbdtC'])
df_f1_unweighted = pd.DataFrame(columns=['origin', 'minmax', 'zscore', 'smote',
                                         'deleteoutlier', 'deleteoutlier_minmax', 'deleteoutlier_zscore'],
                                index=['logistic', 'svc', 'random_forest', 'gbdtC'])

for j, feature in enumerate(features):
    # print(CT(f"----------数据处理{i+1}----------").pink())
    for i, df in enumerate(dfs):
        # print(CT(f"----------数据处理{i+1}，特征{j+1}----------").pink())
        cal_df = cal_data.Linear(df)
        cal_df.cal_logistic(feature, 'u_score', detailed=False)
        df_acc.loc['logistic', df_acc.columns[i]] = cal_df.ACC
        df_f1_weight.loc['logistic', df_f1_weight.columns[i]] = cal_df.F1_weight
        df_f1_unweighted.loc['logistic', df_f1_weight.columns[i]] = cal_df.F1_unweighted

        cal_df = cal_data.SVM(df)
        cal_df.cal_svc(feature, 'u_score')
        df_acc.loc['svc', df_acc.columns[i]] = cal_df.ACC
        df_f1_weight.loc['svc', df_f1_weight.columns[i]] = cal_df.F1_weight
        df_f1_unweighted.loc['svc', df_f1_weight.columns[i]] = cal_df.F1_unweighted

        cal_df = cal_data.Tree(df)
        cal_df.cal_random_forest(feature, 'u_score')
        df_acc.loc['random_forest', df_acc.columns[i]] = cal_df.ACC
        df_f1_weight.loc['random_forest', df_f1_weight.columns[i]] = cal_df.F1_weight
        df_f1_unweighted.loc['random_forest', df_f1_weight.columns[i]] = cal_df.F1_unweighted

        cal_df = cal_data.GBDT(df)
        cal_df.cal_gbdtC(feature, 'u_score')
        df_acc.loc['gbdtC', df_acc.columns[i]] = cal_df.ACC
        df_f1_weight.loc['gbdtC', df_f1_weight.columns[i]] = cal_df.F1_weight
        df_f1_unweighted.loc['gbdtC', df_f1_weight.columns[i]] = cal_df.F1_unweighted
    ToMd.text_to_md(f"特征集{j+1}：", md_flag=True, md_h=4)
    ToMd.text_to_md(f"{feature}", md_flag=True)
    ToMd.text_to_md("准确率表", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_acc, md_flag=True, md_index=True)
    ToMd.text_to_md("F1表(weight)", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_f1_weight, md_flag=True, md_index=True)
    ToMd.text_to_md("F1表(unweighted)", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_f1_unweighted, md_flag=True, md_index=True)

print(CT("----------准确率表----------").pink())
print(df_acc)  # 这是最后一次的，也就是feature_important
# ToMd.df_to_md(df_acc, md_flag=True, md_index=True)

print(CT("----------使用训练集和测试集----------").pink())
ToMd.text_to_md("使用训练集和测试集", md_flag=True, md_h=4)
ToMd.text_to_md("以下是测试集的输出信息", md_flag=True)
for j, feature in enumerate(features):
    for i, df in enumerate(dfs):
        X_train, X_test, y_train, y_test = train_test_split(df[feature], df['u_score'], test_size=0.2, random_state=42)
        train_dataset = X_train.copy()
        train_dataset['u_score'] = y_train
        test_dataset = X_test.copy()
        test_dataset['u_score'] = y_test

        cal_df = cal_data.Linear(train_dataset)
        cal_df.cal_logistic(feature, 'u_score', detailed=False)
        acc = cal_df.log_reg.score(X_test, y_test)
        y_pred = cal_df.log_reg.predict(X_test)
        f1_weight = f1_score(y_test, y_pred, average='weighted')
        f1_unweighted = f1_score(y_test, y_pred, average='macro')
        df_acc.loc['logistic', df_acc.columns[i]] = acc
        df_f1_weight.loc['logistic', df_f1_weight.columns[i]] = f1_weight
        df_f1_unweighted.loc['logistic', df_f1_weight.columns[i]] = f1_unweighted

        cal_df = cal_data.SVM(train_dataset)
        cal_df.cal_svc(feature, 'u_score')
        acc = cal_df.svc.score(X_test, y_test)
        y_pred = cal_df.svc.predict(X_test)
        f1_weight = f1_score(y_test, y_pred, average='weighted')
        f1_unweighted = f1_score(y_test, y_pred, average='macro')
        df_acc.loc['svc', df_acc.columns[i]] = acc
        df_f1_weight.loc['svc', df_f1_weight.columns[i]] = f1_weight
        df_f1_unweighted.loc['svc', df_f1_weight.columns[i]] = f1_unweighted

        cal_df = cal_data.Tree(train_dataset)
        cal_df.cal_random_forest(feature, 'u_score')
        acc = cal_df.tree.score(X_test, y_test)
        y_pred = cal_df.tree.predict(X_test)
        f1_weight = f1_score(y_test, y_pred, average='weighted')
        f1_unweighted = f1_score(y_test, y_pred, average='macro')
        df_acc.loc['random_forest', df_acc.columns[i]] = acc
        df_f1_weight.loc['random_forest', df_f1_weight.columns[i]] = f1_weight
        df_f1_unweighted.loc['random_forest', df_f1_weight.columns[i]] = f1_unweighted

        cal_df = cal_data.GBDT(train_dataset)
        cal_df.cal_gbdtC(feature, 'u_score')
        acc = cal_df.gbdt.score(X_test, y_test)
        y_pred = cal_df.gbdt.predict(X_test)
        f1_weight = f1_score(y_test, y_pred, average='weighted')
        f1_unweighted = f1_score(y_test, y_pred, average='macro')
        df_acc.loc['gbdtC', df_acc.columns[i]] = acc
        df_f1_weight.loc['gbdtC', df_f1_weight.columns[i]] = f1_weight
        df_f1_unweighted.loc['gbdtC', df_f1_weight.columns[i]] = f1_unweighted
    ToMd.text_to_md(f"特征集{j+1}：", md_flag=True, md_h=4)
    ToMd.text_to_md(f"{feature}", md_flag=True)
    ToMd.text_to_md("准确率表", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_acc, md_flag=True, md_index=True)
    ToMd.text_to_md("F1表(weight)", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_f1_weight, md_flag=True, md_index=True)
    ToMd.text_to_md("F1表(unweighted)", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_f1_unweighted, md_flag=True, md_index=True)


# 使用随机森林与feature_main，对df_origin，df_origin_zscore进行训练与交叉验证
print(CT("----------使用随机森林与feature_main进行训练与交叉验证----------").pink())

df_avg_score = pd.DataFrame(columns=['df_origin', 'df_origin_zscore', 'df_origin_smote'],
                            index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

for j, feature in enumerate(features):
    ToMd.text_to_md(f"特征集{j+1}：", md_flag=True, md_h=4)
    ToMd.text_to_md(f"{feature}", md_flag=True)
    for i, df in enumerate([df_origin, df_origin_zscore, df_origin_smote]):
        X_train, X_test, y_train, y_test = train_test_split(df[feature], df['u_score'], test_size=0.2, random_state=42)
        train_dataset = X_train.copy()
        train_dataset['u_score'] = y_train
        test_dataset = X_test.copy()
        test_dataset['u_score'] = y_test
        cal_df = cal_data.Tree(train_dataset)
        cal_df.cal_random_forest(feature, 'u_score')
        f1_unweighted = f1_score(y_test, cal_df.tree.predict(X_test), average='macro')
        print(f"df_origin: {f1_unweighted:.4f}")

        cal_cv = cal_data.CrossValidation(train_dataset)
        for score in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            cal_cv.cal_cross_validation(cal_df.tree, feature, 'u_score', cv=5, scoring=score)
            df_avg_score.loc[score, df_avg_score.columns[i]] = cal_cv.avg_score
    ToMd.text_to_md("准确率表：", md_flag=True, md_bold=True)
    ToMd.df_to_md(df_avg_score, md_flag=True, md_index=True)


# 检验smote的效果
# 将原始数据集划分为两部分，一部分用于训练，一部分用于测试。训练的数据进行smote，但是测试数据不进行smote
print(CT("----------检验smote的效果----------").pink())
# 这里为了简便，只使用feature_main，且只使用df_origin
X_train, X_test, y_train, y_test = train_test_split(df_origin[feature_main], df_origin['u_score'],
                                                    test_size=0.4, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test

des_df = read_data.DescDF(train_dataset)
des_df.transform_df(target='u_score')
train_dataset_smote = des_df.smote_df
train_dataset_adasyn = des_df.adasyn_df
train_dataset_smoteenn = des_df.smoteenn_df
train_dataset_smotetomek = des_df.smotetomek_df

for i, train_dataset in enumerate([train_dataset_smote, train_dataset_adasyn,
                                   train_dataset_smoteenn, train_dataset_smotetomek]):
    print(CT(f"----------使用第{i+1}种处理类别不平衡问题的方法----------").pink())
    cal_df = cal_data.Tree(train_dataset)
    cal_df.cal_random_forest(feature_main, 'u_score')
    acc = cal_df.tree.score(X_test, y_test)
    f1_unweighted = f1_score(y_test, cal_df.tree.predict(X_test), average='macro')

    print(f"accuracy: {acc:.4f}")
    print(f"f1_unweighted: {f1_unweighted:.4f}")
    print(classification_report(y_test, cal_df.tree.predict(X_test)))
    print(confusion_matrix(y_test, cal_df.tree.predict(X_test)))


print(CT("----------使用class_weight与VotingClassifier----------").pink())
X_train, X_test, y_train, y_test = train_test_split(df_origin[feature_main], df_origin['u_score'],
                                                    test_size=0.4, random_state=42)
train_dataset = X_train.copy()
train_dataset['u_score'] = y_train
test_dataset = X_test.copy()
test_dataset['u_score'] = y_test

rf = RandomForestClassifier(class_weight={0: 1, 1: 100}, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))


log_reg = LogisticRegression(class_weight={0: 1, 1: 100}, random_state=42, max_iter=10000)
rf = RandomForestClassifier(class_weight={0: 1, 1: 100}, random_state=42)
svc = SVC(class_weight={0: 1, 1: 100}, probability=True, random_state=42, max_iter=10000)

voting_clf = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('rf', rf),
    ('svc', svc)], voting='soft')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# 检查数据的分布，使用t-SNE、PCA算法进行降维到二维，然后绘制散点图
print(CT("----------检查数据的分布----------").pink())
cal_DR = cal_data.DimReduction(df_origin)
# 创建1行2列的图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
cal_DR.cal_pca(feature_main, 'u_score', n_components=2, draw_DR=True, ax=axes[0], show_plt=False)
cal_DR.cal_tsne(feature_main, 'u_score', n_components=2, draw_DR=True, ax=axes[1], show_plt=False)
plt.show()
plt.close()

# 聚类
print(CT("----------聚类----------").pink())
print(CT("----------feature_main----------").pink())
cal_cluster = cal_data.Cluster(df_origin)
cal_cluster.cal_kmeans(feature_main, 'u_score', n_clusters=2, draw_cluster=False)
cal_cluster.cal_agg(feature_main, 'u_score', n_clusters=2, draw_cluster=False)
cal_cluster.cal_dbscan(feature_main, 'u_score', eps=0.5, min_samples=5, draw_cluster=False)
print(CT("----------二维(绘图)----------").pink())
cal_cluster = cal_data.Cluster(df_origin[["coin_rate", "fav_rate", "u_score"]])
cal_cluster.cal_kmeans(["coin_rate", "fav_rate"], 'u_score', n_clusters=2, draw_cluster=True)
cal_cluster.cal_agg(["coin_rate", "fav_rate"], 'u_score', n_clusters=2, draw_cluster=True)
cal_cluster.cal_dbscan(["coin_rate", "fav_rate"], 'u_score', eps=0.5, min_samples=5, draw_cluster=True)





