# %% [markdown]
# # 特征工程

# %% [markdown]
# ## 数据加载

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
warnings.filterwarnings('ignore')


# %%
def time_transform(df):
    df['date'] = pd.to_datetime(df['time_stamp'], unit='s')
    df['date_ymd'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

    return df


ad_user_sample_data = pd.read_pickle('data/final_data/ad_user_sample_data.pkl')
ad_user_sample_data['date_ymd'] = ad_user_sample_data['date'].dt.date
ad_data = pd.read_pickle('data/final_data/ad_data.pkl')
user_data = pd.read_pickle('data/final_data/user_data.pkl')
user_behavior_data_pv = pd.read_pickle('data/final_data/user_behavior_data_pv.pkl')
user_behavior_data_cart = pd.read_pickle('data/final_data/user_behavior_data_cart.pkl')
user_behavior_data_fav = pd.read_pickle('data/final_data/user_behavior_data_fav.pkl')
user_behavior_data_buy = pd.read_pickle('data/final_data/user_behavior_data_buy.pkl')

user_behavior_data_pv = user_behavior_data_pv.rename({'user': 'userid', 'cate': 'cate_id'}, axis=1)
user_behavior_data_cart = user_behavior_data_cart.rename({'user': 'userid', 'cate': 'cate_id'}, axis=1)
user_behavior_data_fav = user_behavior_data_fav.rename({'user': 'userid', 'cate': 'cate_id'}, axis=1)
user_behavior_data_buy = user_behavior_data_buy.rename({'user': 'userid', 'cate': 'cate_id'}, axis=1)

user_behavior_data_pv = time_transform(user_behavior_data_pv)
user_behavior_data_cart = time_transform(user_behavior_data_cart)
user_behavior_data_fav = time_transform(user_behavior_data_fav)
user_behavior_data_buy = time_transform(user_behavior_data_buy)

# %% [markdown]
# ## 用户侧特征工程

# %% [markdown]
# ### 静态特征
# 用户ID/性别/年龄/用户价值/消费等级/是否是大学生/所在城市
#

# %%
user_static_feas = ['userid', 'final_gender_code', 'age_level',
                    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']

# %%
user_data[user_static_feas].head()

# %% [markdown]
# ### 统计特征
# 1）userid-time_stamp-adgroup_id确定一条数据
# 2）user近3天点击行为
# 3）不同年龄、价值、消费水平、消费城市用户前一天的点击情况

# %% [markdown]
# #### 用户近三天点击行为

# %%
# user-time-adgroup_id对应唯一一条数据，所以同一userid在统一时间下有多条数据，所以先进行user-date级别的统计

user_by_date_data = ad_user_sample_data.groupby(['userid', 'date_ymd'])['clk'].agg(['count', 'sum']).rename(
    {'count': 'disp_count', 'sum': 'clk_cnt'}, axis=1).reset_index()
user_by_date_data_ = user_by_date_data.sort_values(by='date_ymd').set_index('date_ymd')
a = user_by_date_data_.groupby('userid')['disp_count'].rolling(3, min_periods=1,
                                                               closed='left').sum().reset_index().rename(
    {'disp_count': 'last_3days_disp_cnt'}, axis=1)
b = user_by_date_data_.groupby('userid')['clk_cnt'].rolling(3, min_periods=1, closed='left').sum().reset_index().rename(
    {'clk_cnt': 'last_3days_clk_cnt'}, axis=1)

user_by_date_stat = a.merge(b, on=['userid', 'date_ymd'], how='left')
user_by_date_stat['last_3days_clk_disp_ratio'] = round(
    100 * user_by_date_stat['last_3days_clk_cnt'] / user_by_date_stat['last_3days_disp_cnt'], 2)


# %% [markdown]
# #### 不同年龄、价值、消费水平、消费城市用户前一天的点击情况

# %%
def history_clk_by_col_stat(df, cols, cols_name):
    grp_cols = cols + ['date_ymd']
    inner_df = df.groupby(grp_cols)['clk'].agg(['count', 'sum']).rename(
        {'count': f'disp_count_by_{cols_name}', 'sum': f'clk_cnt_by_{cols_name}'}, axis=1).reset_index()
    inner_df_ = inner_df.sort_values(by='date_ymd').set_index('date_ymd')
    inner_df_[f'lastday_disp_cnt_by_{cols_name}'] = inner_df_.groupby(cols)[f'disp_count_by_{cols_name}'].shift(1)
    inner_df_[f'lastday_clk_cnt_by_{cols_name}'] = inner_df_.groupby(cols)[f'clk_cnt_by_{cols_name}'].shift(1)
    inner_df_[f'lastday_clk_disp_ratio_by_{cols_name}'] = round(
        100 * inner_df_[f'lastday_clk_cnt_by_{cols_name}'] / inner_df_[f'lastday_disp_cnt_by_{cols_name}'], 2)
    inner_df_.drop([f'disp_count_by_{cols_name}', f'clk_cnt_by_{cols_name}'], axis=1, inplace=True)
    inner_df_.fillna(0, inplace=True)
    return inner_df_.reset_index()


# %%
# groupby一个user基础特征
lastday_user_clk_by_age = history_clk_by_col_stat(ad_user_sample_data, ['age_level'], 'age')
lastday_user_clk_by_pvalue = history_clk_by_col_stat(ad_user_sample_data, ['pvalue_level'], 'pvalue')
lastday_user_clk_by_shopping_level = history_clk_by_col_stat(ad_user_sample_data, ['shopping_level'], 'shopping')
lastday_user_clk_by_new_user_class_level = history_clk_by_col_stat(ad_user_sample_data, ['new_user_class_level'],
                                                                   'new_user_class')

# groupby组合两个user基础特征
lastday_user_clk_by_age_gender = history_clk_by_col_stat(ad_user_sample_data, ['age_level', 'final_gender_code'],
                                                         'age_gender')
lastday_user_clk_by_age_occupation = history_clk_by_col_stat(ad_user_sample_data, ['age_level', 'occupation'],
                                                             'age_occupation ')
lastday_user_clk_by_age_pvalue = history_clk_by_col_stat(ad_user_sample_data, ['age_level', 'pvalue_level'],
                                                         'age_pvalue ')

# %% [markdown]
# #### 合并用户侧特征

# %%
ad_user_sample_data_fea = ad_user_sample_data.copy()
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(user_by_date_stat, on=['userid', 'date_ymd'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_age, on=['date_ymd', 'age_level'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_pvalue, on=['date_ymd', 'pvalue_level'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_shopping_level,
                                                        on=['date_ymd', 'shopping_level'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_new_user_class_level,
                                                        on=['date_ymd', 'new_user_class_level'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_age_gender,
                                                        on=['date_ymd', 'age_level', 'final_gender_code'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_age_occupation,
                                                        on=['date_ymd', 'age_level', 'occupation'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_user_clk_by_age_pvalue,
                                                        on=['date_ymd', 'age_level', 'pvalue_level'], how='left')

# %% [markdown]
# ## 广告侧特征工程

# %% [markdown]
# ### 静态特征

# %%
ad_static_feas = ['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price']
ad_data[ad_static_feas].head()

# %% [markdown]
# ### 统计特征
# 1）广告近三天的被点击情况
# 2）不同品牌&品类&计划&广告主的广告前一天的点击情况

# %% [markdown]
# #### 广告近三天的被点击情况

# %%
ad_by_date_data = ad_user_sample_data.groupby(['adgroup_id', 'date_ymd'])['clk'].agg(['count', 'sum']).rename(
    {'count': 'disp_count', 'sum': 'clk_cnt'}, axis=1).reset_index()
ad_by_date_data_ = ad_by_date_data.sort_values(by='date_ymd').set_index('date_ymd')
a = ad_by_date_data_.groupby('adgroup_id')['disp_count'].rolling(3, min_periods=1,
                                                                 closed='left').sum().reset_index().rename(
    {'disp_count': 'last_3days_disp_cnt_ad'}, axis=1)
b = ad_by_date_data_.groupby('adgroup_id')['clk_cnt'].rolling(3, min_periods=1,
                                                              closed='left').sum().reset_index().rename(
    {'clk_cnt': 'last_3days_clk_cnt_ad'}, axis=1)

ad_by_date_stat = a.merge(b, on=['adgroup_id', 'date_ymd'], how='left')
ad_by_date_stat['last_3days_clk_disp_ratio_ad'] = round(
    100 * ad_by_date_stat['last_3days_clk_cnt_ad'] / ad_by_date_stat['last_3days_disp_cnt_ad'], 2)

# %% [markdown]
# #### 不同品牌&品类&计划&广告主的广告前一次的点击情况

# %%
# groupby一个ad基础特征
lastday_ad_clk_by_cate = history_clk_by_col_stat(ad_user_sample_data, ['cate_id'], 'cate')
lastday_ad_clk_by_campaign = history_clk_by_col_stat(ad_user_sample_data, ['campaign_id'], 'campaign')
lastday_ad_clk_by_customer = history_clk_by_col_stat(ad_user_sample_data, ['customer'], 'customer')
lastday_ad_clk_by_brand = history_clk_by_col_stat(ad_user_sample_data, ['brand'], 'brand')

# groupby组合两个ad基础特征
lastday_ad_clk_by_brand_campaign = history_clk_by_col_stat(ad_user_sample_data, ['brand', 'campaign_id'],
                                                           'brand_campaign')
lastday_ad_clk_by_customer_campaign = history_clk_by_col_stat(ad_user_sample_data, ['customer', 'campaign_id'],
                                                              'customer_campaign ')
lastday_ad_clk_by_customer_cate = history_clk_by_col_stat(ad_user_sample_data, ['customer', 'cate_id'],
                                                          'customer_cate ')

# %% [markdown]
# #### 合并广告侧特征

# %%
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(ad_by_date_data, on=['adgroup_id', 'date_ymd'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_cate, on=['date_ymd', 'cate_id'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_campaign, on=['date_ymd', 'campaign_id'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_customer, on=['date_ymd', 'customer'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_brand, on=['date_ymd', 'brand'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_brand_campaign,
                                                        on=['date_ymd', 'brand', 'campaign_id'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_customer_campaign,
                                                        on=['date_ymd', 'customer', 'campaign_id'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_clk_by_customer_cate,
                                                        on=['date_ymd', 'customer', 'cate_id'], how='left')

# %% [markdown]
# ## 用户-广告交叉特征
# 1）用户-广告交叉过去三天点击情况
# 2）二阶交叉特征：用户性别 x 广告品牌、用户年龄 x 广告品牌、用户性别 x 广告类别、用户年龄 x 广告类别前一天点击情况
# 3）三阶交叉特征：用户性别 x 是否大学生 x 广告品牌、用户性别 x 是否大学生 x 广告类别

# %% [markdown]
# ### 交叉统计特征

# %% [markdown]
# #### 用户-广告交叉过去三天点击情况

# %%
ad_user_by_date_data = ad_user_sample_data.groupby(['userid', 'adgroup_id', 'date_ymd'])['clk'].agg(
    ['count', 'sum']).rename({'count': 'disp_count', 'sum': 'clk_cnt'}, axis=1).reset_index()
ad_user_by_date_data_ = ad_user_by_date_data.sort_values(by='date_ymd').set_index('date_ymd')
a = ad_user_by_date_data_.groupby(['userid', 'adgroup_id'])['disp_count'].rolling(3, min_periods=1,
                                                                                  closed='left').sum().reset_index().rename(
    {'disp_count': 'last_3days_disp_cnt_ad_user'}, axis=1)
b = ad_user_by_date_data_.groupby(['userid', 'adgroup_id'])['clk_cnt'].rolling(3, min_periods=1,
                                                                               closed='left').sum().reset_index().rename(
    {'clk_cnt': 'last_3days_clk_cnt_ad_user'}, axis=1)
ad_user_by_date_stat = a.merge(b, on=['userid', 'adgroup_id', 'date_ymd'], how='left')
ad_user_by_date_stat['last_3days_clk_disp_ratio_ad_user'] = round(
    100 * ad_user_by_date_stat['last_3days_clk_cnt_ad_user'] / ad_user_by_date_stat['last_3days_disp_cnt_ad_user'], 2)

# %% [markdown]
# #### 用户性别 x 广告品牌、用户年龄 x 广告品牌、用户性别 x 广告类别、用户年龄 x 广告类别前一次点击情况

# %%
# groupby二阶交叉特征
lastday_ad_user_clk_by_gender_brand = history_clk_by_col_stat(ad_user_sample_data, ['final_gender_code', 'brand'],
                                                              'gender_brand')
lastday_ad_user_clk_by_age_brand = history_clk_by_col_stat(ad_user_sample_data, ['age_level', 'brand'], 'age_brand')
lastday_ad_user_clk_by_gender_cate = history_clk_by_col_stat(ad_user_sample_data, ['final_gender_code', 'cate_id'],
                                                             'gender_cate')
lastday_ad_user_clk_by_age_cate = history_clk_by_col_stat(ad_user_sample_data, ['age_level', 'cate_id'], 'age_cate')

# %% [markdown]
# #### 用户性别 x 是否大学生 x 广告品牌、用户性别 x 是否大学生 x 广告类别

# %%
# groupby三阶交叉特征
lastday_ad_user_clk_by_gender_occupation_brand = history_clk_by_col_stat(ad_user_sample_data,
                                                                         ['final_gender_code', 'occupation', 'brand'],
                                                                         'gender_occp_brand')
lastday_ad_user_clk_by_gender_occupation_cate = history_clk_by_col_stat(ad_user_sample_data,
                                                                        ['final_gender_code', 'occupation', 'cate_id'],
                                                                        'gender_occp_cate')

# %% [markdown]
# #### 合并交叉特征

# %%
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(ad_user_by_date_stat, on=['userid', 'adgroup_id', 'date_ymd'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_gender_brand,
                                                        on=['date_ymd', 'final_gender_code', 'brand'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_age_brand,
                                                        on=['date_ymd', 'age_level', 'brand'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_gender_cate,
                                                        on=['date_ymd', 'final_gender_code', 'cate_id'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_age_cate,
                                                        on=['date_ymd', 'age_level', 'cate_id'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_gender_occupation_brand,
                                                        on=['date_ymd', 'occupation', 'brand', 'final_gender_code'],
                                                        how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(lastday_ad_user_clk_by_gender_occupation_cate,
                                                        on=['date_ymd', 'occupation', 'cate_id', 'final_gender_code'],
                                                        how='left')


# %% [markdown]
# ## 用户行为特征
#
# 1）用户过去3/7天的pv/cart/fav/buy
# 2）用户前一天/周同比的pv/cart/fav/buy量

# %% [markdown]
# ### 行为统计特征

# %% [markdown]
# #### 过去i天的pv/cart/fav/buy 的mean/std/max/min

# %%
def user_behavior_past_days(df, col, past_days):
    inner_df = df.groupby(['userid', 'date_ymd'])['btag'].count().reset_index().rename({'btag': f'{col}_cnt'}, axis=1)
    inner_df = inner_df.sort_values(by='date_ymd').set_index('date_ymd')
    inner_df_max = inner_df.groupby('userid')[f'{col}_cnt'].rolling(past_days, min_periods=1,
                                                                    closed='left').max().reset_index().rename(
        {f'{col}_cnt': f'last_{past_days}days_{col}_cnt_max'}, axis=1)
    inner_df_min = inner_df.groupby('userid')[f'{col}_cnt'].rolling(past_days, min_periods=1,
                                                                    closed='left').min().reset_index().rename(
        {f'{col}_cnt': f'last_{past_days}days_{col}_cnt_min'}, axis=1)
    inner_df_mean = inner_df.groupby('userid')[f'{col}_cnt'].rolling(past_days, min_periods=1,
                                                                     closed='left').mean().reset_index().rename(
        {f'{col}_cnt': f'last_{past_days}days_{col}_cnt_mean'}, axis=1)
    inner_df_std = inner_df.groupby('userid')[f'{col}_cnt'].rolling(past_days, min_periods=1,
                                                                    closed='left').std().reset_index().rename(
        {f'{col}_cnt': f'last_{past_days}days_{col}_cnt_std'}, axis=1)

    inner_df_ = inner_df_max.merge(inner_df_min, on=['userid', 'date_ymd'], how='left')
    inner_df_ = inner_df_.merge(inner_df_mean, on=['userid', 'date_ymd'], how='left')
    inner_df_ = inner_df_.merge(inner_df_std, on=['userid', 'date_ymd'], how='left')
    inner_df_ = inner_df_.fillna(0)

    del inner_df_max
    del inner_df_min
    del inner_df_mean
    del inner_df_std
    return inner_df_


# %%
user_behavior_past3_days_pv = user_behavior_past_days(user_behavior_data_pv, 'pv', 3)

# %%
user_behavior_past3_days_cart = user_behavior_past_days(user_behavior_data_cart, 'cart', 3)

# %%
user_behavior_past3_days_fav = user_behavior_past_days(user_behavior_data_fav, 'fav', 3)

# %%
user_behavior_past3_days_buy = user_behavior_past_days(user_behavior_data_buy, 'buy', 3)


# %% [markdown]
# #### 用户前一次的pv/cart/fav/buy量

# %%
def user_behavior_shift_stat(df, col, shift_days):
    df_ = df.copy()
    inner_df = df_.groupby(['userid', 'date_ymd'])['btag'].count().reset_index().rename({'btag': f'{col}_cnt'}, axis=1)
    inner_df_ = inner_df.sort_values(by='date_ymd').set_index('date_ymd')
    inner_df_[f'{col}_cnt_{shift_days}days_before'] = inner_df_.groupby('userid')[f'{col}_cnt'].shift(1)
    inner_df_ = inner_df_.reset_index().drop(f'{col}_cnt', axis=1)

    return inner_df_


# %%
user_behavior_before_shift1_pv = user_behavior_shift_stat(user_behavior_data_pv, 'pv', 1)
user_behavior_before_shift1_cart = user_behavior_shift_stat(user_behavior_data_cart, 'cart', 1)
user_behavior_before_shift1_fav = user_behavior_shift_stat(user_behavior_data_fav, 'fav', 1)
user_behavior_before_shift1_buy = user_behavior_shift_stat(user_behavior_data_buy, 'buy', 1)

# %% [markdown]
# #### 合并用户行为特征

# %%
from functools import reduce

dfs_past = [user_behavior_past3_days_pv, user_behavior_past3_days_cart, user_behavior_past3_days_fav,
            user_behavior_past3_days_buy]
df_past_final = reduce(lambda df_left, df_right: pd.merge(df_left, df_right, on=['userid', 'date_ymd'], how='left'),
                       dfs_past).fillna(0)

# %%
dfs_shift = [user_behavior_before_shift1_pv, user_behavior_before_shift1_cart, user_behavior_before_shift1_fav,
             user_behavior_before_shift1_buy]
df_shift_final = reduce(lambda df_left, df_right: pd.merge(df_left, df_right, on=['userid', 'date_ymd'], how='left'),
                        dfs_shift).fillna(0)

# %%
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(df_past_final, on=['userid', 'date_ymd'], how='left')
ad_user_sample_data_fea = ad_user_sample_data_fea.merge(df_shift_final, on=['userid', 'date_ymd'], how='left')

# %%
ad_user_sample_data_fea.to_pickle('ad_user_sample_data_fea.pkl')


