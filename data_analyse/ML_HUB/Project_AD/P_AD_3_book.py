from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

processed_data_path = "G:/DataSets/Ali_Display_Ad_Click/processed_data"

ad_u_data = pd.read_csv(f'{processed_data_path}/ad_u_data.csv')
# 取前2048*100个，这样速度更快
ad_u_data = ad_u_data.iloc[:2048 * 300]
# 数据去重，防止同一用户-广告对重复
ad_user_sample_data_ = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)

# 将用户和广告 ID 映射为连续索引
ad_user_sample_data_['user_id'] = pd.Categorical(ad_user_sample_data_['user_id']).codes
ad_user_sample_data_['adgroup_id'] = pd.Categorical(ad_user_sample_data_['adgroup_id']).codes
# 用户和广告的总数（Embedding 层输入维度）
user_length = ad_user_sample_data_['user_id'].nunique()
ad_length = ad_user_sample_data_['adgroup_id'].nunique()
print(f"User length: {user_length}, Ad length: {ad_length}")  # 204800时这里是User length: 68437, Ad length: 29330


# 定义用户-广告 Embedding 模型
def ad_user_embedding_model(embedding_size=50):
    # 输入层
    ad = Input(name='adgroup_id', shape=[1])
    user = Input(name='user_id', shape=[1])

    # 广告和用户的 Embedding 层
    ad_embedding = Embedding(name='ad_embedding', input_dim=ad_length, output_dim=embedding_size)(ad)
    user_embedding = Embedding(name='user_embedding', input_dim=user_length, output_dim=embedding_size)(user)

    # 计算用户和广告 Embedding 的点积（归一化）
    merged = Dot(name='dot_product', normalize=True, axes=2)([ad_embedding, user_embedding])
    merged = Reshape(target_shape=[1])(merged)  # 将点积结果拉平

    # 使用 Dense 层和 sigmoid 激活计算点击预测概率
    merged = Dense(1, activation='sigmoid')(merged)

    # 构建模型
    model = Model(inputs=[ad, user], outputs=merged)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# 创建 Embedding 模型
embedding_model = ad_user_embedding_model()
embedding_model.summary()

# 转换为 TensorFlow Dataset 格式，适合模型训练
input_data = {'user_id': ad_user_sample_data_['user_id'].values,
              'adgroup_id': ad_user_sample_data_['adgroup_id'].values}
target_data = ad_user_sample_data_['clk'].values

print(f"Input data: {input_data['user_id'].shape, input_data['adgroup_id'].shape}, "
      f"Target data: {target_data.shape}")  # Input data: ((175410,), (175410,)), Target data: (175410,)


from sklearn.model_selection import train_test_split
combined_data = pd.DataFrame({
    'user_id': ad_user_sample_data_['user_id'],
    'adgroup_id': ad_user_sample_data_['adgroup_id']
})

train_input, val_input, train_target, val_target = train_test_split(
    combined_data, target_data, test_size=0.2, random_state=42
)


# dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).batch(2048)  # 批量大小 2048 的数据集
# 转换为适合模型输入的 TensorFlow Dataset 格式
dataset = tf.data.Dataset.from_tensor_slices((
    {'user_id': input_data['user_id'], 'adgroup_id': input_data['adgroup_id']},
    target_data
)).batch(2048)
val_input = {
    'user_id': val_input['user_id'].values,
    'adgroup_id': val_input['adgroup_id'].values
}
# 检查训练数据的样本
for batch in dataset.take(1):
    print(batch)

# 训练模型
print("开始训练 Embedding 模型")
# embedding_model.fit(dataset, epochs=1)
embedding_model.fit(dataset, validation_data=(val_input, val_target), epochs=3)

# 用模型对validation数据集进行预测，查看0,1的分布
val_pred = embedding_model.predict(val_input)
print(f"Validation predictions: {val_pred[:10]}")
# 查看预测为0,1的数量和比例
print(f"Predicted 0: {np.sum(val_pred < 0.5)}, Predicted 1: {np.sum(val_pred >= 0.5)}")
print(f"Predicted 0 ratio: {np.sum(val_pred < 0.5) / len(val_pred)}, "
        f"Predicted 1 ratio: {np.sum(val_pred >= 0.5) / len(val_pred)}")

# 查看auc
from sklearn.metrics import roc_auc_score
val_auc = roc_auc_score(val_target, val_pred)
print(f"Validation AUC: {val_auc}")


# test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target)).batch(2048)
# test_loss, test_acc = embedding_model.evaluate(test_dataset)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
#

# 提取并归一化 Embedding 权重
def extract_embedding_weights(layer_name, model):
    embedding_layer = model.get_layer(layer_name)
    embedding_weights = embedding_layer.get_weights()[0]
    # 对向量归一化
    embedding_weights = embedding_weights / np.linalg.norm(embedding_weights, axis=1, keepdims=True)
    return embedding_weights


ad_embedding_weights = extract_embedding_weights('ad_embedding', embedding_model)
user_embedding_weights = extract_embedding_weights('user_embedding', embedding_model)

# Ad embedding weights: (29330, 50), User embedding weights: (68437, 50)
print(f"Ad embedding weights: {ad_embedding_weights.shape}, User embedding weights: {user_embedding_weights.shape}")

exit("降维较为耗时，不展示~~~")
# 降维：TSNE 和 UMAP
from sklearn.manifold import TSNE
from umap import UMAP


def reduce_dim(weights, components=2, method='tsne'):
    if method == 'tsne':
        return TSNE(n_components=components, metric='cosine').fit_transform(weights)
    elif method == 'umap':
        return UMAP(n_components=components, metric='cosine').fit_transform(weights)


# # 降维并可视化
# sample_indices = np.random.choice(user_embedding_weights.shape[0], 10000, replace=False)
# ad_embedding_weights_sample = ad_embedding_weights[sample_indices]
# user_embedding_weights_sample = user_embedding_weights[sample_indices]
#
# ad_tsne = reduce_dim(ad_embedding_weights_sample, method='tsne')
# user_tsne = reduce_dim(user_embedding_weights_sample, method='tsne')
# 降维并可视化
# ad_sample_indices = np.random.choice(ad_embedding_weights.shape[0], 1000, replace=False)
# user_sample_indices = np.random.choice(user_embedding_weights.shape[0], 1000, replace=False)
#
# ad_embedding_weights_sample = ad_embedding_weights[ad_sample_indices]
# user_embedding_weights_sample = user_embedding_weights[user_sample_indices]
#
# ad_tsne = reduce_dim(ad_embedding_weights_sample, method='tsne')
# user_tsne = reduce_dim(user_embedding_weights_sample, method='tsne')

# 数据采样及降维
def sample_and_reduce(weights, sample_size=1000, method='tsne'):
    indices = np.random.choice(weights.shape[0], sample_size, replace=False)
    sampled_weights = weights[indices]
    reduced = reduce_dim(sampled_weights, method=method)
    return reduced, indices


ad_tsne, ad_sample_indices = sample_and_reduce(ad_embedding_weights, sample_size=1000, method='tsne')
user_tsne, user_sample_indices = sample_and_reduce(user_embedding_weights, sample_size=1000, method='tsne')


# def plot_by_col(orign_df, df_tsne, col, top_k, model='TSNE'):
#     """进行可视化"""
#
#     #     print(orign_df[col].value_counts().sort_values(ascending=False).head(top_k).reset_index().rename({'index': col, 'cate_id': 'cnt'}, axis=1))
#
#     plt.figure(figsize=(10, 8))
#     plt.scatter(df_tsne[:, 0], df_tsne[:, 1], marker='.', color='lightblue', alpha=0.2)
#     plt.xlabel(f'{model} 1')
#     plt.ylabel(f'{model} 2')
#     plt.title(f'Embeddings Visualized with {model}')
#     top_10_col_ids = list(orign_df[col].value_counts().head(10).index)
#     for id_ in top_10_col_ids:
#         sel_indexs = list(orign_df[orign_df[col] == id_].index)
#         plt.scatter(df_tsne[sel_indexs, 0], df_tsne[sel_indexs, 1], alpha=0.6,
#                     cmap=plt.cm.tab10, marker='.', s=50)
#     plt.show()
#
#
# plot_by_col(ad_user_sample_data_, ad_tsne, 'adgroup_id', 10)
# plot_by_col(ad_user_sample_data_, user_tsne, 'user_id', 10)


# 绘图函数
def plot_by_col(orign_df, df_tsne, sampled_indices, col, top_k=10, model='TSNE'):
    plt.figure(figsize=(10, 8))
    plt.scatter(df_tsne[:, 0], df_tsne[:, 1], marker='.', color='lightblue', alpha=0.2)
    plt.xlabel(f'{model} 1')
    plt.ylabel(f'{model} 2')
    plt.title(f'Embeddings Visualized with {model}')

    # 选择前 top_k 类别
    sampled_data = orign_df.iloc[sampled_indices]
    top_ids = sampled_data[col].value_counts().head(top_k).index

    # 绘制每个类别
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    for i, id_ in enumerate(top_ids):
        indices = sampled_data[sampled_data[col] == id_].index
        selected_tsne_indices = [sampled_indices.tolist().index(idx) for idx in indices if idx in sampled_indices]
        plt.scatter(df_tsne[selected_tsne_indices, 0], df_tsne[selected_tsne_indices, 1], alpha=0.6,
                    color=colors[i], label=f'{col}: {id_}', s=50)

    plt.legend()
    plt.show()

# 绘图
plot_by_col(ad_u_data, ad_tsne, ad_sample_indices, 'adgroup_id', top_k=10)
plot_by_col(ad_u_data, user_tsne, user_sample_indices, 'user_id', top_k=10)