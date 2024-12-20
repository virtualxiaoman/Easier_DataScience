from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 数据路径
PROCESSED_DATA_PATH = "G:/DataSets/Ali_Display_Ad_Click/processed_data"


# 加载和预处理数据
def load_and_preprocess_data(path, sample_size):
    # 加载数据
    ad_u_data = pd.read_csv(f'{path}/ad_u_data.csv')
    ad_u_data = ad_u_data.iloc[:sample_size]

    # 数据去重
    ad_u_data = ad_u_data.drop_duplicates(subset=['user_id', 'adgroup_id']).reset_index(drop=True)

    # 将用户和广告 ID 映射为连续索引
    ad_u_data['user_id'] = pd.Categorical(ad_u_data['user_id']).codes
    ad_u_data['adgroup_id'] = pd.Categorical(ad_u_data['adgroup_id']).codes

    return ad_u_data


# 定义用户-广告 Embedding 模型
def ad_user_embedding_model(user_length, ad_length, embedding_size=50):
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


# 转换数据为 TensorFlow Dataset 格式
def create_tf_dataset(input_data, target_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).batch(batch_size)
    return dataset


# 评估模型性能
def evaluate_model(model, val_input, val_target):
    # 预测值
    val_pred = model.predict(val_input)

    # 统计预测分布
    predicted_0 = np.sum(val_pred < 0.5)
    predicted_1 = np.sum(val_pred >= 0.5)
    predicted_0_ratio = predicted_0 / len(val_pred)
    predicted_1_ratio = predicted_1 / len(val_pred)

    print(f"Predicted 0: {predicted_0}, Predicted 1: {predicted_1}")
    print(f"Predicted 0 ratio: {predicted_0_ratio:.2f}, Predicted 1 ratio: {predicted_1_ratio:.2f}")

    # 计算 AUC
    val_auc = roc_auc_score(val_target, val_pred)
    print(f"Validation AUC: {val_auc:.4f}")

    # 输出分类报告
    val_pred = np.round(val_pred)
    print(classification_report(val_target, val_pred))

    return val_pred, val_auc


if __name__ == "__main__":
    # 数据加载与预处理
    SAMPLE_SIZE = 2048 * 300
    BATCH_SIZE = 2048
    EPOCHS = 3

    ad_user_sample_data_ = load_and_preprocess_data(PROCESSED_DATA_PATH, SAMPLE_SIZE)

    # 用户和广告的总数
    user_length = ad_user_sample_data_['user_id'].nunique()
    ad_length = ad_user_sample_data_['adgroup_id'].nunique()
    print(f"User length: {user_length}, Ad length: {ad_length}")

    # 准备输入与目标数据
    input_data = {
        'user_id': ad_user_sample_data_['user_id'].values,
        'adgroup_id': ad_user_sample_data_['adgroup_id'].values
    }
    target_data = ad_user_sample_data_['clk'].values

    # 划分训练集与验证集
    combined_data = pd.DataFrame({
        'user_id': ad_user_sample_data_['user_id'],
        'adgroup_id': ad_user_sample_data_['adgroup_id']
    })
    train_input, val_input, train_target, val_target = train_test_split(
        combined_data, target_data, test_size=0.2, random_state=42
    )

    print(f"Input data: {input_data['user_id'].shape, input_data['adgroup_id'].shape} ")  # ((524761,), (524761,))
    # 输出train_input的shape
    print(train_input.shape)  # (419808, 2)
    # 查找train_input和val_input有没有相同的值的shape
    print(train_input[train_input['user_id'].isin(val_input['user_id'])].shape)  # (289265, 2)
    print(train_input[train_input['adgroup_id'].isin(val_input['adgroup_id'])].shape)  # (341164, 2)
    # 查找train_input的(x, y)有没有和val_input的(x, y)相同的值
    print(train_input[train_input['user_id'].isin(val_input['user_id']) &
                      train_input['adgroup_id'].isin(val_input['adgroup_id'])].shape)  # (227251, 2)

    val_input = {
        'user_id': val_input['user_id'].values,
        'adgroup_id': val_input['adgroup_id'].values
    }

    # 创建模型
    embedding_model = ad_user_embedding_model(user_length, ad_length)
    embedding_model.summary()

    # 转换为 TensorFlow Dataset
    dataset = create_tf_dataset(input_data, target_data, BATCH_SIZE)

    # 训练模型
    print("开始训练 Embedding 模型")
    embedding_model.fit(dataset, validation_data=(val_input, val_target), epochs=EPOCHS)

    # 评估模型
    print("评估模型性能")
    evaluate_model(embedding_model, val_input, val_target)
