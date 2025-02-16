import os
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 初始化 DINOv2 模型
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 图库路径和保存路径
path_local = "F:/Picture/pixiv/BA"
save_path = "./data"
os.makedirs(save_path, exist_ok=True)

# 创建空的 DataFrame
columns = ["id", "path", "vector"]
data = pd.DataFrame(columns=columns)


# 提取特征并保存到 DataFrame
def extract_and_save_features(image_folder, save_folder):
    global data
    for idx, image_name in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        # 提取特征
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        # 将特征和路径保存到 DataFrame
        new_row = pd.DataFrame({
            "id": [idx],
            "path": [image_path],
            "vector": [features]
        })
        data = pd.concat([data, new_row], ignore_index=True)

        print(f"Processed {image_name} and saved to DataFrame")

    # 保存 DataFrame 到 .pkl 文件
    data.to_pickle(os.path.join(save_folder, "image_features.pkl"))
    print("DataFrame saved to .pkl file.")


# 提取图库中所有图片的特征
# extract_and_save_features(path_local, save_path)

# 查询图片路径
path_origin = "pic/119550361_p0.jpg"
path_origin = "pic/clip.png"
# 提取查询图片的特征
query_image = Image.open(path_origin).convert("RGB")
query_inputs = processor(images=query_image, return_tensors="pt").to(device)
with torch.no_grad():
    query_outputs = model(**query_inputs)
    query_features = query_outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

# 加载保存的 DataFrame
features_df = pd.read_pickle(os.path.join(save_path, "image_features.pkl"))
print(features_df.head())
print(features_df["vector"].values[0].shape)  # 特征向量的维度 (768,)


# 计算相似度
def search_similar_images(query_features, features_df, top_k=5):
    similarities = []
    for idx, row in features_df.iterrows():
        image_features = row["vector"]
        similarity = cosine_similarity([query_features], [image_features])[0][0]
        similarities.append((row["id"], row["path"], similarity))

    # 按相似度排序并返回前 top_k 个结果
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


# 搜索相似图片
similar_images = search_similar_images(query_features, features_df, top_k=10)

# 打印结果
for id, path, similarity in similar_images:
    print(f"ID: {id}, Path: {path}, Similarity: {similarity:.4f}")
