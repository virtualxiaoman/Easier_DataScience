import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def img2vector(path):
    # 读取图像并提取SIFT特征
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # 使用GMM生成视觉词汇表
    gmm = GaussianMixture(n_components=100, covariance_type='full', reg_covar=1e-6).fit(descriptors)
    fv = fisher_vector(descriptors, gmm)
    print("Fisher Vector, shape:", fv.shape)
    return fv

# 计算Fisher Vector
def fisher_vector(descriptors, gmm):
    means = gmm.means_
    covariances = gmm.covariances_ + np.eye(gmm.covariances_.shape[1]) * 1e-6  # 添加正则化项
    weights = gmm.weights_

    fv = np.zeros((2 * len(weights) * len(descriptors[0]),))

    # 对协方差进行对角化处理，确保sqrt的计算是对单值而不是数组
    for j in range(len(weights)):
        covariances[j] = np.diag(np.diag(covariances[j]))

    for i, descriptor in enumerate(descriptors):
        for j in range(len(weights)):
            try:
                prob = multivariate_normal.pdf(descriptor, means[j], covariances[j])
            except np.linalg.LinAlgError:
                prob = 0  # 如果协方差矩阵不可逆，直接将概率设为0

            diff = descriptor - means[j]
            sqrt_cov = np.sqrt(np.diag(covariances[j]))

            fv[j * len(descriptor):(j + 1) * len(descriptor)] += (diff / sqrt_cov) * prob / weights[j]
            fv[(len(weights) + j) * len(descriptor):(len(weights) + j + 1) * len(descriptor)] += \
                ((diff ** 2 - np.diag(covariances[j])) / np.diag(covariances[j])) * prob / weights[j]

    return fv


fv_o = img2vector("input/arona_origin.jpg")
fv_l = img2vector("input/arona_local.jpg")
# 计算两个Fisher Vector的相似度
similarity = np.dot(fv_o, fv_l) / (np.linalg.norm(fv_o) * np.linalg.norm(fv_l))
print("Similarity:", similarity)
