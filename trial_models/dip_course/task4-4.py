import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_cdf(hist):
    """计算累积分布函数 (CDF)"""
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # 正规化
    return cdf


def histogram_specification(source_channel, reference_channel):
    """对单个颜色通道进行直方图规定化"""
    # 计算源图像和参考图像的直方图
    source_hist, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])

    # 计算CDF
    source_cdf = calculate_cdf(source_hist)
    reference_cdf = calculate_cdf(reference_hist)

    # 创建映射表
    mapping = np.zeros(256, dtype=np.uint8)
    g_j = 0
    for g_i in range(256):
        while reference_cdf[g_j] < source_cdf[g_i] and g_j < 255:
            g_j += 1
        mapping[g_i] = g_j

    # 使用映射表调整源图像
    matched_channel = cv2.LUT(source_channel, mapping)

    return matched_channel


def apply_histogram_specification(source_img, reference_img):
    """对彩色图像进行直方图规定化"""
    # 分离源图像和目标图像的三个通道（B, G, R）
    source_b, source_g, source_r = cv2.split(source_img)
    reference_b, reference_g, reference_r = cv2.split(reference_img)

    # 分别对每个通道进行直方图规定化
    matched_b = histogram_specification(source_b, reference_b)
    matched_g = histogram_specification(source_g, reference_g)
    matched_r = histogram_specification(source_r, reference_r)

    # 合并处理后的三个通道
    matched_img = cv2.merge([matched_b, matched_g, matched_r])

    return matched_img


# 读取两幅彩色图像：Fig7A为源图像，Fig7B为参考图像
source_image = cv2.imread('Fig7A.jpg')  # 读取彩色图像
reference_image = cv2.imread('Fig7B.jpg')

# 进行直方图规定化
output_image = apply_histogram_specification(source_image, reference_image)


plt.figure(figsize=(10, 7))
plt.subplot(131)
plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
plt.title('Fig7A')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
plt.title('Fig7B')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Transformed Fig7A')
plt.axis('off')

plt.show()
