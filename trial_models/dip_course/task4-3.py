# HSI空间均衡化：只对亮度分量进行处理，图像色调保持不变，但整体亮度提升较明显。
# BGR通道均衡化：每个通道均衡化后，图像的颜色对比度可能会发生较大变化，色彩会变得更鲜艳或扭曲。

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
img = cv2.imread('fig6.png')

# Step 1: 在HSI空间对亮度分量均衡化
# OpenCV中通常使用HSV来代替HSI
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# 对亮度分量V进行直方图均衡化
v_equalized = cv2.equalizeHist(v)

# 将均衡化后的亮度分量合并回HSV图像
hsv_equalized = cv2.merge([h, s, v_equalized])

# 将HSV图像转换回BGR空间
img_hsv_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

# Step 2: 对B、G、R空间逐一做均衡化
b, g, r = cv2.split(img)

# 对每个通道进行直方图均衡化
b_equalized = cv2.equalizeHist(b)
g_equalized = cv2.equalizeHist(g)
r_equalized = cv2.equalizeHist(r)

# 合并均衡化后的B、G、R通道
img_bgr_equalized = cv2.merge([b_equalized, g_equalized, r_equalized])

# 展示原始图像、HSI空间均衡化和BGR空间均衡化的结果
titles = ['Original Image', 'HSI Brightness Equalized', 'BGR Channels Equalized']
images = [img, img_hsv_equalized, img_bgr_equalized]

plt.figure(figsize=(10, 7))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.show()
