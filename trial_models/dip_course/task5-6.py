import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 请自行完成在不同均值滤波核值下的操作（以上是在5x5 模版下实行）
def mean_filter(image_path, kernel_size):
    """
    使用指定大小的均值滤波器对图像进行滤波。
    image_path: 图像文件路径
    kernel_size: 均值滤波器的大小 (n*n 模板中的 n 值)
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("无法加载图像，请检查路径！")
        return

    # 使用指定大小的均值滤波
    filtered_img = cv2.blur(img, (kernel_size, kernel_size))

    # 显示原始图像和滤波后的图像
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered_img, cmap='gray')
    plt.title(f'Filtered Image with {kernel_size}x{kernel_size} Kernel'), plt.xticks([]), plt.yticks([])
    plt.show()

    return filtered_img


# 调用函数，使用不同的模板大小 (如5x5)
mean_filter('test.png', 5)

# LPF（低通滤波）将高频部分去除实现傅立叶变换.
# 读取图像并转换为灰度图像
img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# 进行傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # 将低频部分移到频谱中心

# 获取图像的尺寸
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# 创建低通滤波掩码，保留低频，去除高频
mask = np.zeros((rows, cols), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# 应用掩码
fshift = fshift * mask

# 逆傅里叶变换回到空间域
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 显示原始图像和低通滤波后的图像
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Low Pass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 高通滤波结果映射到B通道：表示图像的高频成分，蓝色区域对应的是空间频率变化较快的区域，如图像的边缘或细节部分。
# 带通滤波结果映射到G通道：表示中频成分，绿色区域对应的是图像的中等空间频率变化区域。
# 低通滤波结果映射到R通道：表示图像的低频成分，红色区域对应的是图像的缓慢变化部分，如平滑区域或大致的轮廓。


# 读取图像并转为灰度图
img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# 获取图像的傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 获取图像的行列数
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# 创建空白滤波结果图像
low_pass = np.copy(fshift)
high_pass = np.copy(fshift)
band_pass = np.copy(fshift)

# 低通滤波：保留低频部分
low_pass[crow - 30:crow + 30, ccol - 30:ccol + 30] = fshift[crow - 30:crow + 30, ccol - 30:ccol + 30]
# 高通滤波：去除低频部分，保留高频部分
high_pass[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
# 带通滤波：保留中频部分
band_pass[:crow - 15, :] = 0
band_pass[crow + 15:, :] = 0
band_pass[:, :ccol - 15] = 0
band_pass[:, ccol + 15:] = 0

# 对应的逆傅里叶变换，得到滤波后的图像
img_low_pass = np.fft.ifft2(np.fft.ifftshift(low_pass))
img_low_pass = np.abs(img_low_pass)
img_high_pass = np.fft.ifft2(np.fft.ifftshift(high_pass))
img_high_pass = np.abs(img_high_pass)
img_band_pass = np.fft.ifft2(np.fft.ifftshift(band_pass))
img_band_pass = np.abs(img_band_pass)

# 标准化结果到0-255
img_low_pass = cv2.normalize(img_low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_high_pass = cv2.normalize(img_high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_band_pass = cv2.normalize(img_band_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 创建彩色图像，将滤波结果映射到B, G, R通道
color_image = cv2.merge([img_high_pass, img_band_pass, img_low_pass])

# 显示原始灰度图像和处理后的彩色图像
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(color_image)
plt.title('Frequency Analysis in BGR'), plt.xticks([]), plt.yticks([])
plt.show()
