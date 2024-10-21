import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('j.png', 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()

# 开
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# 梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.subplot(131), plt.imshow(opening, cmap='gray'), plt.title('Opening')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(closing, cmap='gray'), plt.title('Closing')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(gradient, cmap='gray'), plt.title('Gradient')
plt.xticks([]), plt.yticks([])
plt.show()

# 边缘检测
image = cv2.imread("j.png", 0)
# 构造一个3×3的结构元素
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(image, element)
erode = cv2.erode(image, element)

# 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilate, erode)

# 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
# 反色，即对二值图每个像素取反
result = cv2.bitwise_not(result)
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result, cmap='gray'), plt.title('Edge')
plt.xticks([]), plt.yticks([])
plt.show()
