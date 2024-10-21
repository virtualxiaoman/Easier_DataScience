import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('flower2.jpg', 0)
plt.hist(img.ravel(), 256, [0, 256]);
plt.show()

img = cv2.imread('flower2.jpg')
color = ('b', 'g', 'r')
# 对一个列表或数组既要遍历索引又要遍历元素时
# 使用内置enumerrate 函数会有更加直接，优美的做法
# enumerate 会将数组或列表组成一个索引序列。
# 使我们再获取索引和索引内容的时候更加方便
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
plt.plot(histr, color=col)
plt.xlim([0, 256])
plt.show()

img = cv2.imread('flower2.jpg', 0)
# flatten() 将数组变成一维
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# 计算累积分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
