import cv2
import numpy as np

# 加载图像
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('hustlogo.bmp')

# 调整 logo 的尺寸比例
scale_percent = 30
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
# 缩小 logo
img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 如果像素值大于 175，它会被设为 255（白色）。如果像素值小于或等于 175，它会被设为 0（黑色）。
ret, mask_front = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)  # 这是图像分割方法，后面讲到。

mask_inv = cv2.bitwise_not(mask_front)  # 取反
# Now black-out the area of logo in ROI
# 取roi 中与mask 中不为零的值对应的像素的值，其他值为0
# 注意这里必须有mask=mask 或者mask=mask_inv, 其中的“mask=” 不能忽略
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_front)  # 将roi中的logo部分去掉
# 取roi 中与mask_inv 中不为零的值对应的像素的值，其他值为0。
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)  # 将logo中的非logo部分去掉
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
img1[0:rows, img1.shape[1] - cols:img1.shape[1]] = dst
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
