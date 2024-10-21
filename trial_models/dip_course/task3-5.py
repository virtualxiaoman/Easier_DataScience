import cv2
import numpy as np

# 全局变量
points = []  # 存储四个顶点
rect_w, rect_h = 300, 400  # 目标矩形的宽和高

# 鼠标回调函数，用于记录点击的四个顶点
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")

        if len(points) == 4:
            print("四个点已选定，进行变换")

            # 将四边形顶点转换为矩形
            pts_src = np.array(points, dtype=np.float32)
            pts_dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], dtype=np.float32)

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)

            # 仿射变换到矩形
            transformed = cv2.warpPerspective(img, M, (rect_w, rect_h))

            # 显示结果
            cv2.imshow('Transformed Image', transformed)

# 读取图像
img = cv2.imread('./1.jpg')

# 创建窗口并设置鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下ESC键退出
        break

# 释放窗口
cv2.destroyAllWindows()
