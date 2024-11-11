import cv2
import numpy as np
import mss


def track_blue_objects_on_screen():
    # 设置屏幕捕获区域，这里设置为全屏
    screen = mss.mss()
    monitor = screen.monitors[1]  # 选择主屏幕

    while True:
        # 捕获屏幕
        img = screen.grab(monitor)
        frame = np.array(img)

        # 转换颜色空间，从 BGRA 转为 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 将帧转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 设定蓝色的阈值
        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([140, 255, 255])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 进行形态学操作以去除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 查找蓝色物体的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果找到轮廓
        if contours:
            for contour in contours:
                # 获取每个轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                # 在帧上绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)

        # 将图像缩小以适应窗口显示
        resized_frame = cv2.resize(frame, (480, 300))  # 调整到你需要的大小

        # 显示缩小后的帧
        cv2.imshow('Blue Object Tracker', resized_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭窗口
    cv2.destroyAllWindows()


# 调用函数
track_blue_objects_on_screen()
