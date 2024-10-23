import cv2
import numpy as np
from datetime import datetime

# 加载背景视频
background_video = cv2.VideoCapture('vtest.avi')

# 摄像头视频
cap = cv2.VideoCapture(0)

# 视频保存设置
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# 动态字幕文本
i = 0
show_subtitle = True


# 定义鼠标点击回调函数
def mouse_callback(event, x, y, flags, param):
    global show_subtitle, i
    if event == cv2.EVENT_LBUTTONDOWN:
        show_subtitle = not show_subtitle
    # 如果是右键
    if event == cv2.EVENT_RBUTTONDOWN:
        i += 1


# 设置鼠标回调函数
cv2.namedWindow('Output')
cv2.setMouseCallback('Output', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    ret_bg, bg_frame = background_video.read()

    if not ret or not ret_bg:
        break

    # 如果背景视频到达末尾，重新播放
    if background_video.get(cv2.CAP_PROP_POS_FRAMES) == background_video.get(cv2.CAP_PROP_FRAME_COUNT):
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 调整背景帧大小以匹配摄像头帧
    bg_frame = cv2.resize(bg_frame, (640, 480))

    # 转换为HSV空间以进行颜色分割
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 设置HSV范围以分割人脸肤色（扩展范围以捕捉更多的肤色）
    lower_skin = np.array([0, 20, 70])  # 下限: 捕捉浅肤色
    upper_skin = np.array([200, 255, 255])  # 上限: 捕捉深肤色

    # 创建掩码
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    # 使用形态学闭运算填充空点
    kernel = np.ones((5, 5), np.uint8)  # 创建5x5的卷积核
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 反转掩码
    mask_inv = cv2.bitwise_not(mask)

    # 分离前景和背景
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = cv2.bitwise_and(bg_frame, bg_frame, mask=mask_inv)

    # 合并前景和背景
    frame = cv2.add(fg, bg)

    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 更新字幕文本
    subtitle_text = f'Caption, count: {i}, Time: {current_time}'

    # 添加动态字幕
    if show_subtitle:
        cv2.putText(frame, subtitle_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 显示输出
    cv2.imshow('Output', frame)

    # 保存视频
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
background_video.release()
out.release()
cv2.destroyAllWindows()
