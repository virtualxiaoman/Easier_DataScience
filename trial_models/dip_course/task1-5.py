import cv2
import os


def mouse_callback(event, x, y, flags, param):
    global show_subtitle
    if event == cv2.EVENT_LBUTTONDOWN:
        show_subtitle = not show_subtitle  


video_path = 'vtest.avi'
subtitle_text = 'This is a test caption'
show_subtitle = True


if not os.path.isfile(video_path):
    print(f"视频文件 {video_path} 不存在，请检查文件路径")
    exit()


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print(f"无法打开视频文件 {video_path}")
    exit()


cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("视频播放结束")
        break


    if show_subtitle:
        cv2.putText(frame, subtitle_text, (200, frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
