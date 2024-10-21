# © virtual小满 2024-10-05
# 以下代码是通过MTCNN检测人脸，InceptionResnetV1获取人脸特征向量，计算欧氏距离判断是否为同一个人的示例代码。
# keywords: 人脸识别、MTCNN、Resnet、vggface2、欧氏距离、人脸特征向量

import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

predictor_path = "./model/official/shape_predictor_68_face_landmarks.dat"


# 标识视频中的68个人脸特征点
def face_landmark_in_video(predictor_path):
    # 使用dlib自带的人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 加载官方提供的68个特征点模型
    predictor = dlib.shape_predictor(predictor_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # 读取每一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        # 转换颜色，dlib使用BGR格式，而cv2默认是RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        dets = detector(img, 0)

        # 遍历检测到的所有人脸
        for k, d in enumerate(dets):
            # 获取68个关键点
            shape = predictor(img, d)

            # 遍历68个点，标记在人脸上
            for index, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                cv2.circle(frame, pt_pos, 1, (0, 255, 0), 2)  # 绿色小圆圈
                cv2.putText(frame, str(index + 1), pt_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 255), 1, cv2.LINE_AA)  # 红色数字

        # 显示标记后的图像
        cv2.imshow("Face Landmarks", frame)

        # 检测按键是否是q，退出程序
        if cv2.waitKey(1) == ord('q'):
            print("q pressed, exiting.")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# 标识图片中的68个人脸特征点
def face_landmark_in_pic(predictor_path, pic_path):
    # dlib预测器
    detector = dlib.get_frontal_face_detector()
    # 读入68点数据
    predictor = dlib.shape_predictor(predictor_path)

    # cv2读取图像
    img = cv2.imread(pic_path)
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    print("Number of faces detected: {}".format(len(rects)))
    print(rects)

    for i in range(len(rects)):
        # 获取点矩阵68*2
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个点，共68个
            cv2.circle(img, pos, 1, (0, 0, 255), -1)

            # 避免数字标签与68点重合，坐标微微移动
            pos = list(pos)
            pos[0] = pos[0] + 5
            pos[1] = pos[1] + 5
            pos = tuple(pos)

            # 利用cv2.putText输出1-68
            cv2.putText(img, str(idx + 1), pos, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.namedWindow("python_68_points", 2)
    cv2.imshow("python_68_points", img)
    cv2.waitKey(0)


# 获得人脸特征向量
def _load_known_faces(dstImgPath, mtcnn, resnet, device):
    aligned = []
    img = cv2.imread(dstImgPath)  # 读取图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)  # 检测人脸，返回一个包含人脸图像数组的结果（每个元素是一张裁剪后的人脸图像）。


    if face is not None:
        aligned.append(face[0])
        # # 显示对齐后的人脸
        # plt.imshow(face[0].permute(1, 2, 0).cpu().numpy())  # 调整维度顺序并转换为numpy
        # plt.show()
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        face_emb = resnet(aligned).detach().cpu()  # 将输入的人脸图像转换为一个高维的特征表示(特征向量)，shape (1, 512)

    # print("\n人脸对应的特征向量为：\n", known_faces_emb)
    # print("人脸对应的特征向量维度是", known_faces_emb.shape)
    return face_emb, img


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def _match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()  # $\|x-y\|_2$
    print("两张人脸的欧式距离为：%.6f" % distance)
    if distance < threshold:
        isExistDst = True
    return isExistDst


# 计算两个人脸图像的欧氏距离，判断是否为同一个人。返回值：是否为同一个人
def cal_face_distance(source_path, target_path, MatchThreshold) -> bool:
    print("设置的人脸特征向量匹配阈值为：", MatchThreshold)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # mtcnn模型加载【设置网络参数，进行人脸检测】
    mtcnn = MTCNN(min_face_size=12, thresholds=[0.2, 0.2, 0.3], keep_all=True, device=device)
    # InceptionResnetV1模型加载【用于获取人脸特征向量】
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    known_faces_emb, _ = _load_known_faces(source_path, mtcnn, resnet, device)  # 已知人物图
    faces_emb, img = _load_known_faces(target_path, mtcnn, resnet, device)  # 待检测人物图
    isExistDst = _match_faces(faces_emb, known_faces_emb, MatchThreshold)  # 人脸匹配

    if isExistDst:
        boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)  # 返回人脸框，概率，5个人脸关键点
        print('由于欧氏距离小于匹配阈值，故匹配，是同一个人')
        # print('人脸框：', boxes)
        # print('概率：', prob)
        # print('人脸关键点：', landmarks)
    else:
        print('由于欧氏距离大于匹配阈值，故不匹配，不是同一个人')
    return isExistDst


if __name__ == '__main__':
    # face_landmark_in_video(predictor_path)
    # face_landmark_in_pic(predictor_path, "./input/face_data/1.jpg")
    # face_landmark_in_pic(predictor_path, "./input/face_data/2.jpg")

    cal_face_distance('./input/face_data/1.jpg', './input/face_data/2.jpg', 0.8)
