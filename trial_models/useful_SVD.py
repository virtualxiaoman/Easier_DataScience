import cv2
import numpy as np
import matplotlib.pyplot as plt

from easier_excel.draw_data import draw_density, plot_xy
from easier_excel.read_data import read_image

def zip_image_by_svd(origin_image, rate=0.8, channel=3, show_img=True):
    """
    使用SVD对图像进行压缩
    :param origin_image: 传入np.ndarray类型的图像数组
    :param rate: 保留率
    :param channel: 通道数，灰度图像为1，彩色图像为3
    :param show_img: 是否显示压缩前后的图片
    :return zip_img: 压缩后的图像
    """
    zip_img = np.zeros(origin_image.shape)  # 用于存储压缩后的图像
    u_shape, s_shape, vT_shape, n_sigmas = 0, 0, 0, 0

    for chan in range(channel):
        # 对每层进行SVD分解
        U, Sigma, VT = np.linalg.svd(origin_image[:, :, chan])
        # 计算达到保留率需要的奇异值数量
        total_Sigma = np.sum(Sigma)
        cum_Sigma = np.cumsum(Sigma)
        n_sigmas = np.argmax(cum_Sigma >= rate * total_Sigma) + 1
        Sigma_n = np.diag(Sigma[:n_sigmas])
        zip_img[:, :, chan] = np.dot(U[:, :n_sigmas], np.dot(Sigma_n, VT[:n_sigmas, :]))
        # 记录每个矩阵的shape
        u_shape = U[:, 0:n_sigmas].shape
        s_shape = Sigma_n.shape
        vT_shape = VT[0:n_sigmas, :].shape

    # 这里暂时没想到更好的方法，应该是让zip_img更接近origin_image的值，使得颜色差异小
    # 如果使用zip_img就是归一化到[0, 1]，但这里使用origin_image的最大最小值来进行归一化，是为了减少颜色差异
    # 使用Z_MIN = np.min(zip_img[:, :, i])    Z_MAX = np.max(zip_img[:, :, i])
    # zip_img[:, :, i] = (O_MAX-O_MIN) * (zip_img[:, :, i] - Z_MIN) / (Z_MAX - Z_MIN) + O_MIN
    # 可能不如只使用O_MAX和O_MIN，应该是因为Z_MIN和Z_MAX的值是离群点
    for i in range(channel):
        O_MAX = np.max(origin_image[:, :, i])
        O_MIN = np.min(origin_image[:, :, i])
        zip_img[:, :, i] = (zip_img[:, :, i] - O_MIN) / (O_MAX - O_MIN)
    zip_img[zip_img < 0] = 0
    zip_img[zip_img > 1] = 1

    # # 因为数据支持RGB data ([0..1] for floats or [0..255] for integers，所以不一定需要调整到[0, 255]
    # zip_img = np.round(zip_img * 255).astype('int')

    # 计算压缩率
    zip_rate = (origin_image.size - 3 * (u_shape[0] * u_shape[1] + s_shape[0] * s_shape[1] + vT_shape[0] * vT_shape[1])) \
        / origin_image.size

    print(f"保留率：         {rate * 100:.1f}%")
    print(f"选择的奇异值数量：{n_sigmas}--->原来的奇异值数量: {Sigma.shape[0]}")
    print(f"原图Shape：      {origin_image.shape}--->Size: {origin_image.size}", )
    print(f"压缩后的矩阵大小：{u_shape} , {s_shape} , {vT_shape}")
    print(f"压缩率为：       {zip_rate * 100:.3f}%")
    if show_img:
        fig, axes = plt.subplots(1, 2)
        if channel == 1:
            axes[0].imshow(origin_image[:, :, 0], cmap='gray')
            axes[1].imshow(zip_img[:, :, 0], cmap='gray')
        else:
            axes[0].imshow(origin_image)
            axes[1].imshow(zip_img)
        axes[0].set_title('Before SVD')
        axes[1].set_title(f'After SVD with rate={zip_rate * 100:.3f}% and n_sigmas={n_sigmas}')
        plt.show()
    return zip_img


path = '../output/WhiteAndYellow.jpg'
img_gray = read_image(path, gray_pic=False, show_details=False)
# img_gray = img_gray.reshape((img_gray.shape[0], img_gray.shape[1], 1))
zip_image_by_svd(img_gray, rate=0.6, channel=3)
