import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


def fix_img(img0, detect, device, threshold=0.5, erosion=2, dilation=6, mask_only=False):
    """
    修复图像中的水印区域。

    :param img0: PIL Image，输入的图像对象。
    :param detect: torch.nn.Module，用于检测水印区域的模型。
    :param device: torch.device, 模型所在的设备。
    :param threshold: float [0, 1]，模型输出的阈值。
                       - 小于0表示对整张图进行重绘；
                       - 0表示在模型不确定为水印的位置进行重绘；
                       - 0.1表示只在模型确信为水印的位置进行重绘。
    :param erosion: int, 表示腐蚀操作的结构元素大小（>=0），去除噪点。
    :param dilation: int, 表示膨胀操作的结构元素大小（>=0），扩大处理区域。
    :param mask_only: bool, 如果为True则返回掩码图（灰度图），否则返回修复后的RGB图。

    :return: 修复后的图像（PIL Image），或仅返回掩码图。
    """
    # 将输入图像转换为RGB格式，并转换为NumPy数组
    img = img0.convert('RGB')
    npa = np.array(img)

    # 取出蓝色通道（假设模型主要基于蓝色通道进行水印检测）
    blue = npa[:, :, 2]
    blue_torch = torch.tensor(blue, dtype=torch.float32, device=device)

    # 使用模型检测水印区域，输入需要增加一个batch维度
    # 注意：模型输出的是一个mask（概率图），需要从GPU上取回并转换为NumPy数组
    mask = detect(blue_torch.unsqueeze(0)).cpu().detach().squeeze().numpy()

    # 根据阈值将概率图转换为二值掩码：
    # 小于阈值的部分设为0，其余部分设为1
    mask = np.where(mask < threshold, 0, 1)

    # 如果erosion参数大于1，则应用二值腐蚀操作以减少噪点
    if erosion > 1:
        # 构造形态学腐蚀的结构元素（全1矩阵，尺寸为erosion x erosion）
        struct_elem = np.ones((erosion, erosion), dtype=bool)
        mask = binary_erosion(mask, structure=struct_elem, iterations=1)

    # 如果dilation参数大于0，则应用二值膨胀操作以扩展处理区域
    if dilation > 0:
        # 构造形态学膨胀的结构元素（全1矩阵，尺寸为dilation x dilation）
        struct_elem = np.ones((dilation, dilation), dtype=bool)
        mask = binary_dilation(mask, structure=struct_elem, iterations=1)

    # 将掩码反转：原先值为1的区域（检测出的水印区域）变为0，这样绘制mask的时候这部分区域是黑色的
    # 而其他区域变为1，以便后续保留原图像中非水印部分
    mask = 1 - mask

    # 如果只需要返回掩码，则将二值掩码转换为灰度图（0~255）并返回
    if mask_only:
        return Image.fromarray((mask * 255).astype(np.uint8), 'L')

    # 将掩码扩展为与原图同样的通道数（利用广播，直接进行逐像素乘法），使得水印区域为黑色
    fixed = npa * mask[:, :, np.newaxis]  # mask从(height, width)扩展为(height, width, 1)

    # 将结果转换为无符号8位整型并转换为PIL Image对象返回
    return Image.fromarray(fixed.astype(np.uint8), 'RGB')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载预训练的检测模型（用于检测水印区域）
    detect = torch.load('detV4.pth').to(device)
    # 打开样本图像（确保路径正确）
    img = Image.open('sample.png')
    # 调用fix_img函数，mask_only=True表示只返回掩码图，否则返回合并的图像
    fixed = fix_img(img, detect, device, threshold=0.3, erosion=1, dilation=2, mask_only=True)
    fixed.save('mask.jpg')
    # 后续需要sd重绘，可能是参考的BV1iM4y1y7oA，这里使用cv2.inpaint()函数进行重绘，但是效果不好
    # 读取输入图像和掩膜
    image = cv2.imread('sample.png')  # 输入图片路径
    mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)  # 单通道掩膜
    # 反转掩膜：将水印区域设置为255，背景设置为0
    mask = cv2.bitwise_not(mask)

    # 检查图像和掩膜尺寸是否一致
    assert image.shape[:2] == mask.shape, "图像和掩膜尺寸不匹配！"

    # 选择修复算法（二选一）
    # method = cv2.INPAINT_TELEA  # 快速行进法（速度快）
    method = cv2.INPAINT_NS    # Navier-Stokes 流体动力学（效果更平滑）

    # 设置修复半径（控制修复区域周围的影响范围）
    inpaint_radius = 3  # 通常为 1~10，根据水印大小调整

    # 执行修复
    result = cv2.inpaint(image, mask, inpaint_radius, flags=method)

    # 保存结果
    cv2.imwrite('result.png', result)