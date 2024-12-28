import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 定义一个函数来生成权重矩阵
def generate_weight_matrix(input_array, size):
    """
    生成Hopfield网络的权重矩阵
    input_array: 输入模式的列表
    size: 输入模式的大小 (例如 6x6 矩阵的 36)
    """
    w = np.zeros((size, size))  # 初始权重矩阵
    for s in input_array:
        w0 = np.zeros((size, size))  # 临时矩阵
        for i in range(size):
            for j in range(size):
                if i != j:
                    w0[i, j] = (2 * s[i] - 1) * (2 * s[j] - 1)  # 更新权重
        w += w0
    w *= 300  # 放大权重矩阵的值，使其能够以图像方式展示
    return w


# 定义一个函数来运行Hopfield网络并恢复图像
def run_hopfield_network(w, initial_state, size, iterations=10):
    """
    使用Hopfield网络从初始状态恢复图像
    w: 权重矩阵
    initial_state: 初始状态（通常为有噪声的图像）
    size: 输入模式的大小
    iterations: 迭代次数
    """
    v0 = initial_state  # 设置初始状态
    Y = np.zeros(size)  # 输出状态初始化为全零
    for t in range(iterations):
        v1 = np.zeros(size)  # 临时状态
        for j in range(size):
            v1[j] = np.dot(w[j, :], v0)  # 计算新的状态
            Y[j] = 1 if v1[j] >= 0 else 0  # 激活函数，负值归零，非负值归一
        v0 = Y  # 更新状态
    return Y


# 训练模式a和b
a = np.array([[0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0]])
b = np.array([[0, 0, 1, 1, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 1, 0, 0]])

# 将a和b展平成一维数组
array_a = a.flatten()
array_b = b.flatten()
input_array = [array_a, array_b]

# 生成Hopfield网络的权重矩阵
w = generate_weight_matrix(input_array, 36)

# 可视化权重矩阵
w_image = Image.fromarray(w)
plt.imshow(w_image)
plt.title('Hopfield Network Weight Matrix')
plt.show()

# 使用c图像（噪声图像）来恢复（去噪）为a图像
c = np.array([[0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0]])

c_flattened = c.flatten()  # 展平图像
result = run_hopfield_network(w, c_flattened, 36)  # 恢复图像

# 将恢复后的图像转换为6x6矩阵并显示
result_image = np.array(result).reshape(6, 6)
result_image = Image.fromarray(result_image * 600)  # 放大显示
plt.imshow(result_image)
plt.title('Restored Image by Hopfield Network')
plt.show()
