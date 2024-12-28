import numpy as np
import matplotlib.pyplot as plt


# 目标函数和梯度
def objective_function(x):
    """
    目标函数: f(x, y) = x^2 + y^2 + 10 * sin(x) + 5 * cos(y)
    """
    x1, x2 = x[0], x[1]
    return x1 ** 2 + x2 ** 2 + 10 * np.sin(x1) + 5 * np.cos(x2)


def gradient(x):
    """
    目标函数的梯度: grad_f(x) = [df/dx1, df/dx2]
    """
    x1, x2 = x[0], x[1]
    df_dx1 = 2 * x1 + 10 * np.cos(x1)
    df_dx2 = 2 * x2 - 5 * np.sin(x2)
    return np.array([df_dx1, df_dx2])


def backtracking_line_search(x, grad, alpha_init=1, beta=0.8, sigma=1e-4):
    """
    回溯线性搜索
    :param x: 当前点
    :param grad: 当前点的梯度
    :param alpha_init: 初始步长
    :param beta: 步长缩小的比例
    :param sigma: 用于确保足够下降的常数
    :return: 最优步长
    """
    # 计算目标函数值
    f_x = objective_function(x)

    alpha = alpha_init
    while objective_function(x - alpha * grad) > f_x - sigma * alpha * np.dot(grad, grad):
        alpha *= beta  # 缩小步长
    return alpha


def hessian(x):
    """
    目标函数的海森矩阵: H_f(x) = [[d^2f/dx1^2, d^2f/dx1dx2], [d^2f/dx1dx2, d^2f/dx2^2]]
    """
    x1, x2 = x[0], x[1]
    d2f_dx1x1 = 2 - 10 * np.sin(x1)
    d2f_dx1x2 = 0
    d2f_dx2x2 = 2 - 5 * np.cos(x2)
    return np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx1x2, d2f_dx2x2]])


# 可视化函数
def plot_contour():
    """
    绘制目标函数的contour图，以便查看最小值、鞍点等信息
    """
    x1_vals = np.linspace(-5, 5, 100)
    x2_vals = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = X1 ** 2 + X2 ** 2 + 10 * np.sin(X1) + 5 * np.cos(X2)

    plt.contour(X1, X2, Z, 50, cmap='RdYlBu_r')
    plt.colorbar()
    plt.title("Objective Function Contour Plot")
    plt.xlabel("x1")
    plt.ylabel("x2")


# 最速下降法
def steepest_descent(starting_point, tolerance=1e-6, max_iter=1000):
    """
    最速下降法（Steepest Descent）
    """
    x = np.array(starting_point)
    iter_count = 0

    while iter_count < max_iter:
        grad = gradient(x)  # 计算梯度
        # 使用回溯线性搜索确定最优步长 alpha
        alpha = backtracking_line_search(x, grad)
        # 更新x值
        x = x - alpha * grad

        # 判断是否收敛
        if np.linalg.norm(grad) < tolerance:
            break

        iter_count += 1

    return x, iter_count


# 牛顿法
def newton_method(starting_point, tolerance=1e-6, max_iter=1000):
    """
    牛顿法（Newton's Method）
    """
    x = np.array(starting_point)
    iter_count = 0

    while iter_count < max_iter:
        grad = gradient(x)
        H = hessian(x)
        # 求解Hessian矩阵的逆
        H_inv = np.linalg.inv(H)
        # 更新规则
        step = H_inv.dot(grad)
        x = x - step

        # 如果梯度的范数小于容忍度，停止迭代
        if np.linalg.norm(grad) < tolerance:
            break

        iter_count += 1

    return x, iter_count


# BFGS算法
def bfgs_algorithm(starting_point, tolerance=1e-6, max_iter=1000):
    """
    BFGS算法（Broyden–Fletcher–Goldfarb–Shanno Algorithm）
    """
    x = np.array(starting_point)
    B = np.eye(len(x))  # 初始B矩阵为单位矩阵
    iter_count = 0

    while iter_count < max_iter:
        grad = gradient(x)
        # 使用回溯线性搜索来确定合适的步长
        alpha = backtracking_line_search(x, grad)
        # 求解B矩阵的逆和梯度
        step = -np.linalg.inv(B).dot(grad)
        x_new = x + alpha * step

        # 计算新的梯度
        grad_new = gradient(x_new)
        # 计算y和s
        y = grad_new - grad
        s = x_new - x
        # 更新B矩阵
        B = B + np.outer(y, y) / np.dot(y, s) - np.outer(np.dot(B, s), np.dot(B, s)) / np.dot(s, np.dot(B, s))

        # 更新x
        x = x_new
        # 如果梯度的范数小于容忍度，停止迭代
        if np.linalg.norm(grad_new) < tolerance:
            break

        iter_count += 1

    return x, iter_count


if __name__ == "__main__":
    # 绘制contour图
    plot_contour()
    plt.show()

    # 初始点
    for starting_point in ([3, 2], [-3, 3], [0, 0], [-2, -4], [-4, 0], [-4, 0.1]):
        print("\n" + "-" * 20)
        print(f"起始点: {starting_point}")

        # 最速下降法
        print("最速下降法:")
        result_sd, iterations_sd = steepest_descent(starting_point)
        print(f"x0, x1: {result_sd}, Iters: {iterations_sd}, F(x0, x1): {objective_function(result_sd)}")

        # 牛顿法
        print("牛顿法:")
        result_nm, iterations_nm = newton_method(starting_point)
        print(f"x0, x1: {result_nm}, Iters: {iterations_nm}, F(x0, x1): {objective_function(result_nm)}")

        # BFGS算法
        print("BFGS:")
        result_bfgs, iterations_bfgs = bfgs_algorithm(starting_point)
        print(f"x0, x1: {result_bfgs}, Iters: {iterations_bfgs}, F(x0, x1): {objective_function(result_bfgs)}")
