from scipy.optimize import linprog

# 目标函数：最小化 x1 + x2 + ... + x7
c = [1, 1, 1, 1, 1, 1, 1]  # 系数对应 x1, x2, ..., x7

# 不等式约束：A @ x >= b 转换为 A @ x <= b 形式
# 对应于 x1 + x7 >= 20, x2 + x1 >= 20, ..., x7 + x6 >= 20
A = [
    [1, 0, 0, 0, 0, 0, 1],  # x1 + x7 >= 20
    [1, 1, 0, 0, 0, 0, 0],  # x2 + x1 >= 20
    [0, 1, 1, 0, 0, 0, 0],  # x3 + x2 >= 20
    [0, 0, 1, 1, 0, 0, 0],  # x4 + x3 >= 20
    [0, 0, 0, 1, 1, 0, 0],  # x5 + x4 >= 20
    [0, 0, 0, 0, 1, 1, 0],  # x6 + x5 >= 20
    [0, 0, 0, 0, 0, 1, 1]   # x7 + x6 >= 20
]

b = [20, 20, 20, 20, 20, 20, 20]  # 右侧常数，都是 20

# 约束条件：x1, x2, ..., x7 >= 12
x_bounds = [(12, None)] * 7  # 每个x都大于等于12

# 使用 linprog 求解
result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

# 输出结果
if result.success:
    print("最优解：", result.x)
    print("最小目标值：", result.fun)
else:
    print("优化失败:", result.message)
