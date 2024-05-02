import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update(frameNum, img, grid, N, ON=255, OFF=0):
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            # 计算每个细胞周围活细胞的数量
            total = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
                         grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
                         grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
                         grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N])
                        / 255)
            # 根据规则更新细胞状态
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON
    # 更新图像
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img


# 初始化参数
N = 100
ON = 255
OFF = 0
prob = 0.1
grid = np.random.choice([ON, OFF], N * N, p=[prob, 1 - prob]).reshape(N, N)
print(grid)


# 创建动画
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='gray')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N), frames=10)
plt.show()
