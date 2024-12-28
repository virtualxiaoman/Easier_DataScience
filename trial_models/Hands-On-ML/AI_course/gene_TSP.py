import numpy as np
import random
import matplotlib.pyplot as plt

# 随机生成20个城市的坐标
City_Map = 100 * np.random.rand(20, 2)


# 计算路径总长度
def calculate_path_length(path, city_map):
    length = 0
    for i in range(len(path) - 1):
        city1 = city_map[path[i]]
        city2 = city_map[path[i + 1]]
        length += np.linalg.norm(city1 - city2)  # 假设使用欧几里得距离
    length += np.linalg.norm(city_map[path[-1]] - city_map[path[0]])  # 旅行社还要返回起点城市
    return length


# 初始化种群，使用随机排列的城市序列
def generate_initial_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        path = list(range(num_cities))
        random.shuffle(path)
        population.append(path)
    return population


# 选择操作
def selection(population, city_map, method='Elitism'):
    if method == 'roulette_wheel':
        # 轮盘赌选择
        fitness_scores = [1 / calculate_path_length(path, city_map) for path in population]
        total_fitness = sum(fitness_scores)
        probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        selected_population = random.choices(population, probabilities, k=len(population) // 2)

    elif method == 'Elitism':
        # 精英选择：选择适应度最好的前50%个体
        population_sorted = sorted(population, key=lambda path: calculate_path_length(path, city_map))
        selected_population = population_sorted[:len(population) // 2]

    else:
        raise ValueError(f"Invalid selection method: {method}")

    return selected_population


# 交叉操作
def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted([random.randint(0, size - 1), random.randint(0, size - 1)])
    child1 = [-1] * size  # -1表示未填充
    child2 = [-1] * size

    # 复制子区间
    child1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    child2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]

    # 填充剩余部分，保持城市的唯一性
    for i in range(size):
        if parent2[i] not in child1:
            for j in range(size):
                if child1[j] == -1:
                    child1[j] = parent2[i]
                    break
        if parent1[i] not in child2:
            for j in range(size):
                if child2[j] == -1:
                    child2[j] = parent1[i]
                    break

    return child1, child2


# 变异操作
def mutation(path):
    size = len(path)
    i, j = random.sample(range(size), 2)
    path[i], path[j] = path[j], path[i]  # 随机交换两个城市的位置
    return path


# 遗传算法求解TSP问题
def genetic_algorithm(city_map, pop_size=100, generations=100, mutation_prob=0.2, crossover_prob=0.7,
                      selection_method='roulette_wheel'):
    num_cities = len(city_map)
    population = generate_initial_population(pop_size, num_cities)
    best_path = None
    best_length = float('inf')

    for gen in range(generations):
        # 选择操作
        selected_population = selection(population, city_map, method=selection_method)
        # 交叉操作
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 变异操作
        for i in range(len(new_population)):
            if random.random() < mutation_prob:
                new_population[i] = mutation(new_population[i])

        # 更新种群
        population = selected_population + new_population

        # 记录最优路径
        for path in population:
            path_length = calculate_path_length(path, city_map)
            if path_length < best_length:
                best_path = path
                best_length = path_length

        print(f"Generation {gen}: Best length = {best_length}")

    return best_path, best_length


# 可视化路径，使用有向箭头
def plot_best_path(city_map, best_path):
    plt.scatter(City_Map[:, 0], City_Map[:, 1], color='red', label='Cities')

    path_coords = city_map[best_path]
    # 对起点和终点这两个点使用特殊标记
    plt.scatter(path_coords[0][0], path_coords[0][1], color='green', label='Start', s=100)
    plt.scatter(path_coords[-1][0], path_coords[-1][1], color='orange', label='End', s=100)

    for i in range(len(path_coords) - 1):
        start = path_coords[i]
        end = path_coords[i + 1]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=3, head_length=4, fc='blue', ec='blue')

    # 从最后一个城市返回起点
    start = path_coords[-1]
    end = path_coords[0]
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_width=1, head_length=2, fc='blue', ec='blue')

    plt.legend()
    plt.show()


# 设置参数
mutation_prob = 0.2
crossover_prob = 0.7
selection_method = 'Elitism'  # 或 'roulette_wheel'

# 运行遗传算法
best_path, best_length = genetic_algorithm(City_Map, mutation_prob=mutation_prob, crossover_prob=crossover_prob,
                                           selection_method=selection_method, generations=500)

# 输出结果
print(f"Best path: {best_path}")
print(f"Total path length: {best_length}")

# 可视化路径
plot_best_path(City_Map, best_path)
