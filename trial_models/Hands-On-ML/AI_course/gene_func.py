import numpy as np
import random
import matplotlib.pyplot as plt

SELECT_WAY = 'Elitism'  # 选择方式：'roulette_wheel' 轮盘赌 or 'Elitism' 精英选择


# 目标函数
def objective_function(x):
    return 10 * np.sin(5 * x) + 7 * np.abs(x - 5) + 10


# 初始化种群
def initialize_population(pop_size, x_min, x_max):
    return np.random.uniform(x_min, x_max, pop_size)


# 选择操作
def select_population(population, fitness, num_select):
    if SELECT_WAY == 'Elitism':
        selected_indices = np.argsort(fitness)[:num_select]
        return population[selected_indices]
    elif SELECT_WAY == 'roulette_wheel':
        fitness_sum = np.sum(fitness)
        selection_probs = fitness / fitness_sum  # 每个个体被选择的概率
        selected_indices = np.random.choice(len(population), size=num_select, p=selection_probs)
        return population[selected_indices]


# 交叉操作
def crossover(population, crossover_rate):
    new_population = []
    for i in range(0, len(population), 2):
        parent1 = population[i]
        parent2 = population[i + 1] if i + 1 < len(population) else population[i]
        if random.random() < crossover_rate:
            # 单点交叉
            crossover_point = random.uniform(0, 1)  # 不妨记为alpha
            child1 = crossover_point * parent1 + (1 - crossover_point) * parent2  # c1 = alpha * p1 + (1 - alpha) * p2
            child2 = (1 - crossover_point) * parent1 + crossover_point * parent2  # c2 = (1 - alpha) * p1 + alpha * p2
            new_population.append(child1)
            new_population.append(child2)
        else:
            new_population.append(parent1)
            new_population.append(parent2)
    return np.array(new_population)


# 变异操作
def mutate(population, mutation_rate, x_min, x_max, generation, max_generations):
    for i in range(len(population)):
        if random.random() < mutation_rate:
            # mutation_value = random.uniform(-0.1, 0.1)  # 控制变异幅度，不妨记为delta
            mutation_range = (x_max - x_min) * (1 - generation / max_generations)  # 逐步减少变异幅度
            mutation_value = random.uniform(-mutation_range, mutation_range)
            population[i] = np.clip(population[i] + mutation_value, x_min, x_max)  # x'=x+delta，然后限制在x固有的范围内
    return population


# 遗传算法主函数
def genetic_algorithm(pop_size, generations, x_min, x_max, crossover_rate, mutation_rate):
    """
    遗传算法
    :param pop_size: 种群大小
    :param generations: 迭代代数
    :param x_min: 最小值，这里是0
    :param x_max: 最大值，这里是10
    :param crossover_rate: 交叉概率
    :param mutation_rate: 变异概率
    :return:
    """
    population = initialize_population(pop_size, x_min, x_max)
    print(f"初始化种群: {population}")

    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for generation in range(generations):
        # 计算适应度，也就是目标函数值
        fitness = np.array([objective_function(x) for x in population])

        # 记录最优解，也就是使得目标函数值最小的解
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]  # 将最小的目标函数值赋值给best_fitness
            best_solution = population[min_fitness_idx]  # 将最小目标函数值对应的解赋值给best_solution

        # 选择操作，选择种群大小的一半的个体
        selected_population = select_population(population, fitness, pop_size // 2)
        # 交叉操作
        population = crossover(selected_population, crossover_rate)
        # 变异操作
        population = mutate(population, mutation_rate, x_min, x_max, generation, generations)

        fitness_history.append(best_fitness)
        print(f"本次是第{generation}/{generations}代, 最优解是: {best_solution}, 最优值是: {best_fitness}")

    return best_solution, best_fitness, fitness_history, population, fitness


# 设置参数
pop_size = 100  # 种群大小
generations = 100  # 迭代代数
x_min = 0  # x 最小值
x_max = 10  # x 最大值
crossover_rate = 0.1  # 交叉概率
mutation_rate = 0.9  # 变异概率

# 运行遗传算法
best_x, best_y, fitness_history, population, fitness = (
    genetic_algorithm(pop_size, generations, x_min, x_max, crossover_rate, mutation_rate))

# 输出结果
print(f"Best x: {best_x}, Best y: {best_y}")

# 绘制适应度变化图
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness History')
plt.show()

plt.scatter(population, fitness, c=fitness, cmap='viridis')
plt.xlabel('Population')
plt.ylabel('Fitness')
plt.title(f'Population Distribution')
plt.colorbar(label='Fitness')
plt.show()

# 展示objective_function
x_vals = np.linspace(x_min, x_max, 100)
y_vals = np.array([objective_function(x) for x in x_vals])
plt.plot(x_vals, y_vals)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Function')
plt.show()
