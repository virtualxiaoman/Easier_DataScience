import gym
import numpy as np
import random

class SimpleAgent:
    def __init__(self, env):
        pass

    # 根据给定的数学表达式进行决策
    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


def play(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励，初始化为0
    observation = env.reset()  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, done)  # 学习
        if done:  # 回合结束，跳出循环
            break
        observation = next_observation
    env.close()  # 关闭环境
    return episode_reward  # 返回回合总奖励


env = gym.make('MountainCar-v0')
env.seed(3)  # 设置随机种子，让结果可复现
agent = SimpleAgent(env)
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))

episode_reward = play(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))

episode_rewards = [play(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))


print("-" * 20)
print("以下是Q-learning的代码：")


class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.01):
        # 初始化 Q-learning 参数
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.exploration_decay = exploration_decay  # 探索率衰减
        self.min_exploration_rate = min_exploration_rate  # 最小探索率

        # 初始化 Q 表（state_space 为状态分箱后的空间）
        self.q_table = np.zeros(state_space + [action_space])

    def discretize_state(self, state, bins):
        """将连续状态空间离散化"""
        discretized = []
        for i in range(len(state)):
            discretized.append(np.digitize(state[i], bins[i]) - 1)
        return tuple(discretized)

    def choose_action(self, state, explore=True):
        """根据 epsilon-greedy 策略选择动作"""
        if explore and random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.action_space))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning 的核心更新公式"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration_rate(self):
        """探索率衰减"""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


def train_agent(episodes=10000, max_steps=200):
    env = gym.make('MountainCar-v0')
    # 将连续状态空间离散化（分成20个箱子）
    bins = [np.linspace(-1.2, 0.6, 20), np.linspace(-0.07, 0.07, 20)]

    agent = QLearningAgent(state_space=[20, 20], action_space=env.action_space.n)

    for episode in range(episodes):
        state = env.reset()
        state = agent.discretize_state(state, bins)

        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            next_state = agent.discretize_state(next_state, bins)

            # 更新 Q 表
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        # 探索率衰减
        agent.decay_exploration_rate()

        # 打印每1000回合的结果
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # 训练完成后展示一次可视化
    visualize_agent(env, agent, bins, max_steps)
    env.close()


def visualize_agent(env, agent, bins, max_steps):
    """展示训练后的 agent 运行效果"""
    state = env.reset()
    state = agent.discretize_state(state, bins)

    for step in range(max_steps):
        env.render()  # 可视化环境
        action = agent.choose_action(state, explore=False)  # 使用训练后的策略，不进行探索
        next_state, _, done, _ = env.step(action)
        state = agent.discretize_state(next_state, bins)

        if done:
            break

    env.close()

train_agent()
