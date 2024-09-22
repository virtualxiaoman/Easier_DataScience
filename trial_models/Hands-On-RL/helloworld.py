import gym

env = gym.make('CartPole-v1', render_mode='human')  # 构建实验环境
env.reset()  # 重置一个回合
for _ in range(200):
    # env.render()  # 显示图形界面。指定 render_mode='human' 参数。就不需要在循环中多次调用 render() 方法。
    action = env.action_space.sample()  # 从动作空间中随机选取一个动作
    observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    print(observation)
env.close()  # 关闭环境
