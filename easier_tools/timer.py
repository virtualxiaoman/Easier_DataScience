import time
import numpy as np

class Timer:
    """
    [功能] 记录多次运行时间。
    [使用示例]
        timer = Timer()
          要测试时间的代码块1
        print(f'{timer.stop():.5f} sec')
          其他无关代码
        timer.start()
          要测试时间的代码块2
        print(f'{timer.stop():.5f} sec')
    [Tips]
         还可以使用：
         如：
           import cProfile
           cProfile.run('np.std(np.random.rand(1000000))')
         其中输出的参数是：
           ncalls: 函数调用的次数。
           tottime: 函数的总运行时间（不包括调用其他函数的时间），单位是秒。
           percall: 平均每次函数调用的时间，即 tottime / ncalls。
           cumtime: 函数及其所有子函数的累积运行时间（包括调用其他函数的时间），单位是秒。
           percall: 平均每次函数调用的累积时间，即 cumtime / ncalls。
           filename:lineno(function): 文件名、行号和函数名，显示函数所在的文件、行号以及函数名字。
    """
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()