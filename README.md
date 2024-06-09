该项目主要是`easier_excel`与`easier_nn`，各个项目的功能如下：
```
easier_excel: 用机器学习处理表格类型的数据
easier_nn:    使用神经网络进行计算机视觉(CV)和自然语言处理(NLP)等任务
easier_nlp:   现在不想搞了
easier_tools: 是一些小工具，比如计时，彩色输出，查看函数、类的参数
trial_models: 是一些机器学习算法的应用尝试或手动实现
```

## 💦一、Quick Start：
运行对应项目的示例文件即可:

`easier_excel` 项目里的`example1.py`提供了Quick Start

`easier_nn`项目内的`example1.py`,`example2.py`提供了Quick Start

你也可以根据我写在下面的[项目功能](#jump_3)来查看一些具体的用法。



## 🍴二、项目结构
项目的结构如下：

`[·]`代表正在开发进行中,`[x]`代表废弃或用处不大的项目,`[√]`代表基本完成的项目，`[🈵]`代表我喜欢的项目。

```
.
├── easier_excel                # [·] 用机器学习处理表格类型的数据
│   ├── __init__.py             # [√] init
│   ├── cal_data.py             # [·] 更轻松地调用机器学习的包
│   ├── draw_data.py            # [🈵] 绘图
│   └── math_formula.py         # [·] 计算积分，导数，求解最优化问题等
│   └── read_data.py            # [🈵] 读取、描述、预处理数据
│   └── example1.py             # [√] easier_excel的Quick Start
|
├── easier_nn                   # [·] 使用神经网络进行计算机视觉(CV)和自然语言处理(NLP)等任务
│   ├── __init__.py             # [√] init
│   ├── calculate_shape.py      # [√] 计算数据的shape，可以将网络传入，然后查看每一层的shape
│   ├── classic_datasets.py     # [🈵] 一些经典的数据集
│   ├── evaluate_net.py         # [·] 评估模型
│   └── load_data.py            # [√] 加载数据
│   └── train_net.py            # [🈵] 模型训练
│   └── example1.py             # [√] easier_nn的Quick Start
│   └── example2.py             # [√] easier_nn的Quick Start
|
├── easier_nlp                  # [×] 使用神经网络进行自然语言处理(NLP)等任务
│   ├── __init__.py             # [√] init
│   ├── preprocessing.py        # [·] 文本预处理
│   ├── spacy_nlp.py            # [×] 使用spacy进行NLP
│   └── example.py              # [√] easier_nlp的Quick Start
|
├── easier_tools                # [·] 一些小工具
│   ├── Colorful_Console.py     # [√] 彩色输出
│   ├── easy_count.py           # [×] 统计小工具，目前只有统计词频的功能
│   ├── print_variables.py      # [√] 查看函数、类的参数
│   └── timer.py                # [√] 计时器
|
├── trial_models                # [·] 一些机器学习算法的应用尝试或手动实现
│   ├── Cellular_Automations.py # [×] 元胞自动机
│   ├── FM.py                   # [√] 因子分解机
│   ├── MF.py                   # [√] 矩阵分解
│   ├── KNN.py                  # [√] KNN算法
│   ├── useful_eig.py           # [√] 特征值分解
│   ├── useful_SVD.py           # [√] SVD分解
```

我个人比较喜欢`easier_excel`和`trial_models`，其余的有空了然后写写。



## <span id="jump_3">📖三、项目功能</span>
这里只给出关键部分的功能与代码示例。
### 🥵3.1 easier_excel
构式机器学习天天都是一样的代码（点名批评plt），给她点颜色看看，让她自己跑去吧。
#### 3.1.1 read_data
能够自动读入csv,xlsx,xls,sav四种格式，示例代码如下：
``` python
from easier_excel import read_data
path = "你的数据.csv"  # 相对路径或绝对路径都可
df = read_data.read_df(path)  # 返回的df就是dataframe类型的数据
```
能够自动描述数据（各种统计量，是否有缺失值），示例代码如下：
``` python
desc = read_data.desc_df(df)
desc.describe()  # 描述数据
desc.fill_missing_values(fill_type='mean')  # 填充缺失值
```
#### 3.1.2 draw_data
能够自动绘制直线图，corr，特征重要性图，散点图，密度曲线图，示例代码如下：
```Python
from easier_excel import draw_data
draw_df = draw_data.draw_df(df)
draw_df.draw_corr(save_path='../output', v_minmax=(-1, 1), show_plt=True)  # 绘制相关性矩阵
draw_df.draw_all_scatter(target_name='某个关键属性', save_path='../output/scatters')  # 绘制散点图
for feature_name in ['其他属性1', '其他属性2', '其他属性3']:
    draw_df.draw_density(target_name="某个关键属性", feature_name=feature_name, show_plt=False, save_path='../output/density')  # 绘制密度曲线图
draw_df.draw_feature_importance(target_name='某个关键属性', save_path='../output', show_plt=False)  # 用随机森林来求解特征的重要性
```

### 😍3.2 easier_nn
请见`easier_nn`的Quick Start，我懒得写。

### 😭3.3 easier_nlp
摆了

### 🥰3.4 easier_tools
#### 3.4.1 print_variables
查看类的某个/某些参数，示例代码如下：
``` python
import torch.nn as nn
from easier_tools.print_variables import print_variables_class
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
print_variables_class(rnn, specific_param=["_all_weights"])  # 查看所有_all_weights(RNN的权重)
```
查看函数的参数是类似的，示例代码如下：
``` python
from easier_tools.print_variables import print_variables_function
print_variables_function(desc.describe_df, show_stats=True, stats_T=False)  # desc的定义请见3.1.1
```
#### 3.4.2 timer
计时器，示例代码如下：
``` python
from easier_tools.timer import Timer
timer = Timer()
# 要测试时间的代码块1
print(f'{timer.stop():.5f} sec')  # 结束计时
# 其他无关代码
timer.start()  # 重新开始计时
# 要测试时间的代码块2
print(f'{timer.stop():.5f} sec')  # 结束计时
```

### 😋 3.5 trial_models
代码比较独立，运行对应的就可以。


### 🔗 3.6 如何使用注释
如需查看**具体的函数/类**的使用方法，需要查看具体代码。一般而言，代码注释如下，代码正在逐渐重构中，因此不一定都有这么详细:

``` python
def fill_missing_values(self, fill_type='mean'):
    """
    填充缺失值
    [Warning]:
        该操作会直接在原数据上进行修改，也就是修改self.df
    [使用方法]:
        desc = read_data.desc_df(df)
        desc.fill_missing_values(fill_type=114514)  # 实际填充的时候可别逸一时误一世了
    [Tips]:
        目前支持的填充类型有：'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)'。
        如需删除缺失值请使用delete_missing_values。
    :param fill_type: 填补类型，支持 'mean', 'median', 'most_frequent', 'constant(直接填入具体数值)'
    """
    pass
```
其中`Warning`是对使用此函数时的一些警告，`使用方法`是函数的使用方法，`Tips`是对函数的一些小提示（比如其他的可能用法），`Bug`是还未修复的一些问题，`todo`是还未完成的任务。


## 🤔四、未来更新方向
主要还是机器学习。神经网络实在是💩，我训练不了。

关于代码其余的具体描述，暂时没时间写描述┭┮﹏┭┮

## ✨五、星野天下第一
小鸟游星野适合结婚的十个理由：
1. 星野虽然外表看起来像小学五年级的学生，实际上的年龄已经接近成年，可以放心大胆地等她到法定结婚年龄。
2. 因为星野小小的，所以能毫不费劲地抱在自己腿上，提前享受养女儿的快乐。
3. 武力值爆表，阿拜多斯扛把子，是能让大学园的执法组织头目忌惮的存在，娶回家有满满的安全感。某种意义上来说也可以算得上是联姻（？）
4. 经过为师的不懈努力，星野从讨厌大人转变到整天黏着为师，这份改变足以说明为师在她心目中的地位。不怕婚后二人整日争吵。
5. 虽然经历了很多，但还是保有少女心，留有自己的一份底线。坚强又可爱的星野，有几个人不喜欢呢。
6. 经历过还债地狱，知道钱来得不容易，不会乱消费，甚至还会精打细算。这么一个勤俭持家的小姑娘，难道不想娶回家吗？
7. 拥有非常正直的三观，以后有了孩子一定能成为一位好妈妈。
8. 有责任心，可以肩负起身为人妻的职责。
9. 我是萝莉控。
10. 星野！我的小星野！