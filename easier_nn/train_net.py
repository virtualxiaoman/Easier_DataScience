import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from easier_nn.load_data import load_array
from easier_nn.evaluate_net import evaluate_accuracy, count_correct_predictions, draw_Loss_or_Accuracy, \
    draw_Loss_or_Accuracy_immediately


class NetTrainer:
    """
    这是一个简易的nn训练器。

    你可以使用下面的代码快速上手：

    [快速上手-1.回归]:
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import torch

        from easier_nn.train_net import NetTrainer

        np.random.seed(42)
        data = np.random.rand(1000, 10)
        target = np.sum(data, axis=1) + np.random.normal(0, 0.1, 1000)  # target is sum of features with some noise

        class RegressionNet(nn.Module):
            def __init__(self):
                super(RegressionNet, self).__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # 创建模型、损失函数和优化器
        net = RegressionNet()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # 训练模型
        trainer = NetTrainer(data, target, net, loss_fn, optimizer, epochs=200)
        trainer.train()
        trainer.evaluate()

        # 查看模型的层与参数(以train_loss_list为例)
        # nn.Module 对象不能直接进行迭代，需要通过访问它的 modules() 或 children() 方法来迭代它的层。
        # modules() 方法返回模块和它所有的子模块，而 children() 方法仅返回模块的直接子模块。
        for layer in net.children():
            print(layer)
            # if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            #     print('Weight:', layer.weight)
            #     print('Bias:', layer.bias)
            # print('-----------------')
        print(trainer.train_loss_list)

    [快速上手-分类，请确保使用了参数net_type="acc"]:
        print("--------分类---------")
        # 生成分类数据
        np.random.seed(42)
        data = np.random.rand(1000, 20)  # 20个特征
        target = (np.sum(data, axis=1) > 10).astype(int)  # 如果特征和大于10，则类别为1，否则为0

        class ClassificationNet(nn.Module):
            def __init__(self):
                super(ClassificationNet, self).__init__()
                self.fc1 = nn.Linear(20, 50)
                self.fc2 = nn.Linear(50, 2)  # 2个类别

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        net = ClassificationNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # 训练模型
        trainer = NetTrainer(data, target, net, loss_fn, optimizer, epochs=50, net_type="acc")
        trainer.train()
        trainer.evaluate()

        # 查看模型的层与参数
        for layer in net.children():
            print(layer)
        print(trainer.test_acc_list)
    """
    def __init__(self, data, target, net, loss_fn, optimizer,
                 test_size=0.2, batch_size=32, epochs=100,
                 net_type="loss", print_interval=20,
                 device=None):
        """
        初始化模型
        分类时应当确保数据的形状是 X:(batch, features), y(batch)，不然使用CrossEntropyLoss会报错。
        :param data: 数据，X
        :param target: 目标，y
        :param net: pytorch网络，需要继承nn.Module
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param test_size: 测试集大小，支持浮点数或整数
        :param batch_size: 批量大小
        :param epochs: 训练轮数
        :param net_type: 模型类型，只可以是"loss"(回归-损失)或"acc"(分类-准确率)
        :param print_interval: 打印间隔，请注意train_loss_list等间隔也是这个
        :param device: 设备，支持"cuda"或"cpu"，默认为None，自动有限选择cuda
        """
        # 设备参数
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 数据参数
        self.data = data  # X
        self.target = target  # y
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.train_loader, self.test_loader = None, None
        # 网络参数
        self.net = net.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # 训练参数
        self.batch_size = batch_size
        self.epochs = epochs
        # 训练输出参数
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []
        self.print_interval = print_interval  # 打印间隔
        # 使用loss还是acc参数
        self.net_type = net_type

        # 初始化
        self.init_loader()

    # [init]初始化训练数据
    def init_loader(self):
        # 检查self.data与self.target的类型，如果不是dataframe则转换为dataframe
        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data)
        if not isinstance(self.target, pd.DataFrame):
            self.target = pd.DataFrame(self.target)

        # 检查self.data与self.target的shape是否一致
        if self.data.shape[0] != self.target.shape[0]:
            raise ValueError(f"data和target的shape[0]不相同: "
                             f"data({self.data.shape[0]}) and target({self.target.shape[0]})")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target,
                                                                                test_size=self.test_size)

        # 创建DataLoaders
        self.train_loader = self.create_dataloader(self.X_train, self.y_train)
        self.test_loader = self.create_dataloader(self.X_test, self.y_test, train=False)
        # 将数据变成tensor，并且dtype依据data的类型而定
        self.X_train = self.dataframe_to_tensor(self.X_train)
        self.X_test = self.dataframe_to_tensor(self.X_test)
        self.y_train = self.dataframe_to_tensor(self.y_train)
        self.y_test = self.dataframe_to_tensor(self.y_test)

        # 对于分类任务，如果y的维度超过了1，就reshape(-1,)
        if self.net_type == "acc":
            # print(self.y_train)
            if self.y_train.dim() > 1:
                self.y_train = self.y_train.view(-1)
            if self.y_test.dim() > 1:
                self.y_test = self.y_test.view(-1)

        # 将数据转移到设备上
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(self.device), self.X_test.to(
            self.device), self.y_train.to(self.device), self.y_test.to(self.device)

    # [子函数]创建dataloader
    def create_dataloader(self, data, target, train=True):
        # dtype依据data的类型而定
        data = self.dataframe_to_tensor(data)
        target = self.dataframe_to_tensor(target)
        # 对于分类任务，如果y的维度超过了1，就reshape(-1,)
        if self.net_type == "acc":
            # print(self.y_train)
            if target.dim() > 1:
                target = target.view(-1)

        # print(target)
        dataset = TensorDataset(data, target)
        if train:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.net.train()
        for epoch in range(self.epochs):
            loss_sum = 0.0
            for X, y in self.train_loader:
                # 初始化数据
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # 前向传播
                outputs = self.net(X)
                # print(X.shape, y.shape, outputs.shape)
                loss = self.loss_fn(outputs, y)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 计算损失
                loss_sum += loss.item()
            if epoch % self.print_interval == 0:
                if self.net_type == "loss":
                    self.train_loss_list.append(loss_sum / len(self.train_loader))
                    self.test_loss_list.append(self.evaluate())
                    print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {self.train_loss_list[-1]}')
                elif self.net_type == "acc":
                    self.train_acc_list.append(self.evaluate())  # todo 该功能还未实现
                    self.test_acc_list.append(self.evaluate())
                    print(f'Epoch {epoch + 1}/{self.epochs}, Accuracy: {self.train_acc_list[-1]}')
                else:
                    raise ValueError("net_type must be 'loss' or 'acc'")

    def evaluate(self):
        self.net.eval()
        if self.net_type == "loss":
            loss_test = 0.0
            self.net.train()
            return self.loss_fn(self.net(self.X_test), self.y_test).item()
        elif self.net_type == "acc":
            predictions = torch.argmax(self.net(self.X_test), dim=1).type(self.y_test.dtype)
            correct = (predictions == self.y_test).sum().item()
            n = self.y_test.numel()
            acc = correct / n
            self.net.train()
            return acc
        # total, correct = 0, 0
        # with torch.no_grad():
        #     for inputs, labels in self.test_loader:
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = self.net(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # print(f'Accuracy: {100 * correct / total}%')

    # 将df转换为tensor，并保持数据类型的一致性
    @staticmethod
    def dataframe_to_tensor(df, float_dtype=torch.float32, int_dtype=torch.int64):
        """
        PyTorch's tensors are homogenous, ie, each of the elements are of the same type.
        将df转换为tensor，并保持数据类型的一致性
        :param df: pd.DataFrame
        :param float_dtype: torch.dtype, default=torch.float32
        :param int_dtype: torch.dtype, default=torch.int32
        :return: torch.Tensor
        """
        # 检查df中的数据类型
        dtypes = []
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                dtypes.append(float_dtype)
            elif pd.api.types.is_integer_dtype(df[col]):
                dtypes.append(int_dtype)
            else:
                raise ValueError(f"[dataframe_to_tensor]Unsupported data type in column {col}: {df[col].dtype}")
        # print(dtypes)
        # 将df中的每一列转换为tensor
        # 对于多维的data
        if len(dtypes) > 1:
            tensors = [torch.as_tensor(df[col].values, dtype=dtype) for col, dtype in zip(df.columns, dtypes)]
            return torch.stack(tensors, dim=1)  # 使用torch.stack将多个tensor堆叠在一起
        # 对于一维的target
        elif len(dtypes) == 1:
            return torch.as_tensor(df.values, dtype=dtypes[0])
        else:
            raise ValueError(f"[dataframe_to_tensor]数据长度有误{len(dtypes)}")


# 下面两个api因为复用性不强，将被弃用，请尽量不要使用
def train_net(X_train, y_train, data_iter=None, net=None, loss=None, optimizer=None, lr=0.001, num_epochs=1000,
              batch_size=64, show_interval=10, hidden=None):
    if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
    if data_iter is None:
        data_iter = load_array((X_train, y_train), batch_size)
    if net is None:
        net = nn.Sequential(nn.Flatten(), nn.Linear(X_train.shape[1], 1))
        net[1].weight.data.normal_(0, 0.01)
    if loss is None:
        loss = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if hidden is None:
        for epoch in range(num_epochs):
            for X, y in data_iter:
                y_hat = net(X)  # 输入的X经过net所计算出的值
                loss_value = loss(y_hat, y)
                optimizer.zero_grad()  # 清除上一次的梯度值
                loss_value.sum().backward()  # 反向传播，求参数的梯度
                # for param in net.parameters():
                #     print(param.grad)
                optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
            if epoch % show_interval == 0:
                loss_value = loss(net(X_train), y_train)
                print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
    else:
        for epoch in range(num_epochs):
            loss_value_sum = 0
            for X, y in data_iter:
                # print(X.shape, y.shape, hidden.shape)  # torch.Size([60, 1, 1]) torch.Size([60, 1, 1]) torch.Size([10, 60, 20])
                y_hat = net(X, hidden)  # 输入的X和隐藏层(h)经过net所计算出的值
                loss_value = loss(y_hat, y)
                optimizer.zero_grad()  # 清除上一次的梯度值
                loss_value.sum().backward()  # 反向传播，求参数的梯度
                # for param in net.parameters():
                #     print(param.grad)
                optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
                loss_value_sum += loss_value.sum()

            if epoch % show_interval == 0:
                # loss_value = loss(net(X_train), y_train)
                print(f'epoch {epoch + 1}, loss {loss_value_sum:f}')

    return net


def train_net_with_evaluation(X_train, y_train, X_test, y_test, data_iter=None, test_iter=None, net=None,
                              loss=None, optimizer=None, lr=0.001, num_epochs=1000, batch_size=64,
                              show_interval=10, draw='loss', if_immediately=True):
    if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
    if data_iter is None:
        data_iter = load_array((X_train, y_train), batch_size)
    if test_iter is None:
        test_iter = load_array((X_test, y_test), batch_size, if_shuffle=False)
    if net is None:
        net = nn.Sequential(nn.Flatten(), nn.Linear(X_train.shape[1], 1))
        net[1].weight.data.normal_(0, 0.01)
    if loss is None:
        loss = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    if if_immediately:
        fig, ax = plt.subplots()

    if draw == 'loss':
        for epoch in range(num_epochs):
            for X, y in data_iter:
                y_hat = net(X)  # 输入的X经过net所计算出的值
                loss_value = loss(y_hat, y)
                optimizer.zero_grad()  # 清除上一次的梯度值
                loss_value.sum().backward()  # 反向传播，求参数的梯度
                # for param in net.parameters():
                #     print(param.grad)
                optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
            if epoch % show_interval == 0:
                loss_value = loss(net(X_train), y_train).detach()
                test_loss_value = loss(net(X_test), y_test).detach()
                print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
                train_loss_list.append(loss_value)
                train_acc_list.append(loss_value)
                test_acc_list.append(test_loss_value)
                if if_immediately:
                    draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1,
                                                      show_interval, content='loss')
        if if_immediately:
            plt.show()
        else:
            draw_Loss_or_Accuracy([train_acc_list, test_acc_list], num_epochs, show_interval, content='acc')
    elif draw == 'acc':
        for epoch in range(num_epochs):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in data_iter:
                y_hat = net(X)  # 输入的X经过net所计算出的值
                loss_value = loss(y_hat, y)
                optimizer.zero_grad()  # 清除上一次的梯度值
                loss_value.sum().backward()  # 反向传播，求参数的梯度
                # for param in net.parameters():
                #     print(param.grad)
                optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
                n += y.shape[0]
                train_loss_sum += loss_value.item()
                train_acc_sum += count_correct_predictions(y_hat, y)
            if epoch % show_interval == 0:
                loss_value = loss(net(X_train), y_train)
                print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
                train_loss_list.append(train_loss_sum / n)
                train_acc_list.append(train_acc_sum / n)
                test_acc_list.append(evaluate_accuracy(net, test_iter))
                if if_immediately:
                    draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1,
                                                      show_interval, content='acc')
        if if_immediately:
            plt.show()
        else:
            draw_Loss_or_Accuracy([train_acc_list, test_acc_list], num_epochs, show_interval, content='acc')
    return net, train_loss_list, train_acc_list, test_acc_list
