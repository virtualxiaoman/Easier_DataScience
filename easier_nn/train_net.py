# 为了便于在kaggle上运行，本文件不依赖于easier_DataScience的其余部分

import os
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NetTrainer:
    """
    这是一个简易的nn训练器。仅支持前馈神经网络，暂未完全确认NetTrainer能完美支持RNN等含有隐藏状态的网络。

    你可以使用下面2个代码快速上手：

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
        trainer.train_net()
        trainer.evaluate_net()

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
        trainer = NetTrainer(data, target, net, loss_fn, optimizer, epochs=50, eval_type="acc")
        trainer.train_net()
        trainer.evaluate_net()

        print(trainer.test_acc_list)
    """

    def __init__(self, data, target, net, loss_fn, optimizer,  # 必要参数，数据与网络的基本信息
                 test_size=0.2, batch_size=64, epochs=100,     # 可选参数，用于训练
                 eval_type="loss",                             # 比较重要的参数，用于选择训练的类型（与评估指标有关）
                 eval_during_training=True,                    # 可选参数，训练时是否进行评估（与显存有关）
                                                               # 补充：经过优化，目前即使训练时评估也不需要额外太多的显存了
                 rnn_input_size=None, rnn_seq_len=None, rnn_hidden_size=None,  # 可选参数，当net是RNN类型时需要传入这些参数
                 # Bug：对RNN的train,test划分不太行，建议传入tuple
                 # batch_size不是1的时候测试集的损失会有问题
                 eval_interval=20,                             # 其他参数，训练时的评估间隔
                 device=None,                                  # 其他参数，设备选择
                 **kwargs):
        """
        初始化模型。

        :param data: 数据(全部数据/已经划分好了的元组)或训练集(包含X, y的DataLoader): X or (X_train, X_test) or train_loader
        :param target: 目标(全部数据/已经划分好了的元组)或验证集(包含X, y的DataLoader): y or (y_train, y_test) or test_loader
        :param net: 支持 net=nn.Sequential() or class Net(nn.Module)
        :param loss_fn: 损失函数，例如：
            nn.MSELoss()  # 回归，y的维度应该是(batch,)
            nn.CrossEntropyLoss()  # 分类，y的维度应该是(batch,)，并且网络的最后一层不需要加softmax
            nn.BCELoss()  # 二分类，y的维度应该是(batch,)，并且网络的最后一层需要加sigmoid
        :param optimizer: 优化器
        :param test_size: 测试集大小，支持浮点数或整数。该参数在data和target是tuple时无效
        :param batch_size: 批量大小
        :param epochs: 训练轮数
        :param eval_type: 模型类型，只可以是"loss"(回归-损失)或"acc"(分类-准确率)
        :param eval_interval: 打印间隔，请注意train_loss_list等间隔也是这个
        :param eval_during_training: 训练时是否进行评估，当显存不够时，可以设置为False，等到训练结束之后再进行评估
          设置为False时，不会影响训练集上的Loss的输出，但是无法输出验证集上的loss、训练集与验证集上的acc，此时默认输出"No eval"
        :param rnn_input_size: RNN的输入维度
        :param rnn_seq_len: RNN的序列长度
        :param rnn_hidden_size: RNN的隐藏层大小
          以上三个参数同时设置时，自动判断网络类型为RNN
        :param device: 设备，支持"cuda"或"cpu"，默认为None，自动优先选择"cuda"
        :param kwargs: 其他参数，包括：
          target_reshape_1D: 是否将y的维度转换为1维，默认为True，用于_target_reshape_1D函数，
            仅在为True且self.eval_type == "acc"且y.dim() > 1时才会转换并发出警告
          drop_last: 是否丢弃最后一个batch，默认为False，用于DataLoader
        """
        self.target_reshape_1D = kwargs.get("target_reshape_1D", True)
        self.drop_last = kwargs.get("drop_last", False)

        # 设备参数
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[__init__] 当前设备为{self.device}")

        # 数据参数
        self.data = data  # X or (X_train, X_test) or train_loader
        self.target = target  # y or (y_train, y_test) or test_loader
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.train_loader, self.test_loader = None, None

        # 网络参数
        self.net = net.to(self.device)
        # self.net = torch.compile(self.net)  # RuntimeError: Windows not yet supported for torch.compile 哈哈哈！
        self.net_type = "FNN"  # 默认是前馈神经网络
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
        self.time_list = []
        self.eval_interval = eval_interval  # 打印间隔

        # 使用loss还是acc参数
        self.eval_type = eval_type

        # 当前训练epoch的各种参数
        self.loss_epoch = None  # 每个epoch的loss
        self.current_gpu_memory = None  # 当前GPU显存

        # 训练时是否进行评估
        self.eval_during_training = eval_during_training
        self.NO_EVAL_MSG = '"No eval"'  # 不在训练时评估的输出
        self.best_epoch = None  # 最佳epoch
        self.best_loss = None  # 最佳loss
        self.best_acc = None  # 最佳acc
        self.auto_save_best_net = False  # 是否自动保存最佳模型

        # self.original_dataset_to_device = False  # False表示数据还没有转移到设备上
        # 是否是RNN类型
        self.rnn_seq_len = rnn_seq_len  # 该参数暂时没有使用，如果代码写完了还没用就删了得了
        self.rnn_hidden_size = rnn_hidden_size  # 同上
        self.rnn_input_size = rnn_input_size  # 在计算损失时需要用到
        self.hidden = None
        if self.rnn_input_size:
            self.net_type = "RNN"

        # 初始化
        self.init_loader()

    # [init]初始化训练数据
    def init_loader(self):
        # 如果传入的是DataLoader实例，则直接赋值
        if isinstance(self.data, DataLoader) and isinstance(self.target, DataLoader):
            self.train_loader = self.data
            self.test_loader = self.target
            print("[init_loader] 传入的data与target是DataLoader实例，直接赋值train_loader和test_loader。")
            # 从DataLoader中获取数据
            self.X_train, self.y_train = self._dataloader_to_tensor(self.train_loader)
            self.X_test, self.y_test = self._dataloader_to_tensor(self.test_loader)
        else:
            # 如果传入的就是tuple，则表示已经划分好了训练集和测试集
            if isinstance(self.data, tuple) and isinstance(self.target, tuple):
                self.X_train, self.X_test = self.data
                self.y_train, self.y_test = self.target
                print("[init_loader] 因为传入的data与target是元组，所以默认已经划分好了训练集和测试集。"
                      "默认元组第一个是train，第二个为test。")
            # 否则，需要划分训练集和测试集
            else:
                if self.data.shape[0] != self.target.shape[0]:
                    raise ValueError(f"data和target的shape[0](样本数)不相同: "
                                     f"data({self.data.shape[0]}) and target({self.target.shape[0]}).")
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.data, self.target, test_size=self.test_size)
                print(f"[init_loader] 传入的data与target是X, y，则按照test_size={self.test_size}划分训练集和测试集")

            # if self.net_type == "RNN":
            #     self.X_train, self.y_train = self._prepare_rnn_data(self.X_train, self.y_train)
            #     self.X_test, self.y_test = self._prepare_rnn_data(self.X_test, self.y_test)
            #     print(f"[init_loader]RNN数据准备完毕，seq_len={self.rnn_seq_len}, hidden_size={self.rnn_hidden_size}")

            # 创建DataLoaders
            self.train_loader = self.create_dataloader(self.X_train, self.y_train)
            self.test_loader = self.create_dataloader(self.X_test, self.y_test, train=False)
        # 将数据变成tensor，并且dtype依据data的类型而定
        self.X_train = self._dataframe_to_tensor(self.X_train)
        self.X_test = self._dataframe_to_tensor(self.X_test)
        self.y_train = self._dataframe_to_tensor(self.y_train)
        self.y_test = self._dataframe_to_tensor(self.y_test)
        self.y_train = self._target_reshape_1D(self.y_train)
        self.y_test = self._target_reshape_1D(self.y_test)

        print(f"[init_loader] 训练集X, y的shape为{self.X_train.shape}, {self.y_train.shape}。"
              f"测试集X, y的shape为{self.X_test.shape}, {self.y_test.shape}。")

    # [子函数]创建dataloader
    def create_dataloader(self, data, target, train=True):
        # dtype依据data的类型而定
        data = self._dataframe_to_tensor(data)
        target = self._dataframe_to_tensor(target)
        target = self._target_reshape_1D(target)

        # print(target)
        dataset = TensorDataset(data, target)
        if train:
            # todo 本处对RNN的处理应该有问题
            if self.net_type == "RNN":
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_last)
            else:
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_last)

    # [主函数]训练模型
    def train_net(self, hidden=None, net_save_path: str = None) -> None:
        """
        训练模型
        :param hidden: 隐藏层，用于RNN
        :param net_save_path: 最佳模型保存path，会在每次评估后保存最佳模型，该参数在eval_during_training=True时有效。
            暂不支持选择state_dict的保存除非改代码，只支持整个模型的保存（因为我平时不怎么用state_dict阿巴阿巴）
        :return: None
        """
        if hidden is not None:
            self.hidden = hidden
        print(f"[train_net] 开始训练模型，总共epochs={self.epochs}，batch_size={self.batch_size}，"
              f"当前设备为{self.device}，网络类型为{self.net_type}，评估类型为{self.eval_type}。")
        self.__check_best_net_save_path(net_save_path)
        self.current_gpu_memory = self._log_gpu_memory()

        for epoch in range(self.epochs):
            self.net.train()  # 确保dropout等在训练时生效
            self.train_epoch()  # 训练的主体部分
            # 打印训练信息
            if epoch % self.eval_interval == 0:
                self.log_and_update_eval_msg(epoch, net_save_path)

        print(f"[train_net]训练结束，总共花费时间: {sum(self.time_list)}秒")
        if self.eval_during_training:
            if self.eval_type == "loss":
                print(f"[train_net] 最佳结果 epoch = {self.best_epoch + 1}, loss = {self.best_loss}")
            elif self.eval_type == "acc":
                print(f"[train_net] 最佳结果 epoch = {self.best_epoch + 1}, acc = {self.best_acc}")
        self.eval_during_training = True  # 训练完成后，可以进行评估

    # 对某个epoch进行训练，仅在train_net中调用。可以通过复写这个函数来实现自定义的训练
    def train_epoch(self):
        start_time = time.time()
        loss_sum = 0.0
        for X, y in self.train_loader:
            # 初始化数据
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # 前向传播
            if self.net_type == "RNN":
                if self.hidden is not None:
                    self.hidden.detach_()
                #     print(self.hidden.shape)
                # print(X.shape, y.shape)
                # print("----------上面是TRAIN的hidden, X, y的shape---------")
                outputs, self.hidden = self.net(X, self.hidden)
                loss = self.loss_fn(outputs, y)
            else:
                # print(X.shape, y.shape)
                # print("----------上面是TRAIN的hidden, X, y的shape---------")
                outputs = self.net(X)
                # print(X.shape, y.shape, outputs.shape)
                loss = self.loss_fn(outputs, y)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            # 计算损失
            loss_sum += loss.item()
            # 计算当前GPU显存
            self.current_gpu_memory = self._log_gpu_memory()
            # 释放显存。如果不释放显存，直到作用域结束时才会释放显存（这部分一直在reserve的显存里面）
            del X, y, outputs, loss
            torch.cuda.empty_cache()
        self.loss_epoch = loss_sum / len(self.train_loader)
        self.time_list.append(time.time() - start_time)

    # 在训练时评估并保存最佳模型，仅在train_net中调用。此函数会不断存储最佳模型，只是怕后面哪一次意外失败了那就白训练了
    def log_and_update_eval_msg(self, epoch, net_save_path):
        if self.eval_type == "loss":
            self.train_loss_list.append(self.loss_epoch)
            self.test_loss_list.append(self.evaluate_net())
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {self.loss_epoch}, '
                  f'Test Loss: {self.test_loss_list[-1]}, '
                  f'Time: {self.time_list[-1]:.2f}s, '
                  f'GPU: {self.current_gpu_memory}')
            if self.eval_during_training:
                # 如果当前loss小于最佳loss，则保存self.epoch和self.loss
                if self.best_loss is None or self.test_loss_list[-1] < self.best_loss:
                    self.best_loss = self.test_loss_list[-1]
                    self.best_epoch = epoch
                    if self.auto_save_best_net:
                        self.__save_net(net_save_path)
        elif self.eval_type == "acc":
            self.train_acc_list.append(self.evaluate_net(eval_type="train"))
            self.test_acc_list.append(self.evaluate_net())
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {self.loss_epoch}, '
                  f'Train Acc: {self.train_acc_list[-1]}, '
                  f'Test Acc: {self.test_acc_list[-1]}, '
                  f'Time: {self.time_list[-1]:.2f}s, '
                  f'GPU: {self.current_gpu_memory}')
            if self.eval_during_training:
                # 如果当前acc大于最佳acc，则保存self.epoch和self.acc
                if self.best_acc is None or self.test_acc_list[-1] > self.best_acc:
                    self.best_acc = self.test_acc_list[-1]
                    self.best_epoch = epoch
                    if self.auto_save_best_net:
                        self.__save_net(net_save_path)
        else:
            raise ValueError("eval_type must be 'loss' or 'acc'")

    # [主函数]评估模型(暂不支持RNN的评估)
    def evaluate_net(self, eval_type: str = "test", delete_train: bool = False) -> float | str:
        """
        评估模型
        :param eval_type: 评估类型，支持"test"和"train"
        :param delete_train: delete_train=True表示删除训练集，只保留测试集，这样可以释放显存
        :return: 损失或准确率，依据self.net_type而定；在不评估时返回self.NO_EVAL_MSG(默认为'"No eval"')
        """
        if delete_train:
            del self.X_train, self.y_train
            torch.cuda.empty_cache()
        # if self.eval_during_training:
        #     self.__original_dataset_to_device()  # 如果要在训练时评估，需要将数据转移到设备上
        # else:
        #     return self.NO_EVAL_MSG  # 不在训练时评估
        if not self.eval_during_training:
            return self.NO_EVAL_MSG  # 不在训练时评估
        self.net.eval()  # 确保评估时不使用dropout等
        with torch.no_grad():  # 在评估时禁用梯度计算，节省内存
            if self.eval_type == "loss":
                if self.net_type == "RNN":
                    if eval_type == "test":
                        output = self._cal_rnn_output(self.net, self.X_test[0], self.hidden[:, -1], len(self.y_test))
                        loss = self.loss_fn(output, self.y_test).item()
                    else:
                        # 事实上一般不调用这个，因为训练集的loss在训练时已经计算了
                        output = self._cal_rnn_output(self.net, self.X_train[0], self.hidden[:, 0], len(self.y_train))
                        loss = self.loss_fn(output, self.y_train).item()
                else:
                    if eval_type == "test":
                        loss = self._cal_fnn_loss(self.net, self.loss_fn, self.X_test, self.y_test)
                        # loss = self.loss_fn(self.net(self.X_test), self.y_test).item()
                    else:
                        # 事实上一般不调用这个，因为训练集的loss在训练时已经计算了
                        loss = self._cal_fnn_loss(self.net, self.loss_fn, self.X_train, self.y_train)
                        # loss = self.loss_fn(self.net(self.X_train), self.y_train).item()
                return loss
            elif self.eval_type == "acc":
                if self.net_type == "RNN":
                    if eval_type == "test":
                        acc = self._cal_rnn_acc(self.net, self.X_test, self.y_test)
                        # predictions = torch.argmax(self.net(self.X_test, self.hidden), dim=1).type(self.y_test.dtype)
                        # correct = (predictions == self.y_test).sum().item()
                        # n = self.y_test.numel()
                        # acc = correct / n
                    else:
                        acc = self._cal_rnn_acc(self.net, self.X_train, self.y_train)
                        # predictions = torch.argmax(self.net(self.X_train, self.hidden), dim=1).type(self.y_train.dtype)
                        # correct = (predictions == self.y_train).sum().item()
                        # n = self.y_train.numel()
                        # acc = correct / n
                else:
                    if eval_type == "test":
                        acc = self._cal_fnn_acc(self.net, self.X_test, self.y_test)
                        # predictions = torch.argmax(self.net(self.X_test), dim=1).type(self.y_test.dtype)
                        # correct = (predictions == self.y_test).sum().item()
                        # n = self.y_test.numel()
                        # acc = correct / n
                    else:
                        acc = self._cal_fnn_acc(self.net, self.X_train, self.y_train)
                        # predictions = torch.argmax(self.net(self.X_train), dim=1).type(self.y_train.dtype)
                        # correct = (predictions == self.y_train).sum().item()
                        # n = self.y_train.numel()
                        # acc = correct / n
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

    # [主函数]查看模型参数。使用Netron(需要安装)可视化更好，这里只是简单的查看
    def view_parameters(self, view_net_struct=False, view_params_count=True, view_params_details=False):
        # if view_layers:
        #     for layer in self.net.children():
        #         print(layer)
        if view_net_struct:
            print("网络结构如下：")
            print(self.net)
        if view_params_count:
            count = 0
            for p in self.net.parameters():
                if view_params_details:
                    print("该层的参数：" + str(list(p.size())))
                count += p.numel()
            print(f"总参数量: {count}")
            # print(f"Total params: {sum(p.numel() for p in self.net.parameters())}")

        # params = list(self.net.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     print(f"该层的名称：{i.size()}")
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        # print("总参数数量和：" + str(k))

    # [子函数]评估FNN的loss
    def _cal_fnn_loss(self, net, criterion, x, y):
        net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_fn_loss]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                del X_batch, y_batch, outputs, loss
                torch.cuda.empty_cache()

        average_loss = total_loss / len(x)
        return average_loss

    # [子函数]评估RNN的loss
    def _cal_rnn_loss(self, net, criterion, x, y):
        net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_rnn_loss]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                hidden = self.hidden.detach()
                outputs, _ = net(X_batch, hidden)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                del X_batch, y_batch, outputs, loss
                torch.cuda.empty_cache()

        average_loss = total_loss / len(x)
        return average_loss

    # [子函数]评估RNN的loss（该函数暂时有问题）
    def _cal_rnn_output(self, net, x, hidden, pred_steps):
        hidden.to(self.device)
        pred_list = []
        # 输出x的shape
        # print(x.shape)
        # print(x)
        # 调整输入形状为 [batch_size, seq_len, input_size]
        # inp = x.view(self.batch_size, self.rnn_seq_len, self.rnn_input_size).to(self.device)

        inp = x.view(-1, self.rnn_input_size).to(self.device)
        # print(x.shape, inp.shape, hidden.shape)
        # print("----------上面是EVAL的x, inp, hidden的shape---------")
        for i in range(pred_steps):
            pred, hidden = net(inp, hidden)
            pred_list.append(pred.detach())
            inp = pred
        return torch.cat(pred_list, dim=0).view(-1)

    # [子函数]评估FNN的acc
    def _cal_fnn_acc(self, net, x, y):
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_accuracy]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                outputs = net(X_batch)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                del X_batch, y_batch, outputs, predictions
                torch.cuda.empty_cache()

        accuracy = correct / total
        return accuracy

    # [子函数]评估RNN的acc
    def _cal_rnn_acc(self, net, x, y):
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                # 如果X_batch的长度不等于batch_size，说明是最后一个batch
                if len(X_batch) != self.batch_size:
                    warnings.warn(f"[_cal_rnn_acc]最后一个batch的长度为{len(X_batch)}≠{self.batch_size}，"
                                  f"暂时的处理方法是跳过，可能会影响准确率的计算")
                    break
                hidden = self.hidden.detach()
                outputs, _ = net(X_batch, hidden)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                del X_batch, y_batch, outputs, predictions
                torch.cuda.empty_cache()

        accuracy = correct / total
        return accuracy

    # # [子函数]准备RNN数据
    # def _prepare_rnn_data(self, data, target):
    #     seq_len = self.rnn_seq_len
    #     data_len = len(data)
    #     num_sequences = data_len // (seq_len + 1) * (seq_len + 1)
    #     data = np.array(data[:num_sequences]).reshape((-1, seq_len + 1, 1))
    #     target = np.array(target[:num_sequences]).reshape((-1, seq_len + 1, 1))
    #     return data[:, :seq_len], data[:, 1:seq_len + 1]

    # 将df转换为tensor，并保持数据类型的一致性

    # [log函数]打印GPU显存
    def _log_gpu_memory(self):
        if not self.eval_during_training:
            return self.NO_EVAL_MSG  # 不在训练时评估
        log = None

        # 获取self.device的设备索引
        self_device_index = None
        # 如果是cuda
        if self.device.type == "cuda":
            self_device_index = self.device.index

        # 获取当前设备索引
        current_device_index = torch.cuda.current_device()
        if current_device_index is None:
            log = "当前没有GPU设备"
            return log
        elif self_device_index is not None and current_device_index != self_device_index:
            warnings.warn(f"[_log_gpu_memory]当前设备为{current_device_index}，与{self_device_index}不一致")
        else:
            log = ""

        props = torch.cuda.get_device_properties(current_device_index)  # 获取设备属性
        used_memory = torch.cuda.memory_allocated(current_device_index)  # 已用显存（字节）
        reserved_memory = torch.cuda.memory_reserved(current_device_index)  # 保留显存（字节）
        total_memory = props.total_memory  # 总显存（字节）
        used_memory_gb = used_memory / (1024 ** 3)  # 已用显存（GB）
        reserved_memory_gb = reserved_memory / (1024 ** 3)  # 保留显存（GB）
        total_memory_gb = total_memory / (1024 ** 3)  # 总显存（GB）
        log += (f"设备{current_device_index}的显存："
                f"已用{used_memory_gb:.2f}+保留{reserved_memory_gb:.2f}/总{total_memory_gb:.2f}(GB)")

        return log

    @staticmethod
    def _dataframe_to_tensor(df, float_dtype=torch.float16, int_dtype=torch.int64):
        """
        PyTorch's tensors are homogenous, ie, each of the elements are of the same type.
        将df转换为tensor，并保持数据类型的一致性
        :param df: pd.DataFrame
        :param float_dtype: torch.dtype, default=torch.float32
        :param int_dtype: torch.dtype, default=torch.int32
        :return: torch.Tensor
        """
        # 先判断df是不是dataframe
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, torch.Tensor):
                return df
            elif isinstance(df, np.ndarray):
                return torch.tensor(df)
            else:
                raise ValueError("既不是dataframe又不是tensor")
        # 检查df中的数据类型
        dtypes = []
        for col in df.column:
            if pd.api.types.is_float_dtype(df[col]):
                dtypes.append(float_dtype)
            elif pd.api.types.is_integer_dtype(df[col]):
                dtypes.append(int_dtype)
            else:
                raise ValueError(f"[_dataframe_to_tensor]Unsupported data type in column {col}: {df[col].dtype}")
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
            raise ValueError(f"[_dataframe_to_tensor]数据长度有误{len(dtypes)}")

    @staticmethod
    def _dataloader_to_tensor(dataloader):
        data_list = []
        target_list = []
        for data, target in dataloader:
            data_list.append(data)
            target_list.append(target)
        return torch.cat(data_list), torch.cat(target_list)

    # 将y的维度转换为1维
    def _target_reshape_1D(self, y):
        """
        将y的维度转换为1维
        :param y: torch.Tensor
        :return: torch.Tensor
        """
        if self.target_reshape_1D and self.eval_type == "acc" and y.dim() > 1:
            warnings.warn(f"[_target_reshape_1D]请注意：y的维度为{y.dim()}: {y.shape}，将被自动转换为1维\n"
                          "如需保持原有维度，请设置 target_reshape_1D=False ")
            return y.view(-1)
        else:
            return y

    # 检查模型路径是否合法，该函数仅在train_net中调用
    def __check_best_net_save_path(self, net_save_path):
        if isinstance(net_save_path, str):
            if not self.eval_during_training:
                self.auto_save_best_net = False
                warnings.warn("net_save_path参数在eval_during_training=False时无效，auto_save_best_net仍然是False")
            else:
                dir_path = os.path.dirname(net_save_path)
                if os.path.exists(dir_path):
                    self.auto_save_best_net = True
                    print(f"[train_net] 最佳模型保存地址net_save_path={net_save_path}")
                else:
                    self.auto_save_best_net = True
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        warnings.warn(f"[train_net] 最佳模型保存文件夹dir_path='{dir_path}'不存在，已自动创建")
                    except Exception as e:
                        warnings.warn(f"[train_net] 最佳模型保存文件夹dir_path='{dir_path}'创建失败，错误信息：{e}")
                if not net_save_path.endswith(".pth"):
                    print(f"[train_net] 请注意net_save_path='{net_save_path}'未以'.pth'结尾")

    # 保存模型
    def __save_net(self, net_save_path, save_type="net"):
        try:
            if save_type == "net":
                torch.save(self.net, net_save_path)
            elif save_type == "state_dict":
                torch.save(self.net.state_dict(), net_save_path)
            else:
                raise ValueError(f"[train_net] 保存类型save_type={save_type}不合法")
            # print(f"[train_net] 已保存{save_type}模型到{net_save_path}")
        except Exception as e:
            warnings.warn(f"[train_net] 保存模型失败，错误信息：{e}")
    # 将原始数据转移到设备上，暂被弃用
    # def __original_dataset_to_device(self):
    #     # 暂时不知道只使用self.original_dataset_to_device是否会有问题，或许可以直接检查self.X_train.device(有问题再改吧)
    #     if not self.original_dataset_to_device:
    #         # 将数据转移到设备上
    #         self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(self.device), self.X_test.to(
    #             self.device), self.y_train.to(self.device), self.y_test.to(self.device)
    #         self.original_dataset_to_device = True


# GRU可以参考下面的代码，结果很好，等有空再将RNN这种网络的训练合并到NetTrainer中
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from easier_nn.classic_dataset import VirtualDataset

x = np.linspace(0, 1000, 1000)
data = np.sin(0.05 * x)

print(data.shape)
num_data = len(data)
split = int(0.8 * num_data)
print(f'数据集大小：{num_data}')
# 数据集可视化
plt.figure()
plt.scatter(np.arange(split), data[:split],
            color='blue', s=10, label='training set')
plt.scatter(np.arange(split, num_data), data[split:],
            color='orange', s=10, label='test set')
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.legend()
plt.show()
# 分割数据集
train_data = np.array(data[:split])
test_data = np.array(data[split:])

# 输入序列长度
seq_len = 20
# 处理训练数据，把切分序列后多余的部分去掉
train_num = len(train_data) // (seq_len + 1) * (seq_len + 1)
train_data = np.array(train_data[:train_num]).reshape(-1, seq_len + 1, 1)
np.random.seed(0)
torch.manual_seed(0)

x_train = train_data[:, :seq_len]  # 形状为(num_data, seq_len, input_size)
y_train = train_data[:, 1: seq_len + 1]
print(f'训练序列数：{len(x_train)}')

# 转为PyTorch张量
x_train = torch.from_numpy(x_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)
x_test = torch.from_numpy(test_data[:-1]).to(torch.float32)
y_test = torch.from_numpy(test_data[1:]).to(torch.float32)


class GRU(nn.Module):
    # 包含PyTorch的GRU和拼接的MLP
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        # GRU模块
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        # 将中间变量映射到预测输出的MLP
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # 前向传播
        # x的维度为(batch_size, seq_len, input_size)
        # GRU模块接受的输入为(seq_len, batch_size, input_size)
        # 因此需要对x进行变换
        # transpose函数可以交换x的坐标轴
        # out的维度是(seq_len, batch_size, hidden_size)
        out, hidden = self.gru(torch.transpose(x, 0, 1), hidden)
        # 取序列最后的中间变量输入给全连接层
        out = self.linear(out.view(-1, hidden_size))
        return out, hidden


# 超参数
input_size = 1  # 输入维度
output_size = 1  # 输出维度
hidden_size = 16  # 中间变量维度
learning_rate = 5e-4

# 初始化网络
gru = GRU(input_size, output_size, hidden_size)
gru_optim = torch.optim.Adam(gru.parameters(), lr=learning_rate)


# GRU测试函数，x和hidden分别是初始的输入和中间变量
def test_gru(gru, x, hidden, pred_steps):
    pred = []
    inp = x.view(-1, input_size)
    for i in range(pred_steps):
        gru_pred, hidden = gru(inp, hidden)
        pred.append(gru_pred.detach())
        inp = gru_pred
    return torch.concat(pred).reshape(-1)


# MLP的超参数
hidden_1 = 32
hidden_2 = 16
mlp = nn.Sequential(
    nn.Linear(input_size, hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, output_size)
)
mlp_optim = torch.optim.Adam(mlp.parameters(), lr=learning_rate)


# MLP测试函数，相比于GRU少了中间变量
def test_mlp(mlp, x, pred_steps):
    pred = []
    inp = x.view(-1, input_size)
    for i in range(pred_steps):
        mlp_pred = mlp(inp)
        pred.append(mlp_pred.detach())
        inp = mlp_pred
    return torch.concat(pred).reshape(-1)


max_epoch = 150
criterion = nn.functional.mse_loss
hidden = None  # GRU的中间变量

# 训练损失
gru_losses = []
mlp_losses = []
gru_test_losses = []
mlp_test_losses = []
# 开始训练
with tqdm(range(max_epoch)) as pbar:
    for epoch in pbar:
        st = 0
        gru_loss = 0.0
        mlp_loss = 0.0
        # 随机梯度下降
        for X, y in zip(x_train, y_train):
            # 更新GRU模型
            # 我们不需要通过梯度回传更新中间变量
            # 因此将其从有梯度的部分分离出来
            if hidden is not None:
                hidden.detach_()
            gru_pred, hidden = gru(X[None, ...], hidden)
            gru_train_loss = criterion(gru_pred.view(y.shape), y)
            gru_optim.zero_grad()
            gru_train_loss.backward()
            gru_optim.step()
            gru_loss += gru_train_loss.item()
            # 更新MLP模型
            # 需要对输入的维度进行调整，变成(seq_len, input_size)的形式
            mlp_pred = mlp(X.view(-1, input_size))
            mlp_train_loss = criterion(mlp_pred.view(y.shape), y)
            mlp_optim.zero_grad()
            mlp_train_loss.backward()
            mlp_optim.step()
            mlp_loss += mlp_train_loss.item()

        gru_loss /= len(x_train)
        mlp_loss /= len(x_train)
        gru_losses.append(gru_loss)
        mlp_losses.append(mlp_loss)

        # 训练和测试时的中间变量序列长度不同，训练时为seq_len，测试时为1
        gru_pred = test_gru(gru, x_test[0], hidden[:, -1], len(y_test))
        mlp_pred = test_mlp(mlp, x_test[0], len(y_test))
        gru_test_loss = criterion(gru_pred, y_test).item()
        mlp_test_loss = criterion(mlp_pred, y_test).item()
        gru_test_losses.append(gru_test_loss)
        mlp_test_losses.append(mlp_test_loss)

        pbar.set_postfix({
            'Epoch': epoch,
            'GRU loss': f'{gru_loss:.4f}',
            'MLP loss': f'{mlp_loss:.4f}',
            'GRU test loss': f'{gru_test_loss:.4f}',
            'MLP test loss': f'{mlp_test_loss:.4f}'
        })

# 最终测试结果
gru_preds = test_gru(gru, x_test[0], hidden[:, -1], len(y_test)).numpy()
mlp_preds = test_mlp(mlp, x_test[0], len(y_test)).numpy()

plt.figure(figsize=(13, 5))

# 绘制训练曲线
plt.subplot(121)
x_plot = np.arange(len(gru_losses)) + 1
plt.plot(x_plot, gru_losses, color='blue', label='GRU training loss')
plt.plot(x_plot, mlp_losses, color='red', label='MLP training loss')
plt.plot(x_plot, gru_test_losses, color='blue',
         linestyle='--', label='GRU test loss')
plt.plot(x_plot, mlp_test_losses, color='red',
         linestyle='--', label='MLP test loss')
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.legend(loc='lower left')

# 绘制真实数据与模型预测值的图像
plt.subplot(122)
plt.scatter(np.arange(split), data[:split], color='blue',
            s=10, label='training set')
plt.scatter(np.arange(split, num_data), data[split:], color='orange',
            s=10, label='test set')
plt.scatter(np.arange(split, num_data - 1), mlp_preds, color='purple',
            s=10, label='MLP preds')
plt.scatter(np.arange(split, num_data - 1), gru_preds, color='green',
            s=10, label='GRU preds')
plt.legend(loc='lower left')
plt.show()
"""

# RNN、LSTM可以参考下面的代码
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from easier_nn.classic_dataset import VirtualDataset

# 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# 数据处理
def prepare_data(x, y, seq_length=10):
    x_seq, y_seq = [], []
    for i in range(len(x) - seq_length):
        x_seq.append(x[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    x_seq = torch.stack(x_seq)
    y_seq = torch.stack(y_seq)
    return x_seq.unsqueeze(-1), y_seq.unsqueeze(-1)


# 创建虚拟数据集
dataset = VirtualDataset()
dataset.sinx(show_plt=True)

# 准备训练数据
seq_length = 10
x_seq, y_seq = prepare_data(dataset.x, dataset.y, seq_length)

# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(x_seq, y_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
input_size = 1
hidden_size = 20
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        hidden = model.init_hidden(inputs.size(0))
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs[:, -1, :], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练好的模型进行逐步预测
model.eval()
test_inputs = dataset.x[:seq_length].unsqueeze(-1).unsqueeze(0)
predicted = []
hidden = model.init_hidden(1)
for _ in range(len(dataset.x) - seq_length):
    with torch.no_grad():
        pred, hidden = model(test_inputs, hidden)
        predicted.append(pred[:, -1, :].item())
        test_inputs = torch.cat((test_inputs[:, 1:, :], pred[:, -1:, :]), dim=1)

# 绘制预测结果
import matplotlib.pyplot as plt

plt.plot(dataset.x.numpy(), dataset.y.numpy(), label='True')
plt.plot(dataset.x[seq_length:].numpy(), predicted, label='Predicted')
plt.legend()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# 定义LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# 虚拟数据集类
class VirtualDataset:
    def __init__(self, start=1, end=100, num_points=None):
        self.start = start
        self.end = end
        if isinstance(num_points, int):
            self.num_points = num_points
        else:
            self.num_points = (end - start + 1) * 10  # 相当于间隔是0.1
        self.x = torch.linspace(self.start, self.end, self.num_points)
        self.y = None

    def sinx(self, w=0.01, noise_mu=0, noise_sigma=0.2, show_plt=False):
        noise = torch.normal(noise_mu, noise_sigma, (self.num_points,))
        self.y = torch.sin(w * self.x) + noise
        if show_plt:
            self.plot_xy(self.x.numpy(), self.y.numpy())

    def plot_xy(self, x, y):
        plt.plot(x, y)
        plt.show()


# 数据处理
def prepare_data(x, y, seq_length=10):
    x_seq, y_seq = [], []
    for i in range(len(x) - seq_length):
        x_seq.append(x[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    x_seq = torch.stack(x_seq)
    y_seq = torch.stack(y_seq)
    return x_seq.unsqueeze(-1), y_seq.unsqueeze(-1)


# 创建虚拟数据集
dataset = VirtualDataset()
dataset.sinx(show_plt=True)

# 数据归一化
x_min, x_max = dataset.x.min(), dataset.x.max()
y_min, y_max = dataset.y.min(), dataset.y.max()
dataset.x = (dataset.x - x_min) / (x_max - x_min)
dataset.y = (dataset.y - y_min) / (y_max - y_min)

# 准备训练数据
seq_length = 20
x_seq, y_seq = prepare_data(dataset.x, dataset.y, seq_length)

# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(x_seq, y_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
input_size = 1
hidden_size = 50
output_size = 1
num_layers = 2
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        hidden = model.init_hidden(inputs.size(0))
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs[:, -1, :], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练好的模型进行逐步预测
model.eval()
test_inputs = dataset.x[:seq_length].unsqueeze(-1).unsqueeze(0)
predicted = []
hidden = model.init_hidden(1)
for _ in range(len(dataset.x) - seq_length):
    with torch.no_grad():
        pred, hidden = model(test_inputs, hidden)
        predicted.append(pred[:, -1, :].item())
        test_inputs = torch.cat((test_inputs[:, 1:, :], pred[:, -1:, :]), dim=1)

# 反归一化预测结果
predicted = np.array(predicted)
y_max_np, y_min_np = y_max.item(), y_min.item()
predicted = predicted * (y_max_np - y_min_np) + y_min_np

# 绘制预测结果
plt.plot(dataset.x.numpy() * (x_max.item() - x_min.item()) + x_min.item(), dataset.y.numpy() * (y_max.item() - y_min.item()) + y_min.item(), label='True')
plt.plot(dataset.x[seq_length:].numpy() * (x_max.item() - x_min.item()) + x_min.item(), predicted, label='Predicted')
plt.legend()
plt.show()
"""

# bert可以参考下面的代码
"""
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('../HF_model/google-bert/bert-base-chinese')
model = BertModel.from_pretrained('../HF_model/google-bert/bert-base-chinese')
chat = BertForSequenceClassification.from_pretrained('../HF_model/google-bert/bert-base-chinese')  # https://www.cnblogs.com/zhangxianrong/p/15066981.html

def text2embedding(text):
    # 对文本进行tokenize
    inputs = tokenizer(text, return_tensors="pt")
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    # # 获取句子的embedding (这里取[CLS] token的embedding)
    # sentence_embedding = last_hidden_states[:, 0, :]
    # # 获取所有token的平均embedding
    # sentence_embedding = torch.mean(last_hidden_states, dim=1)
    # 取所有token的平均值
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

text1 = "I like apples."
text2 = "The weather is cold."

# 通过模型获取文本的embedding，然后计算余弦相似度
embedding1 = text2embedding(text1)
embedding2 = text2embedding(text2)
similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
print(similarity.item())
# 相似度太高的问题可以参考：
# https://github.com/terrifyzhao/bert-utils/issues/70  里提及的：
# https://spaces.ac.cn/archives/8541
# https://www.cnblogs.com/shona/p/12021304.html#l1

def compute_similarity(text1, text2):
    embedding1 = text2embedding(text1)
    embedding2 = text2embedding(text2)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()
# # 例子1：完全不同的主题
# text1 = "The cat is sleeping on the mat."
# text2 = "Photosynthesis is the process by which plants make their food."
# similarity1 = compute_similarity(text1, text2)
# print(f"Cosine Similarity between '{text1}' and '{text2}': {similarity1}")
#
# # 例子2：完全不同的内容
# text1 = "The Eiffel Tower is in Paris."
# text2 = "Quantum mechanics is a fundamental theory in physics."
# similarity2 = compute_similarity(text1, text2)
# print(f"Cosine Similarity between '{text1}' and '{text2}': {similarity2}")
#
# # 例子3：短语之间的相似性
# text1 = "I enjoy playing soccer."
# text2 = "Soccer is a popular sport."
# similarity3 = compute_similarity(text1, text2)
# print(f"Cosine Similarity between '{text1}' and '{text2}': {similarity3}")
#
# # 例子4：完全无关的内容
# text1 = "How to cook spaghetti?"
# text2 = "The theory of relativity."
# similarity4 = compute_similarity(text1, text2)
# print(f"Cosine Similarity between '{text1}' and '{text2}': {similarity4}")

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

text = "我要玩碧蓝档案"
predictions = predict(text, tokenizer, chat)
print(predictions)

# def predict_masked_word(text):
#     # 将文本中的[MASK]替换为BERT的[MASK]标记
#     input_text = text.replace('[MASK]', tokenizer.mask_token)
#
#     # 对文本进行tokenize
#     inputs = tokenizer(input_text, return_tensors="pt")
#
#     # 获取模型输出
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     # 获取预测的logits
#     predictions = outputs.last_hidden_state
#
#     # 获取[MASK] token的位置
#     mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
#
#     # 获取[MASK] token的logits
#     mask_token_logits = predictions[0, mask_token_index, :]
#
#     # 获取每个词的概率
#     top_k = 10
#     top_k_indices = torch.topk(mask_token_logits, top_k, dim=-1).indices[0].tolist()
#     top_k_probabilities = torch.nn.functional.softmax(mask_token_logits, dim=-1)[0, top_k_indices].tolist()
#
#     # 获取top_k词的预测结果
#     predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
#
#     result = {}
#     for token, prob in zip(predicted_tokens, top_k_probabilities):
#         result[token] = prob
#
#     return result
#
#
# # 示例文本
# text = "生活的真谛是[MASK]。"
#
# # 获取预测结果
# predictions = predict_masked_word(text)
# for word, prob in predictions.items():
#     print(f"{word}: {prob:.3f}")
"""

# 下面两个api因为复用性不强，将被弃用，请尽量不要使用
# def train_net(X_train, y_train, data_iter=None, net=None, loss=None, optimizer=None, lr=0.001, num_epochs=1000,
#               batch_size=64, show_interval=10, hidden=None):
#     if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
#         X_train = torch.tensor(X_train.values, dtype=torch.float32)
#         y_train = torch.tensor(y_train.values, dtype=torch.float32)
#     if data_iter is None:
#         data_iter = load_array((X_train, y_train), batch_size)
#     if net is None:
#         net = nn.Sequential(nn.Flatten(), nn.Linear(X_train.shape[1], 1))
#         net[1].weight.data.normal_(0, 0.01)
#     if loss is None:
#         loss = nn.MSELoss()
#     if optimizer is None:
#         optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#
#     if hidden is None:
#         for epoch in range(num_epochs):
#             for X, y in data_iter:
#                 y_hat = net(X)  # 输入的X经过net所计算出的值
#                 loss_value = loss(y_hat, y)
#                 optimizer.zero_grad()  # 清除上一次的梯度值
#                 loss_value.sum().backward()  # 反向传播，求参数的梯度
#                 # for param in net.parameters():
#                 #     print(param.grad)
#                 optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
#             if epoch % show_interval == 0:
#                 loss_value = loss(net(X_train), y_train)
#                 print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
#     else:
#         for epoch in range(num_epochs):
#             loss_value_sum = 0
#             for X, y in data_iter:
#                 # print(X.shape, y.shape, hidden.shape)  # torch.Size([60, 1, 1]) torch.Size([60, 1, 1]) torch.Size([10, 60, 20])
#                 y_hat = net(X, hidden)  # 输入的X和隐藏层(h)经过net所计算出的值
#                 loss_value = loss(y_hat, y)
#                 optimizer.zero_grad()  # 清除上一次的梯度值
#                 loss_value.sum().backward()  # 反向传播，求参数的梯度
#                 # for param in net.parameters():
#                 #     print(param.grad)
#                 optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
#                 loss_value_sum += loss_value.sum()
#
#             if epoch % show_interval == 0:
#                 # loss_value = loss(net(X_train), y_train)
#                 print(f'epoch {epoch + 1}, loss {loss_value_sum:f}')
#
#     return net

#
# def train_net_with_evaluation(X_train, y_train, X_test, y_test, data_iter=None, test_iter=None, net=None,
#                               loss=None, optimizer=None, lr=0.001, num_epochs=1000, batch_size=64,
#                               show_interval=10, draw='loss', if_immediately=True):
#     if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
#         X_train = torch.tensor(X_train.values, dtype=torch.float32)
#         y_train = torch.tensor(y_train.values, dtype=torch.float32)
#     if data_iter is None:
#         data_iter = load_array((X_train, y_train), batch_size)
#     if test_iter is None:
#         test_iter = load_array((X_test, y_test), batch_size, if_shuffle=False)
#     if net is None:
#         net = nn.Sequential(nn.Flatten(), nn.Linear(X_train.shape[1], 1))
#         net[1].weight.data.normal_(0, 0.01)
#     if loss is None:
#         loss = nn.MSELoss()
#     if optimizer is None:
#         optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#
#     train_loss_list = []
#     train_acc_list = []
#     test_acc_list = []
#     if if_immediately:
#         fig, ax = plt.subplots()
#
#     if draw == 'loss':
#         for epoch in range(num_epochs):
#             for X, y in data_iter:
#                 y_hat = net(X)  # 输入的X经过net所计算出的值
#                 loss_value = loss(y_hat, y)
#                 optimizer.zero_grad()  # 清除上一次的梯度值
#                 loss_value.sum().backward()  # 反向传播，求参数的梯度
#                 # for param in net.parameters():
#                 #     print(param.grad)
#                 optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
#             if epoch % show_interval == 0:
#                 loss_value = loss(net(X_train), y_train).detach()
#                 test_loss_value = loss(net(X_test), y_test).detach()
#                 print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
#                 train_loss_list.append(loss_value)
#                 train_acc_list.append(loss_value)
#                 test_acc_list.append(test_loss_value)
#                 if if_immediately:
#                     draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1,
#                                                       show_interval, content='loss')
#         if if_immediately:
#             plt.show()
#         else:
#             draw_Loss_or_Accuracy([train_acc_list, test_acc_list], num_epochs, show_interval, content='acc')
#     elif draw == 'acc':
#         for epoch in range(num_epochs):
#             train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
#             for X, y in data_iter:
#                 y_hat = net(X)  # 输入的X经过net所计算出的值
#                 loss_value = loss(y_hat, y)
#                 optimizer.zero_grad()  # 清除上一次的梯度值
#                 loss_value.sum().backward()  # 反向传播，求参数的梯度
#                 # for param in net.parameters():
#                 #     print(param.grad)
#                 optimizer.step()  # 步进 根据指定的优化算法进行参数的寻优
#                 n += y.shape[0]
#                 train_loss_sum += loss_value.item()
#                 train_acc_sum += count_correct_predictions(y_hat, y)
#             if epoch % show_interval == 0:
#                 loss_value = loss(net(X_train), y_train)
#                 print(f'epoch {epoch + 1}, loss {loss_value.sum():f}')
#                 train_loss_list.append(train_loss_sum / n)
#                 train_acc_list.append(train_acc_sum / n)
#                 test_acc_list.append(evaluate_accuracy(net, test_iter))
#                 if if_immediately:
#                     draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1,
#                                                       show_interval, content='acc')
#         if if_immediately:
#             plt.show()
#         else:
#             draw_Loss_or_Accuracy([train_acc_list, test_acc_list], num_epochs, show_interval, content='acc')
#     return net, train_loss_list, train_acc_list, test_acc_list
