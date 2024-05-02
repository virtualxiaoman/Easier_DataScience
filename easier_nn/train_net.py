import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

from easier_nn.load_data import load_array
from easier_nn.evaluate_net import evaluate_accuracy, count_correct_predictions, draw_Loss_or_Accuracy, \
    draw_Loss_or_Accuracy_immediately


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
