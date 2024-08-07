import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
import time


# 下面四个函数不够好，可以使用easier_nn.train_net中的函数
# 评估函数没有考虑到device的问题
# 绘图函数占用了太多的空间，使得训练的时候变慢了很多
# def evaluate_accuracy(net, data_iter):
#     """计算分类模型的准确率"""
#     net.eval()  # 将模型设置为评估模式
#     accuracy_sum, n = 0, 0
#     with torch.no_grad():
#         for X, y in data_iter:
#             accuracy_sum += count_correct_predictions(net(X), y)
#             n += y.numel()
#     net.train()  # 恢复模型为训练模式
#     return accuracy_sum / n
#
#
# def count_correct_predictions(y_hat, y):
#     """计算分类正确的数量"""
#     predictions = torch.argmax(y_hat, dim=1).type(y.dtype)
#     correct = (predictions == y).sum().item()
#     return correct
#
#
# def draw_Loss_or_Accuracy(y, epochs, show_interval, content='loss'):
#     """
#     :param y: shape：[2, epochs]
#     :param epochs: int类型的数值
#     :param show_interval: 显示间隔
#     :param content:
#         content='loss' -> 传入y=[train_loss_values, test_acc_values]和epochs的值，绘制Loss
#         content='acc' -> 传入y=[train_acc_values, test_acc_values]和epochs的值，绘制Accuracy"""
#     x = range(1, epochs + 1, show_interval)
#     fig, ax = plt.subplots()
#     if content == 'loss':
#         ax.plot(x, y[0], label='Train Loss')
#         ax.plot(x, y[1], label='Test Loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.set_title('Training & Test  Loss')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     elif content == 'acc':
#         ax.plot(x, y[0], label='Train Acc')
#         ax.plot(x, y[1], label='Test Acc')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Acc')
#         ax.set_title('Training & Test  Acc')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#         ax.yaxis.set_major_formatter(FuncFormatter(lambda y_axis, _: '{:.0%}'.format(y_axis)))
#     plt.show()
#     plt.close()
#
# def draw_Loss_or_Accuracy_immediately(ax, y, epoch, show_interval, content='loss'):
#     """
#     [使用示例]：
#         fig, ax = plt.subplots()
#         draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1, show_interval, content='acc')
#     :param ax: matplotlib.axes._subplots.AxesSubplot对象，用于绘制图像
#     :param y: shape[2, epoch]
#     :param epoch: int类型的数值，表示当前的epoch数
#     :param show_interval: 显示间隔
#     :param content:
#         content='loss' -> 传入y=[train_loss_values, test_acc_values]，绘制Loss
#         content='acc' -> 传入y=[train_acc_values, test_acc_values]，绘制Accuracy
#     """
#     x = range(1, epoch + 1, show_interval)
#     ax.clear()  # 清除之前的绘图
#     if content == 'loss':
#         ax.plot(x, y[0], label='Train Loss')
#         ax.plot(x, y[1], label='Test Loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.set_title('Training & Test Loss')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     elif content == 'acc':
#         ax.plot(x, y[0], label='Train Acc')
#         ax.plot(x, y[1], label='Test Acc')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Acc')
#         ax.set_title('Training & Test Acc')
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#         ax.yaxis.set_major_formatter(FuncFormatter(lambda y_axis, _: '{:.0%}'.format(y_axis)))
#     plt.pause(0.1)



