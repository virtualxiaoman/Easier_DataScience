import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
import time


def evaluate_accuracy(net, data_iter):
    """计算分类模型的准确率"""
    net.eval()  # 将模型设置为评估模式
    accuracy_sum, n = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            accuracy_sum += count_correct_predictions(net(X), y)
            n += y.numel()
    net.train()  # 恢复模型为训练模式
    return accuracy_sum / n


def count_correct_predictions(y_hat, y):
    """计算分类正确的数量"""
    predictions = torch.argmax(y_hat, dim=1).type(y.dtype)
    correct = (predictions == y).sum().item()
    return correct


def draw_Loss_or_Accuracy(y, epochs, show_interval, content='loss'):
    """
    :param y: shape：[2, epochs]
    :param epochs: int类型的数值
    :param show_interval: 显示间隔
    :param content:
        content='loss' -> 传入y=[train_loss_values, test_acc_values]和epochs的值，绘制Loss
        content='acc' -> 传入y=[train_acc_values, test_acc_values]和epochs的值，绘制Accuracy"""
    x = range(1, epochs + 1, show_interval)
    fig, ax = plt.subplots()
    if content == 'loss':
        ax.plot(x, y[0], label='Train Loss')
        ax.plot(x, y[1], label='Test Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Test  Loss')
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    elif content == 'acc':
        ax.plot(x, y[0], label='Train Acc')
        ax.plot(x, y[1], label='Test Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Acc')
        ax.set_title('Training & Test  Acc')
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y_axis, _: '{:.0%}'.format(y_axis)))
    plt.show()
    plt.close()

def draw_Loss_or_Accuracy_immediately(ax, y, epoch, show_interval, content='loss'):
    """
    [使用示例]：
        fig, ax = plt.subplots()
        draw_Loss_or_Accuracy_immediately(ax, [train_acc_list, test_acc_list], epoch + 1, show_interval, content='acc')
    :param ax: matplotlib.axes._subplots.AxesSubplot对象，用于绘制图像
    :param y: shape[2, epoch]
    :param epoch: int类型的数值，表示当前的epoch数
    :param show_interval: 显示间隔
    :param content:
        content='loss' -> 传入y=[train_loss_values, test_acc_values]，绘制Loss
        content='acc' -> 传入y=[train_acc_values, test_acc_values]，绘制Accuracy
    """
    x = range(1, epoch + 1, show_interval)
    ax.clear()  # 清除之前的绘图
    if content == 'loss':
        ax.plot(x, y[0], label='Train Loss')
        ax.plot(x, y[1], label='Test Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Test Loss')
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    elif content == 'acc':
        ax.plot(x, y[0], label='Train Acc')
        ax.plot(x, y[1], label='Test Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Acc')
        ax.set_title('Training & Test Acc')
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y_axis, _: '{:.0%}'.format(y_axis)))
    plt.pause(0.1)


def show_images(imgs, num_rows, num_cols, titles=None):
    """绘制图像列表"""
    _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()  # 以num_rows=3,num_cols=6为例，flatten将(3, 6)的axes转换成(18,)，这样更方便使用单个索引来访问每个子图轴对象
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 通过对当前的子图轴对象ax的操作直接影响了axes中相应位置的元素
        if torch.is_tensor(img):
            # 图片张量
            img = img.detach().cpu().numpy()
            ax.imshow(img)
        else:
            # PIL图片
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    plt.close()


