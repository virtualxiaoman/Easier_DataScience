from sklearn.datasets import fetch_openml
import torch
import torchvision
from torchvision.datasets import FashionMNIST
import numpy as np
import re

from easier_nn.evaluate_net import show_images
from easier_nn.load_data import trainset_to_dataloader, testset_to_dataloader
from easier_excel.draw_data import plot_xy


def load_mnist(if_reshape_X=False, print_shape=False):
    """
    载入mnist数据集
    [使用示例]:
        X, y = load_mnist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
        X_train = X_train.float()
        X_test = X_test.float()
        train_iter = trainset_to_dataloader(X_train, y_train)
        test_iter = testset_to_dataloader(X_test, y_test)
    :return: data, target
    """
    mnist = fetch_openml('MNIST_784', parser='auto')
    data = mnist['data']
    target = mnist['target']
    data = np.array(data)  # 将DataFrame转换为NumPy数组
    target = np.array(target, dtype=np.int64)
    data = torch.tensor(data)  # 转换为PyTorch Tensor对象
    if if_reshape_X:
        data = data.reshape(-1, 28, 28)
    target = torch.tensor(target)
    if print_shape:
        print(target.shape)  # torch.Size([70000])
        print(data.shape)  # torch.Size([70000, 784])
    return data, target


class fashion_mnist:
    """
    载入
    [使用示例]:
        fm = fashion_mnist()
        fm.load_fashion_mnist()
        train_iter, test_iter = fm.load_dataiter()
    """
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_fashion_mnist(self, data_path='data', flatten=True):
        transform = torchvision.transforms.ToTensor()  # 定义数据变换

        train_dataset = FashionMNIST(root=data_path, train=True, download=True, transform=transform)
        self.X_train = train_dataset.data
        self.y_train = train_dataset.targets
        test_dataset = FashionMNIST(root=data_path, train=False, download=True, transform=transform)
        self.X_test = test_dataset.data
        self.y_test = test_dataset.targets

        if flatten:
            # 调整数据维度（由 (N, 28, 28) 转换为 (N, 784)）
            self.X_train = self.X_train.view(self.X_train.size(0), -1)
            self.X_test = self.X_test.view(self.X_test.size(0), -1)
        else:
            # 调整数据维度（由 (N, 28, 28) 转换为 (N, 1, 28, 28)）
            self.X_train = self.X_train.view(self.X_train.size(0), 1, 28, 28)
            self.X_test = self.X_test.view(self.X_test.size(0), 1, 28, 28)
        # 归一化像素值至 0 到 1 之间
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0
        # print(self.X_train.shape)  # torch.Size([60000, 784])
        # print(self.y_train.shape)  # torch.Size([60000])

    def load_dataiter(self, batch_size=256):
        self.train_iter = trainset_to_dataloader(self.X_train, self.y_train, batch_size=batch_size)
        self.test_iter = testset_to_dataloader(self.X_test, self.y_test, batch_size=batch_size)
        return self.train_iter, self.test_iter

    def predict(self, net, test_iter, n=6, num_rows=2, num_cols=3):
        """对前n个验证集样本进行预测"""
        X, y = next(iter(test_iter))
        trues = self.get_fashion_mnist_labels(y)
        preds = self.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = ["True:" + true + '\n' + "Pred:" + pred for true, pred in zip(trues, preds)]
        show_images(X[0:n].reshape((n, 28, 28)), num_rows, num_cols, titles=titles[0:n])

    @staticmethod
    def get_fashion_mnist_labels(labels):
        """输入数值，返回文本标签：t‐shirt T恤、trouser裤子、pullover套衫、dress连衣裙、coat外套、sandal凉鞋、shirt衬衫、
        sneaker运动鞋、bag包、ankle boot短靴"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[i] for i in labels]


def load_time_machine(path='data/timemachine.txt', raw=True, show_details=False):
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'r') as f:
        lines = f.readlines()
    if show_details:
        print(f'文本总行数: {len(lines)}')
        print("第1行：", lines[0])
        print("第10行：", lines[10])
    if raw:
        return content
    else:
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


class VirtualDataset:
    def __init__(self, start=1, end=100, num_points=1000):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.x = torch.linspace(self.start, self.end, self.num_points)
        self.y = None

    def sinx(self, w=0.01, noise_mu=0, noise_sigma=0.2, show_plt=False):
        noise = torch.normal(noise_mu, noise_sigma, (self.num_points,))
        self.y = torch.sin(w*self.x) + noise
        if show_plt:
            plot_xy(self.x.numpy(), self.y.numpy())
