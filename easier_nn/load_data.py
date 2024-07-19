import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# 本文件最好不要再调用了，但是由于历史遗留问题，现在还不能完全删除。
def load_array(data_arrays, batch_size=64, if_shuffle=True):
    """
    [底层函数] 构造一个PyTorch数据迭代器
    [使用示例]
        load_array((features, labels), batch_size)
        load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    :param data_arrays: 一个包含数据数组的元组或列表。通常包括输入特征和对应的标签(features, labels)。
    :param batch_size: 每个小批量样本的数量。
    :param if_shuffle: True数据将被随机洗牌(用于训练);False数据将按顺序提供(用于模型的评估或测试)。
    """
    dataset = TensorDataset(*data_arrays)  # 将数据数组转换为TensorDataset对象(将数据存储为Tensor对象，并允许按索引访问)
    return DataLoader(dataset, batch_size, shuffle=if_shuffle)


def trainset_to_dataloader(X_train, y_train, batch_size=64, y_reshape=False):
    """
    将训练集转为DataLoader
    :return: DataLoader数据类型
    """
    # 要注意X_train的type是DataFrame
    if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
    if isinstance(X_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
        pass
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
    if y_reshape:
        return load_array((X_train, y_train.reshape(-1, 1)), batch_size)
    else:
        return load_array((X_train, y_train), batch_size)

def testset_to_dataloader(X_test, y_test, batch_size=64, y_reshape=False):
    """
    将测试集转为DataLoader
    :return: DataLoader数据类型
    """
    if isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.DataFrame):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    if isinstance(X_test, torch.Tensor) and isinstance(y_test, torch.Tensor):
        pass
    else:
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
    if y_reshape:
        return load_array((X_test, y_test.reshape(-1, 1)), batch_size, if_shuffle=False)
    else:
        return load_array((X_test, y_test), batch_size, if_shuffle=False)




