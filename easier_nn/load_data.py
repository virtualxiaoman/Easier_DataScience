import torch
from torch.utils.data import TensorDataset, DataLoader


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
    [easy函数] 将训练集转为DataLoader
    :return: DataLoader数据类型
    """
    if y_reshape:
        return load_array((X_train, y_train.reshape(-1, 1)), batch_size)
    else:
        return load_array((X_train, y_train), batch_size)

def testset_to_dataloader(X_test, y_test, batch_size=64, y_reshape=False):
    """
    [easy函数] 将测试集转为DataLoader
    :return: DataLoader数据类型
    """
    if y_reshape:
        return load_array((X_test, y_test.reshape(-1, 1)), batch_size, if_shuffle=False)
    else:
        return load_array((X_test, y_test), batch_size, if_shuffle=False)




