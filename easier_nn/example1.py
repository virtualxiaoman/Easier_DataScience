# mnist数据集，分类
from torch import nn
import torch
from easier_nn.classic_dataset import load_mnist, fashion_mnist
from easier_nn.classic_net import CNN_Net_for_Mnist, Unet, Depth_CNN_Net_for_Mnist, ResNet, ResidualBlock
from easier_nn.train_net import NetTrainer

X, y = load_mnist(flatten=False)  # 载入数据集，设置flatten是为了转换为(N, 1, 28, 28)以适应CNN


# 你可以依次尝试下面的网络，不同的网络所需要的显存、时间、效果是不同的
net = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)  # ResNet18，需要改为X, y = load_mnist(flatten=True)
# net = CNN_Net_for_Mnist()  # 需要改为X, y = load_mnist(flatten=True)
# CNN_Net_for_Mnist这个net我电脑跑不动，复制到kaggle上跑出来的结果是：
# Epoch 1/50, Train Acc: 0.982, Test Acc: 0.9817142857142858
# Epoch 21/50, Train Acc: 0.9996607142857142, Test Acc: 0.9932142857142857
# Epoch 41/50, Train Acc: 0.9997321428571428, Test Acc: 0.9925714285714285
# net = Depth_CNN_Net_for_Mnist()  # 需要改为X, y = load_mnist(flatten=True)
# net = Unet()  # 该网络目前似乎有问题，另外还需要设置NetTrainer的target_reshape_1D为False
# 你也可以使用下面的网络进行简单测试(acc其实也有90%多)，但是在读入时应该设为X, y = load_mnist(flatten=False)
# net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(X.shape[1], 64),
#     nn.ReLU(),
#     nn.Linear(64, 10),
# )
# net[1].weight.data.normal_(0, 0.01)
# net[1].bias.data.fill_(0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

trainer = NetTrainer(X, y, net, loss_fn, optimizer, epochs=2, net_type="acc", batch_size=16, target_reshape_1D=False,
                     print_interval=1, eval_during_training=False)
trainer.view_parameters()
trainer.train_net()
acc = trainer.evaluate_net(delete_train=True)  # delete_train=True表示删除训练集，只保留测试集
print(f"Accuracy: {acc}")
# torch.save(trainer.net, '../model/mnist/mnist_model_small.pth')
