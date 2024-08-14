# 受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 定义RBM类
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)  # 权重矩阵
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # 可见层偏置
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # 隐藏层偏置

    def sample_from_p(self, p):
        return torch.bernoulli(p)  # 采样函数

    def visible_to_hidden(self, v):
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)  # 计算隐藏层激活概率
        return p_h_given_v, self.sample_from_p(p_h_given_v)  # 返回概率和采样值

    def hidden_to_visible(self, h):
        p_v_given_h = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)  # 计算可见层激活概率
        return p_v_given_h, self.sample_from_p(p_v_given_h)  # 返回概率和采样值

    def forward(self, v):
        p_h, h = self.visible_to_hidden(v)
        p_v, v = self.hidden_to_visible(h)
        return v

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W.t()) + self.h_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return -hidden_term - vbias_term


# 数据加载和预处理
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# 定义RBM模型
n_visible = 28 * 28  # 可见层节点数
n_hidden = 128  # 隐藏层节点数
rbm = RBM(n_visible, n_hidden)

# 训练RBM模型
n_epochs = 5
learning_rate = 0.1
optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    loss_ = []
    for _, (data, _) in enumerate(train_loader):
        data = data.view(-1, n_visible)  # 展平成二维
        sample_data = data.bernoulli()

        v = sample_data
        v1 = rbm(v)
        ph0, _ = rbm.visible_to_hidden(v)
        ph1, _ = rbm.visible_to_hidden(v1)

        loss = torch.mean(rbm.free_energy(v)) - torch.mean(rbm.free_energy(v1))
        loss_.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}: Loss = {torch.mean(torch.FloatTensor(loss_))}')

# 可视化部分重建图像
with torch.no_grad():
    sample_data = next(iter(train_loader))[0].view(-1, n_visible)
    v = sample_data.bernoulli()
    v_reconstructed = rbm(v).view(-1, 28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axes[0, i].imshow(v[i].view(28, 28).numpy(), cmap='gray')
        axes[1, i].imshow(v_reconstructed[i].numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.show()
