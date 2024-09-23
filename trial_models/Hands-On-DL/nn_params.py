# 查看训练过程中的nn变化
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from easier_nn.classic_dataset import load_mnist
from easier_nn.train_net import NetTrainer

X, y = load_mnist(flatten=True)  # 载入数据集

# 使用一个简单的网络
net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(X.shape[1], 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epoch = 10

N1_w_list, N1_b_list, N3_w_list, N3_b_list = [], [], [], []
ACC_list = []

trainer = NetTrainer(X, y, net, loss_fn, optimizer, epochs=1, eval_type="acc", batch_size=16, eval_interval=1)
for i in range(epoch):
    trainer.train_net()
    N1_w_list.append(trainer.net[1].weight.data.clone())
    N1_b_list.append(trainer.net[1].bias.data.clone())
    N3_w_list.append(trainer.net[3].weight.data.clone())
    N3_b_list.append(trainer.net[3].bias.data.clone())
    ACC_list.append(trainer.evaluate_net(delete_train=False))


# 展示训练过程中的参数变化
fig, axs = plt.subplots(4, epoch)
axs = axs.flatten()  # 将 axs 展平成一维数组
# 遍历每个 epoch
for i in range(epoch):
    # 在(1,i)绘制 N1_w_list
    sns.heatmap(N1_w_list[i].cpu().numpy(), ax=axs[i], cmap="RdYlBu_r", vmax=0.25, vmin=-0.25)
    axs[i].set_title(f'N1_w_list Epoch {i + 1}')
    # 在(2,i)绘制 N1_b_list
    sns.heatmap(N1_b_list[i].cpu().numpy().reshape(1, -1), ax=axs[i + epoch], cmap="RdYlBu_r", vmax=0.25, vmin=-0.25)
    axs[i + epoch].set_title(f'N1_b_list Epoch {i + 1}')
    # 在(3,i)绘制 N3_w_list
    sns.heatmap(N3_w_list[i].cpu().numpy(), ax=axs[i + epoch*2], cmap="RdYlBu_r", vmax=0.25, vmin=-0.25)
    axs[i + epoch*2].set_title(f'N3_w_list Epoch {i + 1}')
    # 在(4,i)绘制 N3_b_list
    sns.heatmap(N3_b_list[i].cpu().numpy().reshape(1, -1), ax=axs[i + epoch*3], cmap="RdYlBu_r", vmax=0.5, vmin=-0.5)
    axs[i + epoch*3].set_title(f'N3_b_list Epoch {i + 1}')

plt.tight_layout()
plt.show()

plt.plot(ACC_list)
plt.title("ACC_list")
plt.show()







