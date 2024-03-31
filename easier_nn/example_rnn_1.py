import matplotlib.pyplot as plt
import torch
from torch import nn
from easier_nn.classic_dataset import VirtualDataset as VD
from easier_nn.load_data import load_array
from easier_nn.train_net import train_net
from easier_excel.draw_data import plot_xys, plot_xy

vd = VD(end=1000)
vd.sinx(noise_sigma=0.15, show_plt=False)

def get_train_iter(train_class, tau=4, batch_size=16, n_train=600):
    features = torch.zeros((train_class.num_points - tau, tau))
    for i in range(tau):
        features[:, i] = train_class.y[i: train_class.num_points - tau + i]
    labels = train_class.y[tau:].reshape((-1, 1))
    train_iter = load_array((features[:n_train], labels[:n_train]), batch_size, if_shuffle=True)
    return train_iter, features, labels

tau = 32
n_train = 600
train_iter, features, labels = get_train_iter(vd, tau=tau, n_train=n_train)  # 大小分别是[996, 4]，[996, 1]
print(features.shape, labels.shape)

# net = nn.Sequential(nn.Linear(tau, 20), nn.ReLU(), nn.Linear(20, 8), nn.ReLU(), nn.Linear(8, 1))
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
# net.apply(init_weights)
# loss = nn.MSELoss(reduction='none')
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# net = train_net(features, labels, data_iter=train_iter, net=net, loss=loss, optimizer=optimizer, num_epochs=300)
# # 保存net模型
# torch.save(net, '../model/test/rnn_predict_net_tau=32.pth')
# 加载net模型
net = torch.load('../model/test/rnn_predict_net_tau=32.pth')

onestep_preds = net(features)
plot_xys(x=vd.x.detach().numpy()[tau:], y_list=[vd.y.detach().numpy()[tau:], onestep_preds.detach().numpy()],
         labels=['data', '1-step preds'], alpha=0.6)
multistep_preds = torch.zeros(vd.num_points)
multistep_preds[: n_train + tau] = vd.y[: n_train + tau]
for i in range(n_train + tau, vd.num_points):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))
plot_xys(x=vd.x.detach().numpy(), y_list=[vd.y.detach().numpy(), multistep_preds.detach().numpy()],
         labels=['data', 'multistep preds'], alpha=0.6)

max_steps = 64
features = torch.zeros((vd.num_points - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = vd.y[i: i + vd.num_points - tau - max_steps + 1]
# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
colors = ['blue', 'green', 'red', 'black', 'purple', 'pink', 'orange', 'cyan']
linstyles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
# y_list的列表的第1个元素是vd.y，后面的2~i+1个元素是features的第i列
y_list = [vd.y.detach().numpy()] + [features[:, tau + i - 1].detach().numpy() for i in steps]
fig, ax = plt.subplots(figsize=(10, 6))
ax = plot_xy(x=vd.x.detach().numpy(), y=y_list[0], label='data', use_ax=True, ax=ax, show_plt=False, alpha=0.3)
for i, n in enumerate(steps):
    ax = plot_xy(x=vd.x.detach().numpy()[tau + n - 1: vd.num_points - max_steps + n], y=y_list[i+1], alpha=0.8,
                 label=f'{n}-step preds', use_ax=True, ax=ax, show_plt=False,
                 color=colors[i+1], linestyle=linstyles[i+1])

# plot_xys(x=vd.x.detach().numpy()[tau - 1: vd.num_points - max_steps + 1],
#          y_list=[vd.y.detach().numpy()] + [features[:, tau + i - 1].detach().numpy() for i in steps],
#          labels=['data'] + [f'{i}-step preds' for i in steps])
plt.show()
