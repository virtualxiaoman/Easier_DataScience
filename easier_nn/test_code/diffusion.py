# 在https://www.kaggle.com/code/vcxiaoman/diffusion/edit上运行

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True,
                                     transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')


def corrupt(x, amount):
    """根据amount为输入x加入噪声，这就是退化过程"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # 整理形状以保证广播机制不出错
    return x * (1 - amount) + noise * amount


# 绘制输入数据
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
# 加入噪声
amount = torch.linspace(0, 1, x.shape[0])  # 从0到1 → 退化更强烈了
noised_x = corrupt(x, amount)
# 绘制加噪版本的图像
axs[1].set_title('Corrupted data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')


class BasicUNet(nn.Module):
    """一个十分简单的UNet网络部署"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # 激活函数
        self.downscale = nn.MaxPool2d(2)  # 下采样
        self.upscale = nn.Upsample(scale_factor=2)  # 上采样

    def forward(self, x):
        h = []  # 用于存储下采样过程中的数据
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # 通过运算层与激活函数
            if i < 2:  # 选择除了第3层（最后一层）以外的层
                h.append(x)  # 排列供残差连接使用的数据
                x = self.downscale(x)  # 进行下采样以适配下一层的输入
        for i, l in enumerate(self.up_layers):
            if i > 0:  # 选择除了第1个上采样层以外的层
                x = self.upscale(x)  # Upscale上采样
                x += h.pop()  # 得到之前排列好的供残差连接使用的数据
            x = self.act(l(x))  # 通过运算层与激活函数
        return x


net = BasicUNet()
x = torch.rand(8, 1, 28, 28)
print(net(x).shape)  # torch.Size([8, 1, 28, 28])
print(sum([p.numel() for p in net.parameters()]))  # 309057

# 训练网络
batch_size = 128
n_epochs = 3
net = BasicUNet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
losses = []

for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)  # 将数据加载到GPU
        noise_amount = torch.rand(x.shape[0]).to(device)  # 随机选取
        noisy_x = corrupt(x, noise_amount)  # 创建“带噪”的输入noisy_x
        pred = net(noisy_x)
        loss = loss_fn(pred, x)  # 输出与真实“干净”的x有多接近？
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # 输出在每个周期训练得到的损失的均值
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss}')
# 查看损失曲线
plt.plot(losses)
plt.ylim(0, 0.1)

# 可视化模型在“带噪”输入上的表现
x, y = next(iter(train_dataloader))
x = x[:8]  # 为了便于展示，只选取前8条数据
amount = torch.linspace(0, 1, x.shape[0])  # 从0到1→退化更强烈了
noised_x = corrupt(x, amount)
# 得到模型的预测结果
with torch.no_grad():
    preds = net(noised_x.to(device)).detach().cpu()
# 绘图
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Corrupted data')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')

# 逐步迈进的扩散过程
# 采样策略：把采样过程拆解为5步，每次只前进一步
n_steps = 5
x = torch.rand(8, 1, 28, 28).to(device)  # 从完全随机的值开始
step_history = [x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():  # 在推理时不需要考虑张量的导数
        pred = net(x)  # 预测“去噪”后的图像
    pred_output_history.append(pred.detach().cpu())
    # 将模型的输出保存下来，以便后续绘图时使用
    mix_factor = 1 / (n_steps - i)  # 设置朝着预测方向移动的比例
    x = x * (1 - mix_factor) + pred * mix_factor  # 移动过程
    step_history.append(x.detach().cpu())  # 记录每一次移动，以便后续绘图

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
axs[0, 0].set_title('x (model input)')
axs[0, 1].set_title('model prediction')
for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')

model = UNet2DModel(
    sample_size=28,  # 目标图像的分辨率
    in_channels=1,  # 输入图像的通道数，RGB图像的通道数为3
    out_channels=1,  # 输出图像的通道数
    layers_per_block=2,  # 设置要在每一个UNet块中使用多少个ResNet层
    block_out_channels=(32, 64, 64),  # 与BasicUNet模型的配置基本相同
    down_block_types=(
        "DownBlock2D",  # 标准的ResNet下采样模块
        "AttnDownBlock2D",  # 带有空域维度self-att的ResNet下采样模块
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # 带有空域维度self-att的ResNet上采样模块
        "UpBlock2D",  # 标准的ResNet上采样模块
    ),
)
# print(model)
print(sum([p.numel() for p in model.parameters()]))


batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
n_epochs = 3
# Create the network
net = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
net.to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)  # Data on the GPU
        noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x = corrupt(x, noise_amount)  # Create our noisy x
        pred = net(noisy_x, 0).sample  # <<< Using timestep 0 always, adding .sample
        loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

# Plot losses and some samples
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Losses
axs[0].plot(losses)
axs[0].set_ylim(0, 0.1)
axs[0].set_title('Loss over time')

# 逐步迈进的扩散过程
# 采样策略：把采样过程拆解为5步，每次只前进一步
n_steps = 5
x = torch.rand(8, 1, 28, 28).to(device)  # 从完全随机的值开始
step_history = [x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():  # 在推理时不需要考虑张量的导数
        pred = net(x, 0).sample  # .sample是为了从模型的输出分布中采样
    pred_output_history.append(pred.detach().cpu())
    # 将模型的输出保存下来，以便后续绘图时使用
    mix_factor = 1 / (n_steps - i)  # 设置朝着预测方向移动的比例
    x = x * (1 - mix_factor) + pred * mix_factor  # 移动过程
    step_history.append(x.detach().cpu())  # 记录每一次移动，以便后续绘图

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
axs[0, 0].set_title('x (model input)')
axs[0, 1].set_title('model prediction')
for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5,label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")

# 可视化：DDPM加噪过程中的不同时间步
# 对一批图片加噪，看看效果
fig, axs = plt.subplots(3, 1, figsize=(16, 10))
xb, yb = next(iter(train_dataloader))
xb = xb.to(device)[:8]
xb = xb * 2. - 1.  # 映射到(-1,1)
print('X shape', xb.shape)

# 展示干净的原始输入
axs[0].imshow(torchvision.utils.make_grid(xb[:8])[0].detach().cpu(), cmap='Greys')
axs[0].set_title('Clean X')
# 使用调度器加噪
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)  # <<注意是使用randn而不是rand
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
print('Noisy X shape', noisy_xb.shape)
# 展示“带噪”版本（使用或不使用截断函数clipping）
axs[1].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu().clip(-1, 1), cmap='Greys')
axs[1].set_title('Noisy X (clipped to (-1, 1))')
axs[2].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu(), cmap='Greys')
axs[2].set_title('Noisy X')

print(x.shape)  # torch.Size([8, 1, 28, 28])
print(noised_x.shape)  # torch.Size([8, 1, 28, 28])

noise = torch.randn_like(xb)  # << 注意是使用randn而不是rand。randn是标准正态分布，rand是均匀分布
noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
model_prediction = model(noisy_x, timesteps).sample
loss = nn.MSELoss(model_prediction, noise)  # 预测结果与噪声
