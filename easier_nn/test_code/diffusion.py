# 在https://www.kaggle.com/code/vcxiaoman/diffusion/edit上运行，这里只作为存档

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


"""第四章"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


def show_images(x):
    """给定一批图像x，创建一个网格并将其转换为PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """给定一个PIL图像列表，将它们叠加成一行以便查看"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from diffusers import StableDiffusionPipeline
# https://huggingface.co/sd-dreambooth-library ，这里有来自社区的各种模型
model_id = "sd-dreambooth-library/mr-potato-head"  # 模型ID
# 加载管线
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16). to(device)
# sks是新引入的唯一标识符（Unique Identifier，UID），帮助模型识别和生成特定的内容
prompt = "an abstract oil painting of sks mr potato head by picasso"  # prompt
# num_inference_steps代表采样步骤的数量，通常设置为20到100之间的值。较高的值通常会产生更高质量的图像
# guidance_scale则决定模型的输出与提示语之间的匹配程度，设置在7.0到15.0之间。较高的值会使生成的图像更加符合提示词的描述
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=3).images
# 如果是一幅图  image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
# 就可以直接显示图像  plt.imshow(images)
# 现在是多幅图，将生成的图像转换为PIL格式并显示
output_image = make_grid(images, size=128)  # 调整size为合适的值
plt.imshow(output_image)
plt.axis('off')
plt.show()

from diffusers import DDPMPipeline
# 加载预设好的管线
butterfly_pipeline = DDPMPipeline.from_pretrained("johnowhitaker/ddpm-butterflies-32px").to(device)
# 生成8张图片
images = butterfly_pipeline(batch_size=8).images
# 输出图片
make_grid(images)


# 4.2 蝴蝶数据集加载
import torchvision
from datasets import load_dataset
from torchvision import transforms
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
# 也可以从本地文件夹中加载图像
# dataset = load_dataset("imagefolder", data_dir="path/to/folder")
# 我们将在32×32像素的正方形图像上进行训练，但你也可以尝试更大尺寸的图像
image_size = 32
# 如果GPU内存不足，你可以减小batch_size
batch_size = 64
# 定义数据增强过程
preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整大小
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.ToTensor(),  # 将张量映射到(0,1)区间
        transforms.Normalize([0.5], [0.5]),  # 映射到(-1, 1)区间
    ])


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
# 创建一个数据加载器，用于批量提供经过变换的图像
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 我们可以从中取出一批图像数据并进行可视化，代码如下：
xb = next(iter(train_dataloader))["images"].to(device)[:8]
print("X shape:", xb.shape)
show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)


# 4.3 噪声调度器
from diffusers import DDPMScheduler
# 创建一个DDPMScheduler对象，用于在训练过程中逐渐加入噪声
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# 仅添加了少量噪声
noise_scheduler1 = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
# 'cosine'调度方式，这种方式可能更适合尺寸较小的图像
noise_scheduler2 = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# 创建1*3的子图
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
# 选择第一个子图
axs[0].plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
axs[0].plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
axs[0].legend(fontsize="x-large")
# 选择第二个子图
axs[1].plot(noise_scheduler1.alphas_cumprod.cpu(), label=r"$\bar{\alpha}_t$")
axs[1].plot((1 - noise_scheduler1.alphas_cumprod.cpu()), label=r"$(1 - \bar{\alpha}_t)$")
axs[1].legend(fontsize="x-large")
# 选择第三个子图
axs[2].plot(noise_scheduler2.alphas_cumprod.cpu(), label=r"$\bar{\alpha}_t$")
axs[2].plot((1 - noise_scheduler2.alphas_cumprod.cpu()), label=r"$(1 - \bar{\alpha}_t)$")
axs[2].legend(fontsize="x-large")
plt.show()

timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
print("Noisy X shape", noisy_xb.shape)
show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)

# 4.4 训练DDPM模型
from diffusers import UNet2DModel
model = UNet2DModel(
    sample_size=image_size,  # 目标图像分辨率
    in_channels=3,  # 输入通道数，对于RGB图像来说，通道数为3
    out_channels=3,  # 输出通道数
    layers_per_block=2,  # 每个UNet块使用的ResNet层数
    block_out_channels=(64, 128, 128, 256),  # 更多的通道→更多的参数
    down_block_types=(
        "DownBlock2D",  # 一个常规的ResNet下采样模块
        "DownBlock2D",
        "AttnDownBlock2D",  # 一个带有空间自注意力的ResNet下采样模块
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # 一个带有空间自注意力的ResNet上采样模块
        "UpBlock2D",
        "UpBlock2D",  # 一个常规的ResNet上采样模块
    ),
)
model.to(device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
losses = []
for epoch in range(30):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # 为图片添加采样噪声
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]
        # 为每张图片随机采样一个时间步
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
        # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        # 获得模型的预测结果
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    if (epoch + 1) % 5 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()

# 4.5 生成图像
from diffusers import DDPMPipeline
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline_output = image_pipe()
pipeline_output.images[0]

sample = torch.randn(8, 3, 32, 32).to(device)  # 从8张随机噪声开始
for i, t in enumerate(noise_scheduler.timesteps):
    with torch.no_grad():
        residual = model(sample, t).sample  # 从模型中采样残差
    sample = noise_scheduler.step(residual, t, sample).prev_sample  # 逐步迈进

show_images(sample)


"""第五章"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")  # 256×256的CelebA-HQ模型
image_pipe.to(device)
images = image_pipe().images
images[0]

# 5.2 DDIM调度器
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)  # 设置推理步数，每40步采样一次

x = torch.randn(4, 3, 256, 256).to(device)  # Batch of 4, 3-channel 256 x 256 px images
# 逐步迈进
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)  # # 准备模型输入：给“带躁”图像加上时间步信息
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]  # 预测噪声
    scheduler_output = scheduler.step(noise_pred, t, x)  # 使用调度器计算更新后的样本应该是什么样子
    x = scheduler_output.prev_sample  # 更新输入图像
    # 每10步或在最后一步时，显示当前的输入图像x和预测的去噪图像
    if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
        axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        axs[0].set_title(f"Current x (step {i})")
        pred_x0 = scheduler_output.pred_original_sample
        grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
        axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        axs[1].set_title(f"Predicted denoised images (step {i})")
        plt.show()
# 也可以直接使用新的调度器替换原有管线中的调度器，然后进行采样，代码如下：
image_pipe.scheduler = scheduler
images = image_pipe(num_inference_steps=40).images
images[0]

# 5.3 微调
dataset_name = "huggan/smithsonian_butterflies_subset"  # @param
dataset = load_dataset(dataset_name, split="train")
image_size = 256  # @param
batch_size = 4  # @param
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
print("Previewing batch:")
batch = next(iter(train_dataloader))
grid = torchvision.utils.make_grid(batch["images"], nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)

# 训练
num_epochs = 2
lr = 1e-5
grad_accumulation_steps = 10  # 梯度累积的步数
optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)
losses = []

for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch["images"].to(device)
        # 为图片添加噪声
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]  # batch size
        # 随机选取一个时间步
        timesteps = torch.randint(0, image_pipe.scheduler.num_train_timesteps, (bs,),
                                  device=clean_images.device, ).long()
        # 前向扩散：根据选中的时间步和确定的幅值，在干净图像上添加噪声
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
        # 预测噪声与真实噪声之间的均方误差
        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        losses.append(loss.item())
        loss.backward(loss)
        # 梯度累积
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}")

plt.plot(losses)

# 生成图像
x = torch.randn(8, 3, 256, 256).to(device)
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)

# 5.4 引导
pipeline_name = "johnowhitaker/sd-class-wikiart-from-bedrooms"
image_pipe = DDPMPipeline.from_pretrained(pipeline_name).to(device)
scheduler = DDIMScheduler.from_pretrained(pipeline_name)
scheduler.set_timesteps(num_inference_steps=40)
x = torch.randn(8, 3, 256, 256).to(device)
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)


def color_loss(images, target_color=(102/255, 204/255, 255/255)):
    """给定一个RGB值，返回一个损失值，用于衡量图片的像素值与目标颜色相差多少；
    这里的目标颜色是一种浅蓝色，对应的RGB值为(102, 204, 255)"""
    target = (torch.tensor(target_color).to(images.device) * 2 - 1)  # 对target_color进行归一化，使它的取值区间为(-1, 1)
    target = target[None, :, None, None]  # 将所生成目标张量的形状改为(b, c, h, w)，以适配输入图像images的shape
    error = torch.abs(images - target).mean()  # 计算图片的像素值以及目标颜色的均方误差
    return error


# 第一种方法：使用损失函数的快捷方式
guidance_loss_scale = 40  # 用于决定引导的强度有多大
x = torch.randn(4, 3, 256, 256).to(device)
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = x.detach().requires_grad_()  # 与方法2相比，这里的x.requires_grad_()的位置不同，这里在计算模型预测之后设置
    x0 = scheduler.step(noise_pred, t, x).pred_original_sample
    loss = color_loss(x0) * guidance_loss_scale
    if i % 10 == 0:
        print(i, "loss:", loss.item())
    cond_grad = -torch.autograd.grad(loss, x)[0]
    x = x.detach() + cond_grad
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
Image.fromarray(np.array(im * 255).astype(np.uint8))

# 第二种方法：在计算模型预测之前设置x.requires_grad
guidance_loss_scale = 40
x = torch.randn(4, 3, 256, 256).to(device)
for i, t in tqdm(enumerate(scheduler.timesteps)):
    x = x.detach().requires_grad_()  # 与方法1相比，这里的x.requires_grad_()的位置不同，这里在计算模型预测之前设置
    model_input = scheduler.scale_model_input(x, t)
    noise_pred = image_pipe.unet(model_input, t)["sample"]
    x0 = scheduler.step(noise_pred, t, x).pred_original_sample
    loss = color_loss(x0) * guidance_loss_scale
    if i % 10 == 0:
        print(i, "loss:", loss.item())
    cond_grad = -torch.autograd.grad(loss, x)[0]
    x = x.detach() + cond_grad
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
Image.fromarray(np.array(im * 255).astype(np.uint8))


# CLIP引导
import open_clip
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.to(device)
# 图像变换：用于修改图像尺寸和增广数据，同时归一化数据，以使数据能够适配CLIP模型
tfms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪
    torchvision.transforms.RandomAffine(5),  # 随机扭曲图片
    torchvision.transforms.RandomHorizontalFlip(),  # 随机左右镜像
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),  # 归一化
])

# 定义一个损失函数，用于获取图片的特征，然后与提示文字的特征进行对比
def clip_loss(image, text_features):
    image_features = clip_model.encode_image(tfms(image))
    input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
    embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
    # Squared Great Circle Distance平方大圆距离(在单位球面上测量两个点之间的距离):
    # $$d({u},{v})^2=2\cdot\arcsin\left(\frac{\|{u}-{v}\|}2\right)^2$$
    dists = (input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2))
    return dists.mean()


prompt = "Red Rose (still life), red flower painting"  # 红玫瑰（静物），红花油画
guidance_scale = 8  # 引导尺度
n_cuts = 4  # 切割次数
scheduler.set_timesteps(50)  # 设置时间步数
text = open_clip.tokenize([prompt]).to(device)  # 将提示文字转换为张量
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = clip_model.encode_text(text)
x = torch.randn(4, 3, 256, 256).to(device)  # 从4张随机噪声图像开始
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    cond_grad = 0
    for cut in range(n_cuts):
        x = x.detach().requires_grad_()
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample  # 预测去噪图像
        loss = clip_loss(x0, text_features) * guidance_scale
        cond_grad -= torch.autograd.grad(loss, x)[0] / n_cuts  # 梯度累积
    if i % 5 == 0:
        print("Step:", i, ", Guidance loss:", loss.item())
    alpha_bar = scheduler.alphas_cumprod[i]  # 获取当前时间步的alpha_bar
    x = (x.detach() + cond_grad * alpha_bar.sqrt())  # 更新x
    x = scheduler.step(noise_pred, t, x).prev_sample  # 更新x
grid = torchvision.utils.make_grid(x.detach(), nrow=4)
im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
Image.fromarray(np.array(im * 255).astype(np.uint8))

# 引导尺度guidance_scale的选择。对于一般的是在开始引导比较好，对于纹理这种风格的大部分应该在结束的时候引导。
plt.plot([1 for a in scheduler.alphas_cumprod], label="no scaling")
plt.plot([a for a in scheduler.alphas_cumprod], label=r"$\alpha_{\bar{t}}$")
plt.plot([a.sqrt() for a in scheduler.alphas_cumprod], label=r"$\sqrt{\alpha_{\bar{t}}}$")
plt.plot([(1 - a).sqrt() for a in scheduler.alphas_cumprod], label=r"$\sqrt{1-\alpha_{\bar{t}}}$")
plt.legend(fontsize="x-large")
plt.title("Possible guidance scaling schedules")


# 5.5 创建一个类别条件扩散模型
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
# Load the dataset

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()
        # 把数字所属的类别映射到一个长度为class_emb_size的特征向量上
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=28,  # target image的尺寸
            in_channels=1 + class_emb_size,  # 加入额外的输入通道，用于接收类别信息
            out_channels=1,  # 输出结果的通道数
            layers_per_block=2,  # 设置一个UNet模块有多少个残差连接层
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # 常规的ResNet下采样模块
                "AttnDownBlock2D",  # 含有spatial self-attention的ResNet下采样模块
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # 含有spatial self-attention的ResNet上采样模块
                "UpBlock2D",  # 常规的ResNet上采样模块
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape  # 分别是batch size, channels, width, height
        # 类别条件将会以额外通道的形式输入到UNet中。x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
        class_cond = self.class_emb(class_labels)  # 将类别映射为向量形式并扩展成类似于(bs, 4, 28, 28)的张量形状
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # 将原始输入和类别条件信息拼接到一起
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')  # 创建一个调度器
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
n_epochs = 10
net = ClassConditionedUnet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        x = x.to(device) * 2 - 1  # (mapped to (-1, 1))
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        pred = net(noisy_x, timesteps, y)  # 预测结果，传入类别信息y
        loss = loss_fn(pred, noise)  # 预测结果与噪声之间的均方误差
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-100:]) / 100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
plt.plot(losses)

x = torch.randn(80, 1, 28, 28).to(device)  # x是随机的
y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)  # 但是y是有规律的，包含类别信息0~9
for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = net(x, t, y)
    x = noise_scheduler.step(residual, t, x).prev_sample
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')


"""第六章"""
# 6.1 稳定扩散
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline
    )


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


# 下载图片
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入管线
model_id = "stabilityai/stable-diffusion-2-1-base"  # 模型ID
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
generator = torch.Generator(device=device).manual_seed(42)  # 生成器，设置随机种子为42
# 运行这个管线
pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape",  # What to generate
    negative_prompt="Oversaturated, blurry, low quality",  # What NOT to generate
    height=480, width=640,  # 尺寸
    guidance_scale=8,  # 引导尺度
    num_inference_steps=35,  # 一次生成需要多少个推理步骤
    generator=generator  # 生成器
)
pipe_output.images[0]

# 比较不同的引导尺度
cfg_scales = [1.1, 8, 12]  # 不同的引导尺度
prompt = "A collie with a pink hat"
fig, axs = plt.subplots(1, len(cfg_scales), figsize=(16, 5))
for i, ax in enumerate(axs):
    im = pipe(prompt, height=480, width=480,
              guidance_scale=cfg_scales[i], num_inference_steps=35,
              generator=torch.Generator(device=device).manual_seed(42)).images[0]
    ax.imshow(im)
    ax.set_title(f'CFG Scale {cfg_scales[i]}')


# 6.2 其他管线
# Img2Img
model_id = "stabilityai/stable-diffusion-2-1-base"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
result_image = img2img_pipe(
    prompt="An oil painting of a man on a bench",
    image=init_image,  # The starting image
    strength=0.6,  # 0 for no change, 1.0 for max strength
).images[0]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(init_image)
axs[0].set_title('Input Image')
axs[1].imshow(result_image)
axs[1].set_title('Result')
plt.show()

# inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe = pipe.to(device)
prompt = "A small robot, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].imshow(init_image)
axs[0].set_title('Input Image')
axs[1].imshow(mask_image)
axs[1].set_title('Mask')
axs[2].imshow(image)
axs[2].set_title('Result')
plt.show()

# Depth2Img
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
pipe = pipe.to(device)
prompt = "An oil painting of a man on a bench"
image = pipe(prompt=prompt, image=init_image).images[0]
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(init_image)
axs[0].set_title('Input Image')
axs[1].imshow(image)
axs[1].set_title('Result')


"""第七章"""
# 7.1 反转
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler


def load_image(url, size=None):
    response = requests.get(url,timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
prompt = 'Beautiful DSLR Photograph of a penguin on the beach, golden hour'
negative_prompt = 'blurry, ugly, stock photo'
im = pipe(prompt, negative_prompt=negative_prompt).images[0]
im.resize((256, 256))
# 查看\alpha
timesteps = pipe.scheduler.timesteps.cpu()
alphas = pipe.scheduler.alphas_cumprod[timesteps]
plt.plot(timesteps, alphas, label='alpha_t')
plt.legend()


# sample：采样
@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None, guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt='', device=device):
    text_embeddings = pipe._encode_prompt(prompt, device, num_images_per_prompt,
                                          do_classifier_free_guidance, negative_prompt)  # 文本提示语的嵌入
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)  # 推理步数
    # 生成随机的起始潜在变量
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    latents = start_latents.clone()  # 潜在变量
    # 逐步迈进
    for i in tqdm(range(start_step, num_inference_steps)):
        t = pipe.scheduler.timesteps[i]
        # 准备模型输入
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # 预测噪声
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # 引导
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # 使用调度器更新步骤:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        # 手动实现：
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    # 后处理
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)
    return images


sample('Watercolor painting of a beach sunset', negative_prompt=negative_prompt,
       num_inference_steps=50)[0].resize((256, 256))


input_image = load_image('https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg', size=(512, 512))
input_image_prompt = "Photograph of a puppy on the grass"
# encode with VAE
with torch.no_grad(): latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
l = 0.18215 * latent.latent_dist.sample()


# invert：反转
@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80, num_images_per_prompt=1,
           do_classifier_free_guidance=True, negative_prompt='', device=device):
    text_embeddings = pipe._encode_prompt(prompt, device, num_images_per_prompt,
                                          do_classifier_free_guidance, negative_prompt)
    latents = start_latents.clone()  # 指定起始步骤
    intermediate_latents = []  # 保存反转的隐层
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)  # 反转的时间步

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        if i >= num_inference_steps - 1:
            continue  # 跳过最后一次迭代
        t = timesteps[i]
        # 如果正在进行CFG(classifier_free_guidance)，则对隐层进行扩展
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # 引导
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        # 反转的更新步(re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next).sqrt() * noise_pred
        intermediate_latents.append(latents)
    return torch.cat(intermediate_latents)


inverted_latents = invert(l, input_image_prompt,num_inference_steps=50)
inverted_latents.shape  # torch.Size([48, 4, 64, 64])
with torch.no_grad():
    im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
pipe.numpy_to_pil(im)[0]  # 显示反转的图像

pipe(input_image_prompt, latents=inverted_latents[-1][None], num_inference_steps=50, guidance_scale=3.5).images[0]

# 我们既可以使用更多的时间步来得到更准确的反转，也可以采取“作弊”的方式，直接从相应反转过程50步中的第20步的隐式表达开始，代码如下：
start_step = 20
sample(input_image_prompt, start_latents=inverted_latents[-(start_step + 1)][None],
       start_step=start_step, num_inference_steps=50)[0]

# 更换prompt
start_step = 10
new_prompt = input_image_prompt.replace('puppy', 'cat')
sample(new_prompt, start_latents=inverted_latents[-(start_step + 1)][None],
       start_step=start_step, num_inference_steps=50)[0]

# 直接对输入图像添加噪声，然后用新的文本提示语直接“去噪”
start_step = 10
num_inference_steps=50
pipe.scheduler.set_timesteps(num_inference_steps)
noisy_l = pipe.scheduler.add_noise(l, torch.randn_like(l), pipe.scheduler.timesteps[start_step])
sample(new_prompt, start_latents=noisy_l, start_step=start_step, num_inference_steps=num_inference_steps)[0]


# 组合封装
def edit(input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5):
    with torch.no_grad():
        latent = pipe.vae.encode( tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)
    final_im = sample(edit_prompt, start_latents=inverted_latents[-(start_step + 1)][None],
                      start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale)[0]
    return final_im


edit(input_image, 'A puppy on the grass', 'an old grey dog on the grass', num_steps=50, start_step=10)
edit(input_image, 'A puppy on the grass', 'A blue dog on the lawn', num_steps=50, start_step=12, guidance_scale=6)
# Inversion test with far more steps:
edit(input_image, 'A puppy on the grass', 'A puppy on the grass', num_steps=350, start_step=1)


# 7.2 ControlNet
# 7.2.1 戴珍珠耳环的少女
from diffusers.utils import load_image
image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
image

import cv2
from PIL import Image
import numpy as np
# 边缘提取
image = np.array(image)
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                         controlnet=controlnet, torch_dtype=torch.float16)
from diffusers import UniPCMultistepScheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
output = pipe(
    prompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    num_inference_steps=20,
    generator=generator,
)
image_grid(output.images, 2, 2)


# 7.2.2 瑜伽
urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
imgs = [
    load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url)
    for url in urls
]
image_grid(imgs, 2, 2)
# 提取瑜伽姿势
from controlnet_aux import OpenposeDetector
model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
poses = [model(img) for img in imgs]
image_grid(poses, 2, 2)

# 生成其他人物做瑜伽的图片
controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose",
                                             torch_dtype=torch.float16)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]
prompt = "super-hero character, best quality, extremely detailed"
output = pipe(
    [prompt] * 4,
    poses,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    generator=generator,
    num_inference_steps=20,
)
image_grid(output.images, 2, 2)


"""第八章"""
import torch, random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import Audio
from matplotlib import pyplot as plt
from diffusers import DiffusionPipeline
from torchaudio import transforms as AT
from torchvision import transforms as IT
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-instrumental-hiphop-256").to(device)

# 在管线中采样一次并将采样结果显示出来
output = pipe()
display(output.images[0])

display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))

# 音频序列
print(output.audios[0].shape)
# 输出的图像（频谱spectrogram）
print(output.images[0].size)

# 音频管线的额外组件
pipe.mel


# 使用torchaudio模块计算并绘制所生成音频样本的频谱
spec_transform = AT.Spectrogram(power=2)
spectrogram = spec_transform(torch.tensor(output.audios[0]))
print(spectrogram.min(), spectrogram.max())
log_spectrogram = spectrogram.log()
plt.imshow(log_spectrogram[0], cmap='gray')

a = pipe.mel.image_to_audio(output.images[0])  # shape: (130560,)
pipe.mel.load_audio(raw_audio=a)
im = pipe.mel.audio_slice_to_image(0)

sample_rate_pipeline = pipe.mel.get_sample_rate()  # 采样率22050

display(Audio(output.audios[0], rate=44100))  # 如果采样率错误，则播放速度被加倍，2x

# 载入数据集
from datasets import load_dataset
dataset = load_dataset('lewtun/music_genres', split='train')
# 查看该数据集中不同类别样本所占的比例：
for g in list(set(dataset['genre'])):
    print(g, sum(x == g for x in dataset['genre']))

audio_array = dataset[0]['audio']['array']
sample_rate_dataset = dataset[0]['audio']['sampling_rate']
print('Audio array shape:', audio_array.shape)  # (1323119,)
print('Sample rate:', sample_rate_dataset)  # 44100
display(Audio(audio_array, rate=sample_rate_dataset))

a = dataset[0]['audio']['array']  # 得到音频序列
pipe.mel.load_audio(raw_audio=a)  # 使用pipe.mel加载音频
pipe.mel.audio_slice_to_image(0)  # 输出第一幅频谱图像

sample_rate_dataset = dataset[0]['audio']['sampling_rate']  # 调整采样率
resampler = AT.Resample(sample_rate_dataset, sample_rate_pipeline, dtype=torch.float32)
to_t = IT.ToTensor()

# 音频片段转换为频谱张量
def to_image(audio_array):
    audio_tensor = torch.tensor(audio_array).to(torch.float32)
    audio_tensor = resampler(audio_tensor)
    pipe.mel.load_audio(raw_audio=np.array(audio_tensor))
    num_slices = pipe.mel.get_number_of_slices()
    slice_idx = random.randint(0, num_slices - 1)  # 每次随机取一张 (除了最后一张)
    im = pipe.mel.audio_slice_to_image(slice_idx)
    return im

# 首先将每个音频转换为频谱图像，然后将它们的张量堆叠起来
def collate_fn(examples):
    # 图像→张量→缩放至(-1,1)区间→堆叠：to image -> to tensor -> rescale to (-1, 1) -> stack into batch
    audio_ims = [to_t(to_image(x['audio']['array'])) * 2 - 1 for x in examples]
    return torch.stack(audio_ims)


# 创建一个只包含Chiptune/Glitch（芯片音乐/电子脉冲）风格的音乐
batch_size = 4
chosen_genre = 'Electronic'  # 尝试在不同的风格上进行训练
indexes = [i for i, g in enumerate(dataset['genre']) if g == chosen_genre]
filtered_dataset = dataset.select(indexes)
dl = torch.utils.data.DataLoader(filtered_dataset.shuffle(), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
batch = next(iter(dl))
print(batch.shape)
epochs = 3
lr = 1e-4
pipe.unet.train()
pipe.scheduler.set_timesteps(1000)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

for epoch in range(epochs):
    for step, batch in tqdm(enumerate(dl), total=len(dl)):
        clean_images = batch.to(device)
        bs = clean_images.shape[0]
        # 为每一张图片设置一个随机的时间步
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
        # 按照噪声调度器，在每个时间步为干净的图片加上噪声
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        noisy_images = pipe.scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = pipe.unet(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        del clean_images, noise, noise_pred, loss
        torch.cuda.empty_cache()
# 装载之前训练好的频谱样本
pipe = DiffusionPipeline.from_pretrained("johnowhitaker/Electronic_test").to(device)
output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=22050))
# 输入一个不同形状的起点噪声张量，得到一个更长的频谱样本
noise = torch.randn(1, 1, pipe.unet.sample_size[0],pipe.unet.sample_size[1]*4).to(device)
output = pipe(noise=noise)
display(output.images[0])
display(Audio(output.audios[0], rate=22050))