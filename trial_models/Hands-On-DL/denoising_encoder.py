import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 自编码器模型
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 将图像展平成一维向量
])

train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 实例化模型、损失函数和优化器
model = DenoisingAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练降噪自编码器
n_epochs = 5
for epoch in range(n_epochs):
    for data, _ in train_loader:
        # 向输入数据添加噪声
        noisy_data = data + 0.5 * torch.randn_like(data)
        noisy_data = torch.clamp(noisy_data, 0., 1.)  # 保证数据在[0, 1]范围内

        # 前向传播
        output = model(noisy_data)
        loss = criterion(output, data)  # 损失计算

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

# 可视化部分原始图像、噪声图像和去噪后的图像
with torch.no_grad():
    sample_data = next(iter(train_loader))[0][:10]
    noisy_sample_data = sample_data + 0.5 * torch.randn_like(sample_data)
    noisy_sample_data = torch.clamp(noisy_sample_data, 0., 1.)
    output = model(noisy_sample_data)

    fig, axes = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(10):
        axes[0, i].imshow(sample_data[i].view(28, 28).numpy(), cmap='gray')
        axes[1, i].imshow(noisy_sample_data[i].view(28, 28).numpy(), cmap='gray')
        axes[2, i].imshow(output[i].view(28, 28).numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    plt.show()
