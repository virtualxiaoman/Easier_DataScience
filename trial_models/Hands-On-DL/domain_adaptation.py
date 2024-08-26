# 运行有问题，以后再看
"""
数据不匹配问题（domain mismatch 或者 domain shift）在机器学习和深度学习中是一个常见的挑战，尤其当训练集和测试集来自不同的分布时。你提到的场景中，训练集是现实世界的动物图像，而测试集是动漫的动物图像，这种问题可以通过以下几种方法来应对：

1. 领域自适应（Domain Adaptation）
领域自适应技术旨在将模型从一个领域（源领域，即真实动物图像）迁移到另一个领域（目标领域，即动漫动物图像）。常用的方法包括：

对抗性训练（Adversarial Training）：通过对抗性神经网络（如DANN）对源领域和目标领域的特征进行对齐，使得特征提取器学到的特征在两个领域都相似。
领域自适应正则化（Domain Adaptation Regularization）：在模型训练过程中加入领域自适应的正则项，使得模型的参数对不同领域的数据有更好的鲁棒性。
MMD（Maximum Mean Discrepancy）：通过最小化不同领域间的分布差异来学习领域不变的特征。
2. 风格迁移（Style Transfer）
虽然你无法在训练时直接使用动漫图像，但可以尝试将真实动物图像转换为动漫风格，然后进行训练。例如使用像CycleGAN这样的技术，将训练集中的真实动物图像转换为动漫风格的图像。模型在动漫风格图像上训练后再测试，可以更好地应对测试集的动漫图像。

3. 特征提取（Feature Extraction）
使用预训练的模型（如ResNet、VGG等）作为特征提取器，将图像转换为高层次的特征表示。然后在这些高层次的特征上训练一个轻量级的分类器，这样可以在一定程度上减少图像风格的影响。

4. 使用对比学习（Contrastive Learning）
对比学习可以通过在训练过程中引入合成的负样本来提升模型的泛化能力。这种方法可以让模型更好地区分不同风格或领域的数据。

5. 数据增强（Data Augmentation）
使用数据增强技术生成更多样化的训练数据，增加模型在不同图像风格上的鲁棒性。例如可以使用颜色抖动、图像模糊、随机裁剪等方式进行数据增强，虽然不能直接生成动漫风格的图像，但可以让模型更好地适应图像的多样性。

6. 使用领域不变特征（Domain-Invariant Features）
尽量设计模型以学习领域不变的特征，这些特征在不同领域上表现相似。比如通过多任务学习或者特征对齐的方法，让模型在多个相关任务中同时学习。

通过这些方法，可以在一定程度上缓解由于数据不匹配带来的模型性能下降问题。具体的选择可以根据你的数据特点和计算资源来确定。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 简单的领域分类器（Domain Classifier）
class DomainClassifier(nn.Module):
    def __init__(self, in_features):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉最后的分类层

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)

# 分类器
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

# 简易训练函数
def train_model(extractor, classifier, domain_classifier, dataloader_real, dataloader_anime, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        for (real_images, _), (anime_images, _) in zip(dataloader_real, dataloader_anime):
            # 获取数据
            real_images, anime_images = real_images.cuda(), anime_images.cuda()
            labels_real, labels_anime = torch.ones(real_images.size(0), 1).cuda(), torch.zeros(anime_images.size(0), 1).cuda()

            # 提取特征
            real_features = extractor(real_images)
            anime_features = extractor(anime_images)

            # 计算分类损失
            real_preds = classifier(real_features)
            anime_preds = classifier(anime_features)

            # 计算领域对抗损失
            domain_preds_real = domain_classifier(real_features.detach())
            domain_preds_anime = domain_classifier(anime_features.detach())
            domain_loss = criterion(domain_preds_real, labels_real) + criterion(domain_preds_anime, labels_anime)

            # 优化特征提取器和分类器
            optimizer.zero_grad()
            loss = criterion(real_preds, labels_real) + criterion(anime_preds, labels_anime) + domain_loss
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_real = datasets.FakeData(transform=transform)  # 用FakeData代替真实数据集
dataset_anime = datasets.FakeData(transform=transform)  # 用FakeData代替动漫数据集

dataloader_real = DataLoader(dataset_real, batch_size=32, shuffle=True)
dataloader_anime = DataLoader(dataset_anime, batch_size=32, shuffle=True)

# 模型初始化
feature_extractor = FeatureExtractor().cuda()
classifier = Classifier(512, 2).cuda()  # 假设有2个分类
domain_classifier = DomainClassifier(512).cuda()

# 优化器和损失函数
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters()), lr=0.001)
criterion = nn.BCELoss()

# 训练模型
train_model(feature_extractor, classifier, domain_classifier, dataloader_real, dataloader_anime, optimizer, criterion)
