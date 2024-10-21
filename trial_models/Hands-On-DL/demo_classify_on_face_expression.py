import os
from PIL import Image
import torch
import torchvision
import torchvision.models as models
from torch import nn
from torchvision import datasets

from easier_nn.classic_dataset import DataTransform
from easier_nn.train_net import NetTrainer

epochs = 10
batch_size = 16
image_path = './input/face_data/The_expression_on_his_face/train'
test_img_path = r'./input/face_data/The_expression_on_his_face/test.jpg'
save_path = './model/demo/classify/expression_model.pkl'

# 1. 数据加载
data_transform = DataTransform().data_transform
dataset = datasets.ImageFolder(root=os.path.join(image_path), transform=data_transform['train'])
train_iter, test_iter = DataTransform.dataset_to_train_test_iter(dataset, 0.8, batch_size)

# 2. 加载MobileNet_v3模型
model = torchvision.models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # 冻结模型参数
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 7)  # 修改最后一层的全连接层
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 优化器

# 3. 模型训练
net_trainer = NetTrainer(train_iter, test_iter, model, criterion, optimizer, epochs=epochs, eval_type="acc",
                         batch_size=batch_size, eval_interval=1, eval_during_training=True)
net_trainer.view_parameters(view_params_details=False)
net_trainer.train_net(net_save_path=save_path)

# best_acc = 0  # 最优精确率
# best_model = None  # 最优模型参数
#
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0  # 损失
#     epoch_acc = 0  # 每个epoch的准确率
#     epoch_acc_count = 0  # 每个epoch训练的样本数
#     train_count = 0  # 用于计算总的样本数，方便求准确率
#     train_bar = tqdm(train_loader)
#     for data in train_bar:
#         images, labels = data
#         # print(images.shape)
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         output = model(images.to(device))
#         loss = criterion(output, labels.to(device))
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
#                                                                  epochs,
#                                                                  loss)
#         # 计算每个epoch正确的个数
#         epoch_acc_count += (output.argmax(axis=1) == labels.view(-1)).sum()
#         train_count += len(images)
#
#     # 每个epoch对应的准确率
#     epoch_acc = epoch_acc_count / train_count
#
#     # 打印信息
#     print("【EPOCH: 】%s" % str(epoch + 1))
#     print("训练损失为%s" % str(running_loss))
#     print("训练精度为%s" % (str(epoch_acc.item() * 100)[:5]) + '%')
#
#     if epoch_acc > best_acc:
#         best_acc = epoch_acc
#         best_model = model.state_dict()
#
#     # 在训练结束保存最优的模型参数
#     if epoch == epochs - 1:
#         # 保存模型
#         torch.save(best_model, save_path)
#
# print('Finished Training')

# 加载索引与标签映射字典

img = Image.open(test_img_path).convert('RGB')  # MobileNet_v3(以及许多其他预训练模型)通常期望输入的是3个通道的RGB图像
img = data_transform['val/test'](img)
img = torch.unsqueeze(img, dim=0).to(net_trainer.device)
cloth_list = dataset.class_to_idx
class_dict = {}
for key, val in cloth_list.items():
    class_dict[val] = key
pred = class_dict[model(img).argmax(axis=1).item()]
print('【预测结果分类】：%s' % pred)
