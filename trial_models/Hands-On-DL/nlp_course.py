# https://www.cnblogs.com/lugendary/p/16192669.html
# https://blog.csdn.net/qq_42365109/article/details/115140450

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from easier_excel.read_data import read_df, set_pd_option
from easier_nn.train_net import NetTrainer

TRAIN_PATH = "input/movie_sentiment/train.tsv"
GLOVE_PATH = "../../model/official/glove/glove.6B.50d.txt"

set_pd_option()
df_train = read_df(TRAIN_PATH)
print(df_train.head())
print(df_train.shape)  # (156060, 4)

# 去除"Phrase"里小于等于3个字符的短语
df_train = df_train[df_train["Phrase"].apply(lambda x: len(x.split()) > 3)]
print(df_train.shape)  # (92549, 4)

# df的列名和含义如下：
# PhraseId（短语ID）：每个短语的唯一标识符。
# SentenceId（句子ID）：短语所在句子的标识符，一个句子可能包含多个短语。
# Phrase（短语）：需要进行情感分析的文本短语。
# Sentiment（情感）：情感标签，表示该短语的情感倾向。从0到4的整数，表示从非常消极（0）到非常积极（4）的情感范围。

# 提取出短语和情感标签
X = df_train["Phrase"].values
y = df_train["Sentiment"].values
print(X[:5])
print(y[:5])

# 查看y的取值的分布：
# 对于整个数据集，情感标签的分布如下：
# (array([0, 1, 2, 3, 4], dtype=int64), array([ 7072, 27273, 79582, 32927,  9206], dtype=int64))
# 如果模型全预测为2，那么准确率为79582/156060=50.994%，因此模型的准确率应该要高于50.994%
# 对于去除了较短短语的数据集，情感标签的分布如下：
# (array([0, 1, 2, 3, 4], dtype=int64), array([ 5979, 19785, 36795, 22651,  7339], dtype=int64))
# 如果模型全预测为2，那么准确率为36795/92549=39.76%，因此模型的准确率应该要高于39.76%
print(np.unique(y, return_counts=True))


# 加载GloVe词向量
class LoadGlove:
    def __init__(self, glove_path):
        self.glove_dict = {}
        self.glove_path = glove_path

# 加载GloVe词向量
    def load_glove_vectors(self):
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.glove_dict[word] = vector

# 将句子转换为二维矩阵
    def sentence_to_matrix(self, sentence, embedding_dim=50):
        words = sentence.split()
        matrix = np.zeros((len(words), embedding_dim))
        for i, word in enumerate(words):
            if word in self.glove_dict:
                matrix[i] = self.glove_dict[word]
            else:
                # 对于未找到的单词，使用1e-3的常量，这样可以避免全零的情况
                matrix[i] = 1e-3 * np.ones(embedding_dim)
        return matrix.astype(np.float32)  # 避免nn的时候出现float与int混用的问题


# 将句子变为embedding矩阵
load_glove = LoadGlove(GLOVE_PATH)
load_glove.load_glove_vectors()
# 转小写是因为GloVe词向量是小写的
X_embedding = [load_glove.sentence_to_matrix(sentence.lower()) for sentence in X]

for i in range(8):
    print(f"Sentence {i}: {np.array(X_embedding[i]).shape}")

print(f"Total sentences: {len(X_embedding)}")

# 基本模型--mean：将句子中所有词的词向量取平均作为句子的表示
X_mean = np.array([sentence.mean(axis=0) for sentence in X_embedding])
# 基本模型--padding：将所有句子的embedding矩阵展平成一个向量
max_length = max(len(sentence) for sentence in X_embedding)
X_padded = np.zeros((len(X_embedding), max_length, 50)).astype(np.float32)
for i, sentence in enumerate(X_embedding):
    X_padded[i, :len(sentence)] = sentence
X_flatten = X_padded.reshape(X_padded.shape[0], -1)


class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(cv, dim=2)[0] for cv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, hidden = self.rnn(x)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


class TransformerModel(nn.Module):
    # embedding_dim要和X_embedding的最后一个维度一致，也就是glove的维度
    def __init__(self, embedding_dim, num_heads, num_layers, output_dim, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        encoded = self.transformer_encoder(x)
        encoded = self.dropout(encoded.mean(dim=0))  # (batch_size, embedding_dim)
        return self.fc(encoded)


def Train_BaseModel(X, y):
    net = BaseModel(X.shape[1], 128, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=10, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_CNNModel(X, y):
    net = CNNModel(50, 100, [2, 3, 4], 5, 0.5)
    # class_weights = 1 / torch.tensor([7072, 27273, 79582, 32927, 9206], dtype=torch.float)
    # weights = class_weights / class_weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights.to('cuda'))  # 类别不平衡
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 0.0005训练的好慢。
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=50, eval_type="acc", batch_size=32, print_interval=5,
                             eval_during_training=False  # 该参数避免显存不足
                             )
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_RNNModel(X, y):
    net = RNNModel(50, 128, 5, 2, False, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=10, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_TransformerModel(X, y):
    net = TransformerModel(50, 2, 2, 5, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=10, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")

# 模型1：该模型无法收敛，loss一直是nan，说明mean丢失的信息太多了
# Train_BaseModel(X_mean, y)

# 模型2：该模型测试集的acc可以增加(10个epoch从0.6增加到了0.75)，但是测试集的acc始终在58%左右，说明泛化能力不强
# Train_BaseModel(X_flatten, y)

# 模型3：似乎epoch=10太小了，此时的acc是0.626。因为loss一直在稳步下降，所以可以尝试增加epoch来提高acc，但是一个epoch要训练半分钟。。
# 设置epoch=50的时候acc是0.645。
Train_CNNModel(X_padded, y)

# 模型4：显存不够我跑不了。。
# Train_RNNModel(X_padded, y)

# 模型5：显存不够我跑不了。。
# Train_TransformerModel(X_padded, y)


