# https://www.cnblogs.com/lugendary/p/16192669.html
# https://blog.csdn.net/qq_42365109/article/details/115140450

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from gensim.models import KeyedVectors

from easier_excel.read_data import read_df, set_pd_option
from easier_nn.train_net import NetTrainer

TRAIN_PATH = "input/movie_sentiment/train.tsv"
GLOVE_PATH = "../../model/official/glove/glove.6B.50d.txt"

set_pd_option()
df_train = read_df(TRAIN_PATH)
print(df_train.head())
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
# 查看y的取值的分布：(array([0, 1, 2, 3, 4], dtype=int64), array([ 7072, 27273, 79582, 32927,  9206], dtype=int64))
# 如果模型全预测为2，那么准确率为79582/156060=50.994%，因此模型的准确率应该要高于50.994%
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

# # 将X: str转化为X: tuple，其中元组的每个元素是一个词，并且转小写
# # X = [list(x.lower().split()) for x in X]
# X = [x.lower().split() for x in X]
# print(X[:5])
# # 将文本转换为词向量
# key_vector = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)
# ans = {}
# for word in words:
#     ans[word] = self.key_vector[word]
#     words.remove(word)
# """
#   if self.key_vector is not None:
#         for word in words:
#             if word in self.key_vector:
#                 ans[word] = self.key_vector[word]
#             else:
#                 ans[word] = self.word_not_found
# """
# # embedding = LoadEmbedding()
# # embedding.glove_load_embedding(path="../../model/official/glove/glove.6B.50d.txt")
# # X1_embedding = embedding.glove_search_vector(X[0], path="../../model/official/glove/glove.6B.50d.txt")
# for k, v in X1_embedding.items():
#     # 如果v是np.array类型
#     if isinstance(v, np.ndarray):
#         print(k, v.shape)
#     else:
#         print(k, v)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=10, eval_type="acc", batch_size=16, print_interval=1,
                             eval_during_training=False  # 该参数避免显存不足
                             )
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

# 模型1：该模型无法收敛，loss一直是nan
# Train_BaseModel(X_mean, y)

# 模型2：该模型测试集的acc可以增加(10个epoch从0.6增加到了0.75)，但是测试集的acc始终在58%左右，说明泛化能力不强
# Train_BaseModel(X_flatten, y)

# 模型3：似乎epoch=10太小了，此时的acc是0.626。可以尝试增加epoch，但是一个epoch要训练半分钟。。
# Train_CNNModel(X_padded, y)

Train_TransformerModel(X_padded, y)

# # 训练模型
# for epoch in range(10):
#     for batch_X, batch_y in dataloader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
