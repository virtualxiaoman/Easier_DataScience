import re
import json
from annoy import AnnoyIndex
from collections import OrderedDict
import pandas as pd
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText, KeyedVectors
import torch
from torch.utils.data import TensorDataset, DataLoader


# 载入embedding
class LoadEmbedding:
    def __init__(self):
        """
        [注意]：
            路径暂时写死了，需要根据实际情况修改
        [使用示例-腾讯]:
            tencent_embedding = LoadEmbedding()
            tencent_embedding.tc_load_embedding()  # 仅第一次需要运行，目的是生成index和annoy文件
            ans = tencent_embedding.tc_search_ann10("干什么")
            print(ans)  # 例如：['干什么', '想干什么', ... , '什么事啊']，这里省略了一部分
            ans = tencent_embedding.tc_search_vector(["干什么", "干啥"])
            vec1 = ans["干什么"]
            vec2 = ans["干啥"]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(similarity)  # 余弦相似度0.871848
        """
        self.key_vector = None
        self.word_index = None
        self.word_not_found = '没有找到这个词'

    # 载入腾讯AI实验室开源的中文词向量
    def tc_load_embedding(self, path="../model/official/tencent_embedding/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"):
        """
        载入腾讯AI实验室开源的中文词向量，本函数一般只要运行一次，之后就可以直接使用ann了
        :param path: 词向量文件路径
        :return: 词向量字典
        """
        print("可能需要较长的时间...")
        self.key_vector = KeyedVectors.load_word2vec_format(path, binary=False)
        print("词向量矩阵的形状:", self.key_vector.vectors.shape)

        self.word_index = OrderedDict()
        for counter, key in enumerate(self.key_vector.key_to_index.keys()):
            self.word_index[key] = counter
        with open('../model/official/tencent_embedding/tc_word_index.json', 'w') as fp:
            json.dump(self.word_index, fp)

        tc_index = AnnoyIndex(200, 'angular')  # 200维的词向量，angular是余弦相似度
        i = 0
        for key in self.key_vector.key_to_index.keys():
            v = self.key_vector[key]
            tc_index.add_item(i, v)
            i += 1
        tc_index.build(10)
        tc_index.save('../model/official/tencent_embedding/tc_index_build10.index')

    # 查询最相似的词
    def tc_search_ann10(self, words='测试', topN=10):
        """
        查询最相似的词
        :param words: 待查询的词或列表
        :param topN: 返回的最相似的词的数量
        :return: dict, key是词，value是最相似的词列表
        """
        # 如果words是字符串，说明只有一个词，转换为列表
        if isinstance(words, str):
            words = [words]

        # 读取词表
        with open('../model/official/tencent_embedding/tc_word_index.json', 'r') as fp:
            self.word_index = json.load(fp)
        # AnnoyIndex对象，用于查询
        tc_index = AnnoyIndex(200, 'angular')
        tc_index.load('../model/official/tencent_embedding/tc_index_build10.index')
        # 反向id==>word映射词表
        reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

        # get_nns_by_item基于annoy查询词最近的topN个向量，返回结果是个list，里面元素是索引
        ans = {}
        for word in words:
            word_neighbors = []
            if word in self.word_index:
                for item in tc_index.get_nns_by_item(self.word_index[word], topN):
                    word_neighbors.append(reverse_word_index[item])
            else:
                word_neighbors.append(self.word_not_found)
            ans[word] = word_neighbors
        return ans

    # 查询某些词的词向量
    def tc_search_vector(self, words=('测试', '人工智能'),
                         path="../model/official/tencent_embedding/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"):
        """
        查询某些词的词向量
        :param words: 待查询的词或列表
        :param path: 词向量文件路径
        :return: dict, key是词，value是词向量
        """
        # 如果words是字符串，说明只有一个词，转换为列表
        if isinstance(words, str):
            words = [words]

        ans = {}
        if self.key_vector is not None:
            for word in words:
                ans[word] = self.key_vector[word]
                words.remove(word)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0]
                    if word in words:
                        embedding = np.array(parts[1:], dtype=np.float32)
                        ans[word] = embedding
                        words.remove(word)  # 加速查找过程
                    if not words:
                        break
        for word in words:
            ans[word] = self.word_not_found
        return ans

    # 载入glove的词向量
    def glove_load_embedding(self, path="../model/official/glove_embedding/glove.6B.50d.txt"):
        """
        载入glove的词向量
        :param path: 词向量文件路径
        :return: 词向量字典
        """
        self.key_vector = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)
        print("词向量矩阵的形状:", self.key_vector.vectors.shape)

    def glove_search_vector(self, words=('test', 'ai'), path="../model/official/glove_embedding/glove.6B.50d.txt"):
        """
        查询某些词的词向量
        :param words: 待查询的词或列表
        :param path: 词向量文件路径
        :return: dict, key是词，value是词向量
        """
        # 如果words是字符串，说明只有一个词，转换为列表
        if isinstance(words, str):
            words = [words]

        ans = {}
        if self.key_vector is not None:
            for word in words:
                ans[word] = self.key_vector[word]
                words.remove(word)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0]
                    if word in words:
                        embedding = np.array(parts[1:], dtype=np.float32)
                        ans[word] = embedding
                        words.remove(word)
                    if not words:
                        break
        for word in words:
            ans[word] = self.word_not_found

        return ans

# NLP预处理(英文)，暂时没有完善
class NLP_EN:
    def __init__(self, text_list, label_list=None):
        """
        初始化

        Attributes:
            text_list: 文本列表 [["Some text"], ["Another text"]]
            label_list: 标签列表 [0, 1]
            word_list: 词列表 [["Some", "text"], ["Another", "text"]]
            corpus_list: ['Some text', 'Another text']

        :param text_list: 文本列表
        :param label_list: 类别标签、情感标签等
        """
        self.text_list = text_list  # 传入的文本列表
        self.label_list = label_list  # 传入的标签列表
        self.word_list = []  # 词列表
        self.corpus_list = []  #

    def process_text(self):
        for text in self.text_list:
            text = re.sub('[^A-Za-z]+', ' ', text).strip().lower()
            self.word_list.append(text.split())

    # Word2Vec，注意不会检查路径合法性
    def word2vec(self, path="../model/word2vec/word2vec.model"):
        # 使用word2vec训练词向量
        model = Word2Vec(sentences=self.word_list, vector_size=100, window=5, min_count=5, workers=4)
        model.save(path)  # 保存模型
        model = Word2Vec.load(path)  # 加载模型
        print("词向量矩阵的形状:", model.wv.vectors.shape)
        print("nice的相似词:", model.wv.most_similar("nice"))
        print("寻找不同词:", model.wv.doesnt_match("good nice terrible".split()))
        print("获取相似度:", model.wv.similarity("happy", "excited"))
        # # 获取词向量
        # print(model.wv['good'])
        # # 获取所有词
        # print(model.wv.index_to_key)
        # # 获取词向量矩阵
        # print(model.wv.vectors)

    # FastText，注意不会检查路径合法性
    def fasttext(self, path="../model/fasttext/fasttext.model"):
        model_FT = FastText(sentences=self.word_list, vector_size=100, window=5, min_count=5, workers=4)
        model_FT.save(path)
        model_FT = FastText.load(path)
        print("词向量矩阵的形状:", model_FT.wv.vectors.shape)
        print("nice的相似词:", model_FT.wv.most_similar("nice"))
        print("寻找不同词:", model_FT.wv.doesnt_match("good nice terrible".split()))
        print("获取相似度:", model_FT.wv.similarity("happy", "excited"))

    # TF-IDF
    def tfidf(self):
        # 初始化TfidfVectorizer
        vectorizer = TfidfVectorizer()
        # 计算TF-IDF
        tfidf_matrix = vectorizer.fit_transform(self.corpus_list)
        # 将结果转换为稀疏矩阵格式
        tfidf_array = tfidf_matrix.toarray()
        # 打印特征名称（单词）
        feature_names = vectorizer.get_feature_names_out()
        # 打印TF-IDF值
        print("Feature Names:", feature_names)
        print("TF-IDF Array:\n", tfidf_array)


# TextRank算法，暂未测试
class TextRank:
    def __init__(self, documents, language='english'):
        """
        初始化TextRank
            # 示例文档
            documents = [["hello", "hee"], ["hi", "your"]]
            # 实例化TextRank类
            textrank = TextRank(documents)
            # 提取关键词
            keywords = textrank.extract_keywords(top_n=5)
            print(keywords)

        :param documents: list of list, 每个子列表代表一个文档，子列表内的元素是单词
        :param language: 语言选择，默认是'english'
        """
        self.documents = documents
        self.stopwords = set(stopwords.words(language))

    def _build_graph(self, sentences):
        """
        构建共现矩阵并创建图
        :param sentences: list of list, 每个子列表代表一个文档，子列表内的元素是单词
        :return: networkx.Graph
        """
        word_graph = nx.Graph()
        for sentence in sentences:
            for w1, w2 in combinations(set(sentence), 2):
                if w1 != w2:
                    word_graph.add_edge(w1, w2, weight=1.0)
        return word_graph

    def _rank_words(self, word_graph):
        """
        使用PageRank算法对词汇进行排序
        :param word_graph: networkx.Graph
        :return: dict, 词汇及其对应的PageRank得分
        """
        return nx.pagerank(word_graph, weight='weight')

    def extract_keywords(self, top_n=10):
        """
        提取关键词
        :param top_n: 返回的关键词数量
        :return: list, 关键词列表
        """
        # 预处理文本
        processed_docs = []
        for doc in self.documents:
            filtered_words = [word.lower() for word in doc if word.lower() not in self.stopwords and word.isalpha()]
            processed_docs.append(filtered_words)

        # 构建图
        word_graph = self._build_graph(processed_docs)

        # 排序词汇
        word_ranks = self._rank_words(word_graph)

        # 提取前N个关键词
        sorted_words = sorted(word_ranks.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, rank in sorted_words[:top_n]]

        return keywords


# 本文件下面三个函数最好不要再调用了(没什么问题，但是已经被集成到了train_net里面了)，但是由于历史遗留问题，现在暂不删除。
def load_array(data_arrays, batch_size=64, if_shuffle=True):
    """
    [底层函数] 构造一个PyTorch数据迭代器
    [使用示例]
        load_array((features, labels), batch_size)
        load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    :param data_arrays: 一个包含数据数组的元组或列表。通常包括输入特征和对应的标签(features, labels)。
    :param batch_size: 每个小批量样本的数量。
    :param if_shuffle: True数据将被随机洗牌(用于训练);False数据将按顺序提供(用于模型的评估或测试)。
    """
    dataset = TensorDataset(*data_arrays)  # 将数据数组转换为TensorDataset对象(将数据存储为Tensor对象，并允许按索引访问)
    return DataLoader(dataset, batch_size, shuffle=if_shuffle)


def trainset_to_dataloader(X_train, y_train, batch_size=64, y_reshape=False):
    """
    将训练集转为DataLoader
    :return: DataLoader数据类型
    """
    # 要注意X_train的type是DataFrame
    if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
    if isinstance(X_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
        pass
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
    if y_reshape:
        return load_array((X_train, y_train.reshape(-1, 1)), batch_size)
    else:
        return load_array((X_train, y_train), batch_size)

def testset_to_dataloader(X_test, y_test, batch_size=64, y_reshape=False):
    """
    将测试集转为DataLoader
    :return: DataLoader数据类型
    """
    if isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.DataFrame):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    if isinstance(X_test, torch.Tensor) and isinstance(y_test, torch.Tensor):
        pass
    else:
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
    if y_reshape:
        return load_array((X_test, y_test.reshape(-1, 1)), batch_size, if_shuffle=False)
    else:
        return load_array((X_test, y_test), batch_size, if_shuffle=False)




