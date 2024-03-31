import re
from collections import Counter
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd
from matplotlib.image import imread
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import gensim
from gensim import corpora, models, similarities

from easier_nlp.Colorful_Console import ColoredText as CT
from easier_tools.easy_count import listfreq_to_df

# nltk.download()


class enWord:
    def __init__(self, word):
        self.word = word
        self.word_wn = None

    def get_meaning(self, meaning_choice=0):
        """
        获取词意
        :param meaning_choice: 选择第几个义项
        """
        self.word_wn = wn.synsets(self.word)
        print("各个词义：", self.word_wn)  # 查看单词的各个词义
        if isinstance(meaning_choice, int) and meaning_choice >= 0:
            print("释义：", self.word_wn[meaning_choice].definition())  # 查看第meaning_choice种词义的解释

    def create(self, meaning_choice=0):
        """
        以词生句
        :param meaning_choice: 选择第几个义项
        """
        if self.word_wn is None:
            self.word_wn = wn.synsets(self.word)
        # 基于第一种词义进行造句
        word_meaning = wn.synsets(self.word)[meaning_choice]  # wn.synsets("dog")[0]相当于wn.synset('dog.n.01')
        print("造句：", word_meaning.examples())
        # 查看这个词的上位词
        print("上位词：", word_meaning.hypernyms())

class enText:
    def __init__(self, text: str):
        """
        得到：
            self.raw_text: str: 原始文本(不做任何改动)\n
            self.raw_tokens: list: 原始文本的小写后的分词\n
            self.raw_t: Text: 原始文本小写后的Text对象\n
            self.text: str: 原始文本(目前是和raw_text一样的)\n
            self.sentence: list: 分句\n
            self.tokens: list: 处理后的文本的分词，比如['seek', 'answer']\n
            self.pos_tag: tuple of list: 处理后的文本的词性标注，比如[('seek', 'VBD'), ('answer', 'NN')]\n
            self.t: Text: 处理后的文本的的Text对象\n
            self.tokens_freq: Counter: 处理后的文本的词频统计，比如Counter({'instead': 2, 'seeks': 1})\n
            self.tc_df: DataFrame: 处理后的文本的词频统计的DataFrame，比如：\n
                         word  count  id      freq
                6     instead      2   1  0.090909
                8     letting      2   2  0.090909
        :param text: 输入的文本，类型为str
        """
        self.raw_text = text
        self.raw_tokens = word_tokenize(self.raw_text)
        self.raw_tokens = [word.lower() for word in self.raw_tokens]
        self.raw_t = Text(self.raw_tokens)

        self.text = text
        self.sentence = None
        self.tokens = None
        self.pos_tag = None
        self.t = None

        self.tokens_freq = None
        self.tc_df = None

    def process(self, drop_stopword=True, shallow_process=True, deep_process=False, print_details=True,
                print_stopword_intersection=False):
        """
        预处理
        [被删除的操作：]
            self.text = re.sub('[\u4e00-\u9fa5]', '', self.text)  # 去中文
            print(stopwords.raw('english').replace("\n", " "))  # 查看停用词
        :param drop_stopword: 是否去除停用词
        :param shallow_process: 是否浅层地预处理(时态变化)
        :param deep_process:  是否深层地预处理
        :param print_details: 是否输出详情(分句和tokens)
        :param print_stopword_intersection: 是否输出输入文本与停用词的交集
        :return:
        """
        self.sentence = nltk.sent_tokenize(self.text)  # 分句子
        word_token = word_tokenize(self.text)  # 分词

        if shallow_process:
            self.tokens = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in word_token]  # 去除所有的非字母字符并转小写
        if print_stopword_intersection:
            # 是否输出 输入数据 与 英文停用词 的交集
            stopword_set = set([word for word in self.tokens])
            print("输入数据与停用词的交集", stopword_set.intersection(set(stopwords.words('english'))))
        if drop_stopword:
            # 是否删除输入数据中的停用词
            self.tokens = [word for word in self.tokens if (word not in stopwords.words('english'))]
        if deep_process:
            # 语态归一化
            lemmatizer = WordNetLemmatizer()
            self.tokens = [lemmatizer.lemmatize(word) for word in self.tokens]
            # 不建议使用下面两个方法，因为会对人名等错误更改：
            # stemmer = PorterStemmer()
            # self.tokens = [stemmer.stem(word) for word in self.tokens]
            # stemmer1 = SnowballStemmer('english')
            # self.tokens = [stemmer1.stem(word) for word in self.tokens]
        self.tokens = [word for word in self.tokens if word]  # 去除空字符串

        if print_details:
            print(CT("分句：").blue(), self.sentence)
            print(CT("tokens：").blue(), self.tokens)

    def query(self, query_text=None, show_postag=False, show_chunk=False):
        if self.tokens is None:
            self.process()
        self.t = Text(self.tokens)  # 生成Text对象

        self.tc_df = listfreq_to_df(self.tokens, list_name="word", show_details=False)  # 词频统计

        # 词性标注
        self.pos_tag = pos_tag(self.tokens)

        if query_text is not None:
            if isinstance(query_text, list):
                # 查询某些字符串
                for query_text in query_text:
                    print(CT(query_text).yellow(), "的次数：", self.t.count(query_text))
                    if self.t.count(query_text) > 0:
                        print(CT(query_text).yellow(), "在原字符串中的位置：", self.raw_t.index(query_text))  # 位置从0开始
                        print(CT(query_text).yellow(), "在预处理后的字符串中的位置：", self.t.index(query_text))  # 位置从0开始
                    else:
                        print(CT(query_text).yellow(), "不在输入的文本中")
            elif isinstance(query_text, str):
                # 查询某个字符串
                print(CT(query_text).yellow(), "的次数：", self.t.count(query_text))
                if self.t.count(query_text) > 0:
                    print(CT(query_text).yellow(), "在原字符串中的位置：", self.raw_t.index(query_text))  # 位置从0开始
                    print(CT(query_text).yellow(), "在预处理后的字符串中的位置：", self.t.index(query_text))  # 位置从0开始
                else:
                    print(CT(query_text).yellow(), "不在输入的文本中")
        if show_postag:
            # 词性标注
            print(self.pos_tag)
        if show_chunk:
            # 命名实体识别
            print(ne_chunk(self.pos_tag))

    def plot(self, topN=10, RegexpParser_rule=None):
        if self.tokens is None:
            self.process()
        if self.t is None:
            self.query()

        if isinstance(topN, int) and topN > 0:
            self.t.plot(topN)
        if RegexpParser_rule is not None:
            cp = RegexpParser(RegexpParser_rule)
            result = cp.parse(self.pos_tag)
            result.draw()


class zhText:
    def __init__(self, text):
        self.text = text
        self.sentences = None
        self.tokens = None
        self.tokens_doc = None
        self.token_list = None

    def process(self, shallow_process=True, print_details=True, drop_stopword=True, split_sentences='punctuation',
                path_jieba=r"xm_nlp/input/jieba/genshin_dict.txt", path_stopword=r"xm_nlp/input/stopwords.txt",
                if_example=False):
        if split_sentences == 'punctuation':
            self.sentences = re.split(r'[。！？]', self.text)
            self.sentences = [sentence.strip() for sentence in self.sentences if sentence.strip()]  # 用if来保证sentence不会是空字符串
            print(CT("正在使用符号作为划分句子的依据").green())
        elif split_sentences == 'newline':
            self.sentences = self.text.split('\n')
            print(CT(r"正在使用\n作为划分句子的依据").green())
        else:
            self.sentences = re.split(r'[。！？]', self.text)
            self.sentences = [sentence.strip() for sentence in self.sentences if sentence.strip()]
            print(CT("Waring！你应该正确地指定使用哪一种分句方法，目前支持punctuation和newline。因为你的输入错误，这里默认使用punctuation。").yellow())
        self.text = re.sub(r'[^\u4e00-\u9fa5]+', '', self.text)

        jieba.load_userdict(path_jieba)  # 本地自定义文档
        # if if_example:
        #     for word in ["颇有", "这样的"]:  # 因为不知道这是一个词，所以要添加。
        #         jieba.add_word(word)
        #     jieba.suggest_freq(('神', '挺'), True)  # 因为一直将'神挺'视为一个词，所以要删除。也可以jieba.del_word('神挺')
        self.tokens = jieba.cut(self.text, cut_all=False)  # 精确匹配
        token_sent = [list(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]+', '', sent))) for sent in self.sentences]
        self.tokens_doc = [" ".join(sent) for sent in token_sent]
        self.tokens = '/'.join(self.tokens)
        if drop_stopword:
            with open(path_stopword, encoding='utf-8') as f:
                zh_stopwords = [line.strip() for line in f]
                self.tokens = '/'.join(word for word in self.tokens.split('/') if word not in zh_stopwords)
                self.tokens_doc = [' '.join(word for word in sent.split(' ') if word not in zh_stopwords)
                                   for sent in self.tokens_doc if sent.split()]
        if shallow_process:
            pass
        if print_details:
            print(CT("全tokens：token总数:" + str(len(self.tokens.split('/'))) + "，不重复的token总数:" +
                     str(len(list(set([word for word in self.tokens.split('/')]))))).pink(), self.tokens)
            print(CT("按句子的tokens：句子总数:" + str(len(self.tokens_doc))).pink(), self.tokens_doc)
            print(CT("句子：总数：:" + str(len(self.sentences))).pink(), self.sentences)

    def query(self, tags_topK=5,  tfidf_topN=5, show_postag=False, use_tfidf=True, show_tfidf_details=False, plot_tfidf=False,
              use_lda=True):
        tags = jieba.analyse.extract_tags(self.text, topK=tags_topK, withWeight=True)
        print(CT("使用jieba.analyse.extract_tags提取文档关键词：").blue())
        for word, weight in tags:
            print(word, weight)
        if show_postag:
            words = pseg.cut(self.text)
            print(CT("使用jieba分析词性：").blue())
            for word, flags in words:
                print(word, flags)
        if use_tfidf:
            tfidf_v = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",  # 将默认的匹配2个及以上字符改为1个及以上
                                      max_df=0.6,  # 去除出现在60%以上句子的词语
                                      ngram_range=(1, 2),  # 允许词表使用1个词语，或者2个词语的组合
                                      )
            tfidf_matrix = tfidf_v.fit_transform(self.tokens_doc)
            tf_feature = tfidf_v.get_feature_names_out()  # 获取词汇表
            tfidf_dict = dict(zip(tf_feature, tfidf_v.idf_))
            tfidf_df = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf_dict), orient='index')
            tfidf_df.columns = ['tfidf']
            tfidf_ascending = tfidf_df.sort_values(by=['tfidf'], ascending=True)
            tfidf_descending = tfidf_df.sort_values(by=['tfidf'], ascending=False)
            print(CT(f"tfidf最低的{tfidf_topN}个词：").blue())
            print(tfidf_ascending.head(tfidf_topN))
            print(CT(f"tfidf最高的{tfidf_topN}个词：").blue())
            print(tfidf_descending.head(tfidf_topN))
            if show_tfidf_details:
                # TF-IDF的shape是(文档个数, 词汇表数)。比如将句子试作一个文档，文档个数就是句子总数。
                print(CT("TF-IDF 矩阵：大小:" + str(tfidf_matrix.shape)).blue())
                print(tfidf_matrix.toarray())
                print(CT("词汇表：大小:" + str(tf_feature.size)).blue())
                print(tf_feature)
            if plot_tfidf:
                # 用svd和tsne降维，从(n,D)降维到(n,2)以便可视化
                svd = TruncatedSVD(n_components=min(30, int(tfidf_matrix.shape[1]/100)), random_state=42)
                svd_tfidf = svd.fit_transform(tfidf_matrix)
                tsne = TSNE(n_components=2, verbose=0, random_state=42)  # verbose=1或2时，会输出一些进度信息
                tsne_tfidf = tsne.fit_transform(svd_tfidf)
                print(tsne_tfidf)
                print(tsne_tfidf.shape)
                tsne_tfidf = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])
                tsne_tfidf['text'] = self.tokens_doc
                p = figure(title="SVD&TSNE降维可视化散点图", x_axis_label='X', y_axis_label='Y')
                p.scatter(x='x', y='y', source=tsne_tfidf, alpha=0.5)  # 绘制散点图
                hover = HoverTool(tooltips=[("文本", "@text{safe}")])  # 定义悬停工具，显示对应的标签
                p.add_tools(hover)
                show(p)  # 显示图形

        if use_lda:
            print(self.tokens_doc)
            list_of_list_tokens = [doc.split(' ') for doc in self.tokens_doc]
            print(list_of_list_tokens)
            dictionary = corpora.Dictionary(list_of_list_tokens)
            corpus = [dictionary.doc2bow(sentence) for sentence in list_of_list_tokens]
            lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)  # 类似kmeans的k值
            for topic in lda.print_topics(num_topics=6, num_words=5):
                print(topic[1])  # 0是主题的序号，1是主题的单词概率分布

    def draw_cloud(self):
        data = {}  # 字典
        for word in self.tokens.split('/'):
            if not data.__contains__(word):
                data[word] = 0
            data[word] += 1
        word_cloud = WordCloud(
            background_color='white',
            max_words=100,
            font_path=r"xm_nlp/input/simsun.ttc",
            mask=imread(r"xm_nlp/input/data/Keqing.jpg"),
            width=2503,
            height=3755,
            # stopwords=zh_stopwords
        ).generate_from_frequencies(data)
        word_cloud.to_file('output/cloud.png')
        plt.imshow(word_cloud)
        plt.axis('off')
        plt.show()
        plt.close()






