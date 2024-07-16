from gensim.models import Word2Vec

# 示例语料
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["I", "enjoy", "machine", "learning"],
    ["I", "love", "programming"],
    ["natural", "language", "processing", "is", "fun"],
    ["machine", "learning", "is", "fun"],
    ["programming", "is", "fun"],
    ["machine", "learning", "is", "amazing"],
    ["deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ["artificial", "intelligence", "is", "the", "future"]
]

# 使用 Skip-gram 模型训练词向量
skipgram_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, seed=42)
# 使用 CBOW 模型训练词向量
cbow_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0, seed=42)
SG_word_vectors = skipgram_model.wv  # 获取 Skip-gram 模型的词向量
CBOW_word_vectors = cbow_model.wv  # 获取 CBOW 模型的词向量
print(SG_word_vectors['natural'])  # 查看'natural'的词向量
print(SG_word_vectors.similarity('love', 'enjoy'))  # 计算'love'和'enjoy'的相似度
print(CBOW_word_vectors['natural'])
print(CBOW_word_vectors.similarity('love', 'enjoy'))

from nltk.corpus import wordnet as wn

poses = {'n': 'noun', 'v': 'verb', 's': 'adj (s)', 'a': 'adj', 'r': 'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()], ", ".join([l.name() for l in synset.lemmas()])))

panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
print(list(panda.closure(hyper)))

