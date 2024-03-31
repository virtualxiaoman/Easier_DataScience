import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = """As the Yuheng of the Liyue Qixing, she is someone who seeks her own answers instead of idly letting chaos run amok in Liyue. 
She chooses her own path with her own power and ability, instead of letting the gods determine her fate."""
tokens = nlp(text)
print(tokens)
sentences = [sent for sent in tokens.sents]
print(sentences)
# 词性
for token in tokens:
    print("{}:{}".format(token, token.pos_))
print('-----')
# 命名实体
for ent in tokens.ents:
    print(f"{ent}:{ent.label_}")
# displacy.render(tokens, style='ent')  # 是一段HTML，在jupyter里运行
