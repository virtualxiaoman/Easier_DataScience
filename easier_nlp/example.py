import easier_nlp
import easier_nlp.preprocessing as xmpp

text = """As the Yuheng of the Liyue Qixing, she is someone who seeks her own answers instead of idly letting chaos run amok in Liyue.
She chooses her own path with her own power and ability, instead of letting the gods determine her fate."""
text = xmpp.enText(text)
text.process()
text.query(query_text=["liyue", "run"])
text.plot()
text.plot(RegexpParser_rule="NP:{<NN>+}")

word = "person"
word = xmpp.enWord(word)
word.get_meaning()
word.create()

# text = """璃月七星之一，玉衡星。对「帝君一言而决的璃月」颇有微词——但实际上，神挺欣赏她这样的人。"""
with open("xm_nlp/input/data/Processed Yet the Butterfly Flutters Away.txt", encoding='utf-8') as f:
    text = f.read()

text = xmpp.zhText(text)
text.process(if_example=False, print_details=False)
text.query(tags_topK=5)
text.draw_cloud()
