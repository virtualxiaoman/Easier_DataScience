import matplotlib.pyplot as plt

from easier_nn.classic_dataset import load_time_machine
import easier_nlp.preprocessing as xmpp
from easier_excel.draw_data import plot_xy
from easier_tools.easy_count import listfreq_to_df

content = load_time_machine()
text = xmpp.enText(content)
text.process(print_details=False, drop_stopword=False)
text.query()
plot_xy(x=text.tc_df['id'], y=text.tc_df['count'], x_label='id', y_label='count', title='111',
        font_name='Times New Roman', xscale_log=True, yscale_log=True)
df2 = listfreq_to_df(raw_list=[pair for pair in zip(text.tokens[:-1], text.tokens[1:])], list_name="word_bigram")
df3 = listfreq_to_df(raw_list=[pair for pair in zip(text.tokens[:-2], text.tokens[1:-1], text.tokens[2:])], list_name="word_trigram")
fig, ax = plt.subplots()
ax = plot_xy(x=text.tc_df['id'], y=text.tc_df['count'], x_label='id', y_label='count', title='freq', label="1", color='blue',
             font_name='Times New Roman', xscale_log=True, yscale_log=True, show_plt=False, use_ax=True, ax=ax)
ax = plot_xy(x=df2['id'], y=df2['count'], x_label='id', y_label='count', title='freq', label="2", color='green',
             font_name='Times New Roman', xscale_log=True, yscale_log=True, show_plt=False, use_ax=True, ax=ax)
ax = plot_xy(x=df3['id'], y=df3['count'], x_label='id', y_label='count', title='freq', label="3", color='red',
             font_name='Times New Roman', xscale_log=True, yscale_log=True, show_plt=False, use_ax=True, ax=ax)
plt.show()
