import math
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import d2l

from easier_nn.classic_dataset import load_time_machine
from easier_nn.load_data import trainset_to_dataloader
from easier_nn.train_net import train_net
import easier_nlp.preprocessing as xmpp
from easier_excel.draw_data import plot_xy
from easier_tools.easy_count import list_to_freqdf
from easier_nlp.preprocessing import seq_data_iter_sequential, load_data_time_machine

content = load_time_machine()
text = xmpp.enText(content)
text.process(print_details=False, drop_stopword=False)
text.query()
print((len(text.tokens)))
print(type(text.tokens))
print(text.tc_df.shape)
print(text.tokens[:10])
print(text.tc_df.head())
word_to_id = {word: word_id for word, word_id in zip(text.tc_df['word'], text.tc_df['id'])}
tokens_ids = [word_to_id[token] for token in text.tokens]
train_iter = seq_data_iter_sequential(corpus=tokens_ids, batch_size=32, num_steps=10)

input_size = len(text.tc_df)  # 4651
hidden_size = 128
output_size = len(text.tc_df)  # 4651
net = nn.Sequential(nn.RNN(input_size, hidden_size, num_layers=1),
                    nn.Linear(hidden_size, output_size))
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
train_net(tokens_ids, tokens_ids, data_iter=train_iter, net=net, loss=loss, optimizer=optimizer,
          num_epochs=5, batch_size=32, show_interval=1)


exit(1)

plot_xy(x=text.tc_df['id'], y=text.tc_df['count'], x_label='id', y_label='count', title='111',
        font_name='Times New Roman', x_scale='log', y_scale='log')
df2 = list_to_freqdf(raw_list=[pair for pair in zip(text.tokens[:-1], text.tokens[1:])], list_name="word_bigram")
df3 = list_to_freqdf(raw_list=[pair for pair in zip(text.tokens[:-2], text.tokens[1:-1], text.tokens[2:])], list_name="word_trigram")
fig, ax = plt.subplots()
ax = plot_xy(x=text.tc_df['id'], y=text.tc_df['count'], x_label='id', y_label='count', title='freq', label="1", color='blue',
             font_name='Times New Roman', x_scale='log', y_scale='log', show_plt=False, use_ax=True, ax=ax)
ax = plot_xy(x=df2['id'], y=df2['count'], x_label='id', y_label='count', title='freq', label="2", color='green',
             font_name='Times New Roman', x_scale='log', y_scale='log', show_plt=False, use_ax=True, ax=ax)
ax = plot_xy(x=df3['id'], y=df3['count'], x_label='id', y_label='count', title='freq', label="3", color='red',
             font_name='Times New Roman', x_scale='log', y_scale='log', show_plt=False, use_ax=True, ax=ax)
plt.show()
#
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
device = d2l.try_gpu()
net = d2l.RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

