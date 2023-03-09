import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class LSTM_Standard(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):

        super(LSTM_Standard, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,dropout = 0.5)
        self.fc = nn.Linear(hidden_dim , 3)
        self.dropout = nn.Dropout(0.8)

    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output= self.rnn(embedding)[0]  #[seq_len, batch, hidden_dim]
        #output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim]

        #query = self.dropout(output)
        #attn_output, attention = self.attention_net(output, query)     
        logit = self.fc(output[-1])
        return logit