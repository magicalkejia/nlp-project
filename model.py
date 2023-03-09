import torch.nn as nn
import torch
import math
import torch.nn.functional as F


#LSTM + self-attention

class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,n_layers,num_class):
        super(LSTM_Attention, self).__init__()
        self.W_Q = nn.Linear(hidden_dim,hidden_dim,bias =True)
        self.W_K = nn.Linear(hidden_dim,hidden_dim,bias =True)
        self.W_V = nn.Linear(hidden_dim,hidden_dim,bias =True)
        
        #embedding
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        #LSTM
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,dropout = 0.5)
        #Linear
        self.fc = nn.Linear(hidden_dim,num_class)
        #dropout
        self.dropout = nn.Dropout(0.8)

    def attention(self,Q,K,V):
        
        d_k = K.size(-1)
        scores = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)
        
        
        output = context.sum(1)
         
        return output,alpha_n
    
    def forward(self, data):
        embedding = self.dropout(self.embedding(data))
        embedding = embedding.transpose(0,1)   
     #   print(embedding.shape)
        output,(h_n,c) = self.rnn(embedding) #output.shape
        Q = self.W_Q(output)
        K = self.W_K(output)
        V = self.W_V(output)
        attention_out,alpha_n= self.attention(Q,K,V)#attention_out.shape
        result  = self.fc(attention_out)

        return result