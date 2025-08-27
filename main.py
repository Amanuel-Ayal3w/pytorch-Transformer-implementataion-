import torch 
import torch.nn as nn
import math

class inputEmbedding(nn.Module):
    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

def forward(self,x):
     return self.embedding(x) * math.sqrt(self.d_model)


class positionalEmbedding(nn.module):
     def __init__(self, d_model: int, seq_len:int, dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

       
       # create a vector of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
       # create matrix of shape seq_len,d_model)
      
        postion = torch.arrange(0,seq_len,d_model)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(1000.0)/d_model))
