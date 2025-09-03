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

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameterw

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
