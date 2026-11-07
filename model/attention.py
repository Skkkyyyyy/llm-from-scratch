import torch 
import torch.nn as nn 

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads 

        self.W_q = nn.Linear(d_in,d_out,qkv_bias)
        self.W_k = nn.Linear(d_in,d_out, qkv_bias)
        self.W_v = nn.Linear(d_in,d_out, qkv_bias)
        self.W_o = nn.Linear(d_out, d_in)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1).bool()
        )
        
    def forward(self, X):
        b, N, _ = X.shape
        Q = self.W_q(X).view(b,N,self.num_heads,self.head_dim).transpose(1,2)
        V = self.W_v(X).view(b,N,self.num_heads,self.head_dim).transpose(1,2)
        K = self.W_k(X).view(b,N,self.num_heads,self.head_dim).transpose(1,2)
        
        E = Q @ K.transpose(2,3)/self.head_dim**0.5
        E = E.masked_fill(self.mask[:N,:N],float('-inf'))
        A = torch.softmax(E, dim=-1)
        A = self.dropout(A)

        Y = (A @ V).transpose(1,2).contiguous().view(b,N,self.d_out)
        O = self.W_o(Y)

        return O 
    




    
