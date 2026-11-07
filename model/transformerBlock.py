import torch.nn as nn
import torch 
from model.feedforward import FeedForward 
from model.attention import MultiheadAttention
from model.layerNorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #create an instance: runs __init__ (builds weights once)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.ff = FeedForward(cfg)
        self.attn = MultiheadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self,X):
        #call the instance, -> runs forward 
        shortcut = X 
        X = self.norm1(X)
        X = self.attn(X)
        X = self.dropout(X)
        X = shortcut + X 

        shortcut = X 
        X = self.norm2(X)
        X = self.ff(X)
        X = self.dropout(X)
        X = shortcut + X 
        return X 




