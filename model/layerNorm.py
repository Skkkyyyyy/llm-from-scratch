import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self,X):
        mean = X.mean(dim=-1, keepdim=True) #dim=-1, apply to rows #keepdim=True, keep the dim you reduced and shrink to size 1
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (X - mean) / (var + self.eps)**0.5
        return self.scale * norm_x + self.shift 
    