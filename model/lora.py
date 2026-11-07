import math 
import torch 

# alpha/r * BAx 
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))#fill A with random numbers in place
        self.scaling = alpha/rank
    def forward(self,x):
        x = self.scaling * (x @ self.A @ self.B)
        return x

# Wx + alpha/r * BAx 
class LinearwithLoRA(torch.nn.Module):
    def __init__(self,linear,rank,alpha):
        super().__init__()
        self.linear = linear 
        self.lora = LoRALayer(linear.in_features, linear.out_features,rank,alpha)
    def forward(self,x):
        x = self.linear(x) + self.lora(x)
        return x 

# go through the whole model and swap every nn.Linear layer for LinearWithLora wrapper
def replace_linear_with_lora(model,rank,alpha):
    for name, module in model.named_children():
        if isinstance(module,torch.nn.Linear):
            setattr(model, name, LinearwithLoRA(module,rank,alpha))
        else:
            replace_linear_with_lora(module,rank,alpha)
    