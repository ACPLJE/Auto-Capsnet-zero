import torch


def redu(x,y):
    if y == 'norm':
        return torch.norm(x)
    if y == 'sum':
        return torch.sum(x)
    if y == 'mean':
        return torch.mean(x)
    if y == 'max':
        return torch.max(x)
    if y == 'min':
        return torch.min(x)
    if y == 'std':
        return torch.std(x)
    if y == 'var':
        return torch.var(x)
    
   