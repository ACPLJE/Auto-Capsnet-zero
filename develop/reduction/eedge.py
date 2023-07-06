import torch


def edge(x,y,z):
    if z == 'add':
        return torch.add(x,y)
    elif z == 'sub':
        return torch.sub(x,y)
    elif z == 'mul':
        return torch.mul(x,y)
    elif z == 'div':
        return torch.div(x,y)