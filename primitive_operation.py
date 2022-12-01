import torch


def operation(x,y):
    if y == 'add1':
        return x + 1
    elif y == 'add2':
        return x + 2
    elif y == 'sub1':
        return x - 1
    elif y == 'sub2':
        return x - 2
    elif y == 'mul1':
        return x * 1
    elif y == 'mul2':
        return x * 2
    elif y == 'div1':
        return x / 1
    elif y == 'div2':
        return x / 2
    elif y == 'sqrt':
        return x ** 0.5
    elif y == 'square':
        return x ** 2
    elif y == 'cube':
        return x ** 3
    elif y == 'log':
        return torch.log(x)
    elif y == 'log2':
        return torch.log2(x)
    elif y == 'log10':
        return torch.log10(x)
