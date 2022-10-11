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
    elif y == 'exp1':
        return x ** 1
    elif y == 'exp2':
        return x ** 2
    elif y == 'sqrt':
        return x ** 0.5
    elif y == 'square':
        return x ** 2
    elif y == 'cube':
        return x ** 3
    elif y == 'sin':
        return torch.sin(x)
    elif y == 'cos':
        return torch.cos(x)
    elif y == 'tan':
        return torch.tan(x)
    elif y == 'asin':
        return torch.asin(x)
    elif y == 'acos':
        return torch.acos(x)
    elif y == 'atan':
        return torch.atan(x)
    elif y == 'sinh':
        return torch.sinh(x)
    elif y == 'cosh':
        return torch.cosh(x)
    elif y == 'tanh':
        return torch.tanh(x)
    elif y == 'asinh':
        return torch.asinh(x)
    elif y == 'acosh':
        return torch.acosh(x)
    elif y == 'atanh':
        return torch.atanh(x)
    elif y == 'log':
        return torch.log(x)
    elif y == 'log2':
        return torch.log2(x)
    elif y == 'log10':
        return torch.log10(x)
    elif y == 'exp':
        return torch.exp(x)
    elif y == 'expm1':
        return torch.expm1(x)
    elif y == 'relu':
        return torch.relu(x)
    elif y == 'sigmoid':
        return torch.sigmoid(x)
    elif y == 'tanh':
        return torch.tanh(x)
    elif y == 'softplus':
        return torch.softplus(x)
    elif y == 'softsign':
        return torch.softsign(x)
    elif y == 'elu':
        return torch.elu(x)
    elif y == 'selu':
        return torch.selu(x)
    elif y == 'celu':
        return torch.celu(x)
    elif y == 'gelu':
        return torch.gelu(x)
    elif y == 'hardshrink':
        return torch.hardshrink(x)
    elif y == 'hardtanh':
        return torch.hardtanh(x)
    elif y == 'leakyrelu':
        return torch.leakyrelu(x)
    elif y == 'logsigmoid':
        return torch.logsigmoid(x)
    elif y == 'prelu':
        return torch.prelu(x)
    elif y == 'rrelu':
        return torch.rrelu(x
