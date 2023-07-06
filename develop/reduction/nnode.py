import torch


def node(x,y):
    if y == 'sin':
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
    elif y == 'exp1':
        return x ** 1
    elif y == 'exp2':
        return x ** 2
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
        return torch.nn.functional.softplus(x)
    elif y == 'softsign':
        return torch.nn.functional.softsign(x)
    elif y == 'elu':
        return torch.nn.functional.elu(x)
    elif y == 'selu':
        return torch.nn.functional.selu(x)
    elif y == 'celu':
        return torch.nn.functional.celu(x)
    elif y == 'gelu':
        return torch.nn.functional.gelu(x)
    elif y == 'hardshrink':
        return torch.nn.functional.hardshrink(x)
    elif y == 'hardtanh':
        return torch.nn.functional.hardtanh(x)
    elif y == 'leakyrelu':
        return torch.nn.functional.leaky_relu(x)
    elif y == 'logsigmoid':
        return torch.nn.functional.logsigmoid(x)
    elif y == 'prelu':
        return torch.nn.functional.prelu(x,x)
    elif y == 'rrelu':
        return torch.nn.functional.rrelu(x)
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