import torch
from torch import relu, tanh, sigmoid
from torch.nn.functional import gelu


def complexRelu(input):
    return torch.complex(relu(input.real), relu(input.imag))


def complexGelu(input):
    return torch.complex(gelu(input.real), gelu(input.imag))


def complexTanh(input):
    return torch.complex(tanh(input.real), tanh(input.imag))


def complexSigmoid(input):
    return torch.complex(sigmoid(input.real), sigmoid(input.imag))
