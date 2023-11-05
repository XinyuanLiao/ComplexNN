import torch
from torch import relu, tanh, sigmoid, nn
from torch.nn.functional import gelu


class cRelu(nn.Module):
    @staticmethod
    def forward(inp):
        return complexRelu(inp)


class cGelu(nn.Module):
    @staticmethod
    def forward(inp):
        return complexGelu(inp)


class cTanh(nn.Module):
    @staticmethod
    def forward(inp):
        return complexTanh(inp)


class cSigmoid(nn.Module):
    @staticmethod
    def forward(inp):
        return complexSigmoid(inp)


def complexRelu(input):
    return torch.complex(relu(input.real), relu(input.imag))


def complexGelu(input):
    return torch.complex(gelu(input.real), gelu(input.imag))


def complexTanh(input):
    return torch.complex(tanh(input.real), tanh(input.imag))


def complexSigmoid(input):
    return torch.complex(sigmoid(input.real), sigmoid(input.imag))
