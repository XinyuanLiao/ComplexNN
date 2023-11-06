import numpy as np
import torch
from torch.nn.functional import dropout, dropout2d, avg_pool2d, max_pool2d, avg_pool1d, max_pool1d, gelu, leaky_relu, \
    elu, softmax
from torch.nn import CosineSimilarity
from torch import relu, tanh, sigmoid


def _retrieve_elements_from_indices(tensor, indices):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complexAvgPool2d(inp, *args, **kwargs):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )


def complexAvgPool1d(inp, *args, **kwargs):
    absolute_value_real = avg_pool1d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool1d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )


def complexMaxPool2d(inp, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    absolute_value, indices = max_pool2d(inp.abs(), kernel_size=kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, ceil_mode=ceil_mode, return_indices=True)
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
            torch.cos(angle).type(torch.complex64)
            + 1j * torch.sin(angle).type(torch.complex64)
    )


def complexMaxPool1d(inp, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    absolute_value, indices = max_pool1d(inp.abs(), kernel_size=kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, ceil_mode=ceil_mode, return_indices=True)
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
            torch.cos(angle).type(torch.complex64)
            + 1j * torch.sin(angle).type(torch.complex64)
    )


def complexDropout(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complexDropout2d(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complexRelu(inp):
    return torch.complex(relu(inp.real), relu(inp.imag))


def complexLeakyRelu(inp):
    return torch.complex(leaky_relu(inp.real), leaky_relu(inp.imag))


def complexSoftmax(inp):
    return torch.complex(softmax(inp.real), softmax(inp.imag))


def complexElu(inp):
    return torch.complex(elu(inp.real), elu(inp.imag))


def complexGelu(inp):
    return torch.complex(gelu(inp.real), gelu(inp.imag))


def complexTanh(inp):
    return torch.complex(tanh(inp.real), tanh(inp.imag))


def complexSigmoid(inp):
    return torch.complex(sigmoid(inp.real), sigmoid(inp.imag))


# Get the parameters of the model inherited from nn.Module.
def get_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params
