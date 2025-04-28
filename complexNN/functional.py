import numpy as np
import math
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


def complexRelu(inp, **factory_kwargs):
    return torch.complex(relu(inp.real, **factory_kwargs), relu(inp.imag, **factory_kwargs))


def complexLeakyRelu(inp, **factory_kwargs):
    return torch.complex(leaky_relu(inp.real, **factory_kwargs), leaky_relu(inp.imag, **factory_kwargs))


def complexSoftmax(inp, **factory_kwargs):
    return torch.complex(softmax(inp.real, **factory_kwargs), softmax(inp.imag, **factory_kwargs))


def complexElu(inp, **factory_kwargs):
    return torch.complex(elu(inp.real, **factory_kwargs), elu(inp.imag, **factory_kwargs))


def complexGelu(inp, **factory_kwargs):
    return torch.complex(gelu(inp.real, **factory_kwargs), gelu(inp.imag, **factory_kwargs))


def complexTanh(inp, **factory_kwargs):
    return torch.complex(tanh(inp.real, **factory_kwargs), tanh(inp.imag, **factory_kwargs))


def complexSigmoid(inp, **factory_kwargs):
    return torch.complex(sigmoid(inp.real, **factory_kwargs), sigmoid(inp.imag, **factory_kwargs))


# Get the parameters of the model inherited from nn.Module.
def get_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

# Efficient implementation equivalent to the following:
def complex_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """
    reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = complexSoftmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
