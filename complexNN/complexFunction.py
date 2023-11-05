import torch
import torch.nn as nn
from torch.nn.functional import dropout, dropout2d, avg_pool2d, max_pool2d, avg_pool1d, max_pool1d


def _retrieve_elements_from_indices(tensor, indices):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complex_avg_pool2d(inp, *args, **kwargs):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
    """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )

def complex_avg_pool1d(inp, *args, **kwargs):
    absolute_value_real = avg_pool1d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool1d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )


def complex_max_pool2d(
        inp,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
    """
    absolute_value, indices = max_pool2d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(inp.imag, inp.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
            torch.cos(angle).type(torch.complex64)
            + 1j * torch.sin(angle).type(torch.complex64)
    )


def complex_max_pool1d(
        inp,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
    """
    absolute_value, indices = max_pool1d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(inp.imag, inp.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
            torch.cos(angle).type(torch.complex64)
            + 1j * torch.sin(angle).type(torch.complex64)
    )


def complex_dropout(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_dropout2d(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm1d, self).__init__()
        self.real_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)

        return torch.complex(real_output, imag_output)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)

        return torch.complex(real_output, imag_output)


class ComplexBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm3d, self).__init__()
        self.real_bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)

        return torch.complex(real_output, imag_output)


class ComplexLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=False):
        super(ComplexLayerNorm, self).__init__()
        self.real_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        self.imag_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_norm(real_input)
        imag_output = self.imag_norm(imag_input)

        return torch.complex(real_output, imag_output)


class ComplexDropout(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout(inp, self.p)
        else:
            return inp


class ComplexDropout2d(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, self.p)
        else:
            return inp


class ComplexMaxPool2d(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexAvgPool2d(torch.nn.Module):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexMaxPool1d(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool1d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)
