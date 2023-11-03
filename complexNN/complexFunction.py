import torch
import torch.nn as nn


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
