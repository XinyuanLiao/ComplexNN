import torch
import torch.nn as nn


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.real_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.imag_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)

        return torch.complex(real_output, imag_output)


class ComplexLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ComplexLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.real_norm = nn.LayerNorm(num_features, eps=eps)
        self.imag_norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, input):
        real_input = torch.real(input)
        imag_input = torch.imag(input)

        real_output = self.real_norm(real_input)
        imag_output = self.imag_norm(imag_input)

        return torch.stack([real_output, imag_output], dim=2)
