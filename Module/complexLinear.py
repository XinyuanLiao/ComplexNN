import math
import torch
import torch.nn as nn


class complexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(complexLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not input.dtype == torch.cfloat:
            input = torch.complex(input, torch.zeros_like(input))
        return torch.matmul(input, self.weight.T) + self.bias
