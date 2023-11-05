import torch
import torch.nn as nn
from complexNN.complexFunction import ComplexBatchNorm1d
from complexNN.complexActivation import cGelu


class complexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(complexLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)

        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.xavier_uniform_(self.bias)

    def forward(self, input):
        if not input.dtype == torch.cfloat:
            input = torch.complex(input, torch.zeros_like(input))
        return torch.matmul(input, self.weight.T) + self.bias


class complexMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(complexMLP, self).__init__()
        self.num_layers = num_layers
        self.input_layer = complexLinear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([complexLinear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = complexLinear(hidden_size, output_size)
        self.bn = ComplexBatchNorm1d(hidden_size)
        self.activation = cGelu()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for i in range(self.num_layers - 1):
            x = self.activation(self.bn(self.hidden_layers[i](x)))
        output = self.output_layer(x)
        return output


class complexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1, groups=1):
        super(complexConv1d, self).__init__()
        assert in_channels % groups == 0, "In_channels should be an integer multiple of groups."
        assert out_channels % groups == 0, "Out_channels should be an integer multiple of groups."

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels // groups, kernel_size), dtype=torch.cfloat))

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        out = torch.nn.functional.conv1d(input=inp, weight=self.weight, bias=self.bias, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out


class complexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1, groups=1):
        super(complexConv2d, self).__init__()
        assert in_channels % groups == 0, "In_channels should be an integer multiple of groups."
        assert out_channels % groups == 0, "Out_channels should be an integer multiple of groups."

        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn((out_channels, in_channels // groups, *kernel_size), dtype=torch.cfloat))

        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        out = torch.nn.functional.conv2d(input=inp, weight=self.weight, bias=self.bias, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out


if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 10, 10, 20, 15
    input_tensor = torch.rand((batch_size, input_size), dtype=torch.cfloat)
    mlp = complexMLP(input_size, hidden_size, output_size, num_layers=3)
    out = mlp(input_tensor)
    print(out.shape)

    in_channels, out_channels, seq_len = 3, 16, 10
    conv_tensor = torch.rand((batch_size, in_channels, seq_len))
    conv1d = complexConv1d(in_channels, out_channels, padding='same')
    print(conv1d(conv_tensor).shape)

    H, W = 256, 256
    conv2d_tensor = torch.rand((batch_size, in_channels, H, W))
    conv2d = complexConv2d(in_channels, out_channels, padding=1)
    print(conv2d(conv2d_tensor).shape)
