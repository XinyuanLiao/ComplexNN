import torch
import torch.nn as nn
from complexFunction import ComplexBatchNorm1d
from complexNN.complexActivation import complexGelu


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

    def forward(self, x):
        x = complexGelu(self.input_layer(x))
        for i in range(self.num_layers - 1):
            x = complexGelu(self.bn(self.hidden_layers[i](x)))
        output = self.output_layer(x)
        return output


if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 10, 10, 20, 15
    input_tensor = torch.rand((batch_size, input_size), dtype=torch.cfloat)
    mlp = complexMLP(input_size, hidden_size, output_size, num_layers=3)
    out = mlp(input_tensor)
    print(out.shape)
