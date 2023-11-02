![](https://img.shields.io/hexpm/l/plug)
![Static Badge](https://img.shields.io/badge/Language-Python_|_PyTorch-green)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
![GitHub all releases](https://img.shields.io/github/downloads/XinyuanLiao/ComplexNN/total)
# ComplexNN
Since subsequent versions of PyTorch support matrix operations and gradient descent on complex parameters, this repository provides the latest complex form of some standard Pytorch network modules. Compared with utilizing two sets of parameters to represent the real and imaginary parts of the network parameters respectively, directly utilizing complex numbers as network parameters will halve the number of trainable parameters. This results in faster training speed.

# Module
The complex form modules include
* complexLinear
    - Linear
* complexRNNcell
    - RNN Cell
    - GRU Cell
    - LSTM Cell
* complexActivation
    - Relu
    - Gelu
    - Tanh
    - Sigmoid
* complexFunction
    - BatchNorm1d
    - LayerNorm1d
 
Other modules will be considered for updates in the future.

# Examples
## Multilayer perceptron
```python
import torch.nn as nn
from complexActivation import complexTanh
from complexLinear import complexLinear


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

    def forward(self, x):
        x = complexTanh(self.input_layer(x))
        for i in range(self.layer_num - 1):
            x = complexTanh(self.hidden_layers[i](x))
        output = self.output_layer(x)
        return output
```

## Recurrent neural network
```python
import torch.nn as nn
from compledRNNcell import complexRNNCell


class complexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(complexRNN, self).__init__()
        self.num_layers = num_layers
        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.rnn_layers.append(complexRNNCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, x, h_0):
        h_prev = h_0
        for i in range(self.laryer_num):
            h_prev = self.rnn_layers[i](x, h_prev)
        return h_prev
```

# Cite as
```
@misc{ComplexNN,
      title={Complex Neural Network Modules},
      author={Xinyuan Liao},
      Url= {https://github.com/XinyuanLiao/ComplexNN}, 
      year={2023}
}
```
