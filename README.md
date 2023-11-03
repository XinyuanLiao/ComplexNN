<p align="center">
  <img src="https://github.com/XinyuanLiao/ComplexNN/blob/main/.github/Images/logo.jpg" width="200px"/>
</p>
<div align="center">
<h1>ComplexNN: Complex Neural Network Modules</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![Static Badge](https://img.shields.io/badge/Language-Python_|_PyTorch-green)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
[![Actions Status](https://github.com/XinyuanLiao/ComplexNN/workflows/CodeQL/badge.svg)](https://github.com/XinyuanLiao/ComplexNN/actions)
[![Scc Count Badge](https://sloc.xyz/github/XinyuanLiao/ComplexNN/?category=code)](https://github.com/XinyuanLiao/ComplexNN/)
[![PyPI version](https://img.shields.io/pypi/v/complexNN?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/complexNN/)
[![Downloads](https://static.pepy.tech/personalized-badge/complexNN?&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads)](https://pepy.tech/project/complexNN)
    
</div>

Since the latest versions of PyTorch support matrix operations and gradient descent on plural parameters, this repository provides the latest plural form of some standard PyTorch network modules. Compared with utilizing two sets of parameters to represent the real and imaginary parts of the network plural parameters respectively, directly utilizing complex numbers as network parameters will _**halve the number of trainable parameters**_, which speeds up the training process.

# Install
To install _complexNN_ for the first time:
```
pip install complexNN
```
To upgrade a previous installation of _complexNN_ to the most recent version:
```
pip install --upgrade complexNN
```

# Versions
```v0.0.1``` Provides the plural form of the base standard PyTorch network module.

```v0.1.1``` Adds support for the [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) (LRU).

```v0.1.2``` Bug fixed. Adds support for BatchNorm2d, and BatchNorm3d.

# Modules
The complex form modules include
<div align="center">
  
| **[complexLayer](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexLayer.py)** | **[complexRNNcell](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexRNNcell.py)** | **[complexActivation](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexActivation.py)** | **[complexFunction](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexFunction.py)** |
|:-----------------:|:------------------:|:---------------------:|:-------------------:|
| Linear            | RNN Cell           | Relu                  | BatchNorm           |
|                   | GRU Cell           | Gelu                  | LayerNorm           |
|                   | LSTM Cell          | Tanh                  | Dropout             |
|                   | LRU Cell [1]       | Sigmoid               |                     |

</div>
 
Note that the native version of ```torch.nn.Dropout``` is supported:exclamation::exclamation: Other modules will be considered for updates in the future.

# Examples
## Multilayer perceptron
```python
import torch.nn as nn
from complexNN.complexActivation import complexTanh
from complexNN.complexLinear import complexLinear


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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = complexTanh(self.input_layer(x))
        x = self.dropout(x)
        for i in range(self.layer_num - 1):
            x = complexTanh(self.hidden_layers[i](x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output
```

## Recurrent neural network
```python
import torch.nn as nn
from complexNN.complexRNNcell import complexRNNCell


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
      title={ComplexNN: Complex Neural Network Modules},
      author={Xinyuan Liao},
      Url= {https://github.com/XinyuanLiao/ComplexNN}, 
      year={2023}
}
```

# Reference
[1] _Orvieto, Antonio, et al. "Resurrecting recurrent neural networks for long sequences." arXiv preprint arXiv:2303.06349 (2023)._
