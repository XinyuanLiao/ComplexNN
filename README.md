<p align="center">
  <img src="https://github.com/XinyuanLiao/ComplexNN/blob/main/.github/Images/logo.jpg" width="200px"/>
</p>
<div align="center">
<h1>ComplexNN: Complex Neural Network Modules</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![Static Badge](https://img.shields.io/badge/Language-Python_|_PyTorch-green)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
[![Actions Status](https://github.com/XinyuanLiao/ComplexNN/workflows/CodeQL/badge.svg)](https://github.com/XinyuanLiao/ComplexNN/actions)
[![Scc Count Badge](https://sloc.xyz/github/XinyuanLiao/ComplexNN/)](https://github.com/XinyuanLiao/ComplexNN/)
[![PyPI version](https://img.shields.io/pypi/v/complexNN?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/complexNN/)
[![Downloads](https://static.pepy.tech/personalized-badge/complexNN?&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads)](https://pepy.tech/project/complexNN)

This repository provides the plural form of standard modules under the PyTorch framework **without any extra trainable parameters**. The parameters and calling methods of the modules in this framework are consistent with those of the PyTorch framework, **incurring no additional learning cost**. This repository is completed due to PyTorch's support for complex gradients. Please refer to the [documentation](https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc) for details.
    
</div>
<p align="center">
  <img src="https://github.com/XinyuanLiao/ComplexNN/blob/main/.github/Images/autograd.png" width="1200px"/>
</p>
<p align="center">
  <img src="https://github.com/XinyuanLiao/ComplexNN/blob/main/.github/Images/derivatives.png" width="1200px"/>
</p>

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

```v0.1.2``` Bug fixes, and new support.

```v0.2.1``` Bug fixes, and new support.

```v0.3.1``` Code structure optimization, bug fixes, and new support.

# Modules
The plural form modules include
<div align="center">
  
| **[complexLayer](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexLayer.py)** | **[complexRNNcell](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexRNNcell.py)** | **[complexActivation](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexActivation.py)** | **[complexFunction](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexFunction.py)** | **[complexRNN](https://github.com/XinyuanLiao/ComplexNN/blob/main/complexNN/complexRNN.py)**|
|:-----------------:|:------------------:|:---------------------:|:-------------------:|:-------------------:|
| Linear            | RNN Cell           | Relu                  | BatchNorm 1d/ 2d/ 3d   |RNN|
|  MLP              | GRU Cell           | Gelu                  | LayerNorm           |GRU|
|  Conv 1d/ 2d      | LSTM Cell          | Tanh                  | dropout 1d/ 2d        |LSTM|
|                   | LRU Cell [1]       | Sigmoid               | avg/ max pool         ||

</div>
 
Other modules will be considered for updates in the future.

# Examples 
<a target="_blank" href="https://github.com/XinyuanLiao/ComplexNN/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
## Convolutional neural network
```python
import torch
from complexNN.complexLayer import complexConv1d, complexConv2d


if __name__ == '__main__':
    batch_size, in_channels, out_channels, seq_len = 10, 3, 16, 10
    conv_tensor = torch.rand((batch_size, in_channels, seq_len))
    conv1d = complexConv1d(in_channels, out_channels, padding='same')
    print(conv1d(conv_tensor).shape)

    H, W = 256, 256
    conv2d_tensor = torch.rand((batch_size, in_channels, H, W))
    conv2d = complexConv2d(in_channels, out_channels, padding=1)
    print(conv2d(conv2d_tensor).shape)
```
## Multilayer perceptron
```python
import torch
form complexNN.complexLayer import complexMLP


if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 10, 10, 20, 15
    input_tensor = torch.rand((batch_size, input_size), dtype=torch.cfloat)
    mlp = complexMLP(input_size, hidden_size, output_size, num_layers=3)
    out = mlp(input_tensor)
    print(out.shape)
```

## Recurrent neural networks
```python
import torch
from complexNN.complexRNN import complexRNN, complexGRU, complexLSTM


if __name__ == '__main__':
    batch_size, input_size, hidden_size, seq_len, num_layers = 10, 10, 20, 15, 3
    input_tensor = torch.rand((seq_len, batch_size, input_size), dtype=torch.cfloat)
    h0, c0 = torch.zeros((num_layers, batch_size, hidden_size)), torch.zeros((num_layers, batch_size, hidden_size))

    rnn = complexRNN(input_size, hidden_size, num_layers)
    gru = complexGRU(input_size, hidden_size, num_layers)
    lstm = complexLSTM(input_size, hidden_size, num_layers)

    rnn_out, _ = rnn(input_tensor, h0)
    gru_out, _ = gru(input_tensor, h0)
    lstm_out, _ = lstm(input_tensor, (h0, c0))

    print(rnn_out.shape, gru_out.shape, lstm_out.shape)
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
