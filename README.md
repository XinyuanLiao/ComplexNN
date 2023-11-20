<p align="center">
  <img src="https://github.com/XinyuanLiao/ComplexNN/blob/main/.github/Images/logo.jpg" width="200px"/>
</p>
<div align="center">
<h1>ComplexNN: Complex Neural Network Modules</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![Static Badge](https://img.shields.io/badge/Language-Python_|_PyTorch-green)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
[![Actions Status](https://github.com/XinyuanLiao/ComplexNN/workflows/CodeQL/badge.svg)](https://github.com/XinyuanLiao/ComplexNN/actions)
[![PyPI version](https://img.shields.io/pypi/v/complexNN?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/complexNN/)
[![Downloads](https://static.pepy.tech/personalized-badge/complexNN?&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads)](https://pepy.tech/project/complexNN)
</div>

# What is ComplexNN?

ComplexNN provides the plural form of standard modules under the PyTorch framework **without any extra trainable parameters**. The parameters and calling methods of the modules in this framework are consistent with those of the PyTorch framework, **incurring no additional learning cost**. This repository is completed due to PyTorch's support for complex gradients. Please refer to the [documentation](https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc) for details.

# Why ComplexNN?

Currently, state-of-the-art complex neural network libraries, such as [deep_complex_networks](https://github.com/ChihebTrabelsi/deep_complex_networks) [1], [complexPytorch](https://github.com/wavefrontshaping/complexPyTorch) [2], etc., implement the complex-valued network module by utilizing two sets of parameters to represent the real and imaginary parts of the complex numbers. This implementation method not only increases the number of parameters but is also not conducive to the backpropagation of gradients and significantly increases the difficulty of training. Therefore, I used PyTorch's support for complex gradient operations to re-implement the complex-valued network module.

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
```v0.0.1``` Provided the plural form of the base standard PyTorch network module.

```v0.1.1``` Added support for the [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) (LRU) [3].

```v0.1.2``` Bug fixed, and added new support.

```v0.2.1``` Bug fixed, and added new support.

```v0.3.1``` Optimized code structure, bug fixed, and added new support.

```v0.3.2``` Bug fixed.

```v0.4.2``` Optimized code structure.

# Modules

## complexNN.nn
* _cRule, cElu, cLeakyRelu, cSoftmax, cGelu, cTanh, cSigmoid_
* _cBatchNorm1d/ 2d/ 3d, cLayerNorm, cDropout, cDropout2d, cMaxPool1d/ 2d, cAvgPool1d/ 2d_
* _cLinear, cMLP, cConv1d, cConv2d, cRNNCell, LRUCell, cGRUCell, cLSTMCell, cRNN, cGRU, cLSTM_
* _EarlyStopping_

## complexNN.functional
* _Corresponding function implementation in complexNN.nn_

 
Other modules will be considered for updates in the future.

# Examples 
<a target="_blank" href="https://drive.google.com/file/d/1O8SzfJANAmcIjjN6b2E5pIkkjutgx5ov/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
## Convolutional neural network
```python
import torch
from complexNN.nn import cConv1d, cConv2d


if __name__ == '__main__':
    batch_size, in_channels, out_channels, seq_len = 10, 3, 16, 10
    conv_tensor = torch.rand((batch_size, in_channels, seq_len))
    conv1d = cConv1d(in_channels, out_channels, padding='same')
    print(conv1d(conv_tensor).shape)

    H, W = 256, 256
    conv2d_tensor = torch.rand((batch_size, in_channels, H, W))
    conv2d = cConv2d(in_channels, out_channels, padding=1)
    print(conv2d(conv2d_tensor).shape)
```
## Multilayer perceptron
```python
import torch
from complexNN.nn import cMLP


if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 10, 10, 20, 15
    input_tensor = torch.rand((batch_size, input_size), dtype=torch.cfloat)
    mlp = cMLP(input_size, hidden_size, output_size, num_layers=3)
    out = mlp(input_tensor)
    print(out.shape)
```

## Recurrent neural networks
```python
import torch
from complexNN.nn import cRNN, cGRU, cLSTM


if __name__ == '__main__':
    batch_size, input_size, hidden_size, seq_len, num_layers = 10, 10, 20, 15, 3
    input_tensor = torch.rand((seq_len, batch_size, input_size), dtype=torch.cfloat)
    h0, c0 = torch.zeros((num_layers, batch_size, hidden_size)), torch.zeros((num_layers, batch_size, hidden_size))

    rnn = cRNN(input_size, hidden_size, num_layers)
    gru = cGRU(input_size, hidden_size, num_layers)
    lstm = cLSTM(input_size, hidden_size, num_layers)

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
[1] _Chiheb Trabelsi, et al. "Deep Complex Networks." arXiv preprint arXiv:1705.09792 (2017)._

[2] _Matthès, Maxime W., et al. "Learning and avoiding disorder in multimode fibers." Physical Review X 11.2 (2021): 021060._

[3] _Orvieto, Antonio, et al. "Resurrecting recurrent neural networks for long sequences." arXiv preprint arXiv:2303.06349 (2023)._
