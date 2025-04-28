<div align="center">
<h1>ComplexNN: Complex Neural Network Modules</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![languages](https://img.shields.io/github/languages/top/XinyuanLiao/ComplexNN)
![Size](https://img.shields.io/github/languages/code-size/XinyuanLiao/ComplexNN)
![Static Badge](https://img.shields.io/badge/Framework-PyTorch-green)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
[![Actions Status](https://github.com/XinyuanLiao/ComplexNN/workflows/CodeQL/badge.svg)](https://github.com/XinyuanLiao/ComplexNN/actions)
[![PyPI version](https://img.shields.io/pypi/v/complexNN?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/complexNN/)
[![Downloads](https://static.pepy.tech/personalized-badge/complexNN?&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads)](https://pepy.tech/project/complexNN)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/XinyuanLiao/ComplexNN)
</div>

`ComplexNN` provides the plural form of standard modules under the PyTorch framework **without any extra trainable parameters**. The parameters and calling methods of the modules in this framework are consistent with those of the PyTorch framework, **incurring no additional learning cost**. This repository is completed due to PyTorch's support for complex gradients. Please refer to the [documentation](https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc) for details.

---

## Why ComplexNN?

Currently, state-of-the-art complex neural network libraries, such as [deep_complex_networks](https://github.com/ChihebTrabelsi/deep_complex_networks) [1], [complexPytorch](https://github.com/wavefrontshaping/complexPyTorch) [2], etc., implement the complex-valued network module by utilizing two sets of parameters to represent the real and imaginary parts of the complex numbers. This implementation method not only increases the number of parameters but is also not conducive to the backpropagation of gradients and significantly increases the difficulty of training. Therefore, I used PyTorch's support for complex gradient operations to re-implement the complex-valued network module.

---

## Installation
To install _complexNN_ for the first time:
```bash
pip install complexNN
```
To upgrade a previous installation of _complexNN_ to the most recent version:
```bash
pip install --upgrade complexNN
```
---

## Module Overview

### Activation Functions

| Class         | Description                                 | Example Usage       |
|---------------|---------------------------------------------|---------------------|
| `cRule`       | Rule-based activation                       | `cRule()`           |
| `cElu`        | Exponential Linear Unit                     | `cElu()`            |
| `cLeakyRelu`  | Leaky Rectified Linear Unit                 | `cLeakyRelu()`      |
| `cSoftmax`    | Softmax over specified dimension            | `cSoftmax()`        |
| `cGelu`       | Gaussian Error Linear Unit                  | `cGelu()`           |
| `cTanh`       | Hyperbolic tangent                          | `cTanh()`           |
| `cSigmoid`    | Sigmoid function                            | `cSigmoid()`        |

### Normalization & Regularization

| Class            | Type           | Description                                    |
|------------------|----------------|------------------------------------------------|
| `cBatchNorm1d`   | Normalization  | Batch normalization over 1D inputs             |
| `cBatchNorm2d`   | Normalization  | Batch normalization over 2D inputs             |
| `cBatchNorm3d`   | Normalization  | Batch normalization over 3D inputs             |
| `cLayerNorm`     | Normalization  | Layer normalization                            |
| `cDropout`       | Regularization | Dropout for 1D inputs                          |
| `cDropout2d`     | Regularization | Dropout for 2D inputs                          |
| `cMaxPool1d`     | Pooling        | 1D max pooling                                 |
| `cMaxPool2d`     | Pooling        | 2D max pooling                                 |
| `cAvgPool1d`     | Pooling        | 1D average pooling                             |
| `cAvgPool2d`     | Pooling        | 2D average pooling                             |

### Layers

| Class                | Description                                          |
|----------------------|------------------------------------------------------|
| `cLinear`            | Fully connected linear layer                        |
| `cMLP`               | Multi-layer perceptron wrapper                       |
| `cConv1d`            | 1D convolution layer                                 |
| `cConv2d`            | 2D convolution layer                                 |
| `cRNNCell`           | Single-step vanilla RNN cell                         |
| `cGRUCell`           | Single-step GRU cell                                 |
| `cLSTMCell`          | Single-step LSTM cell                                |
| `cRNN`               | Multi-step vanilla RNN                               |
| `cGRU`               | Multi-step GRU                                      |
| `cLSTM`              | Multi-step LSTM                                     |
| `cMultiHeadAttention`| Scaled dot-product multi-head attention mechanism    |

---
## Quick Start

### Multi-head attention
```python
import torch
from complexNN.nn import cMultiHeadAttention


if __name__ == '__main__':
    batch_size, embed_size, seq_len = 10, 512, 15
    input_tensor = torch.rand((seq_len, batch_size, embed_size), dtype=torch.cfloat)
    mha = cMultiHeadAttention(E_q=embed_size, E_k=embed_size, E_v=embed_size, E_total=embed_size, nheads=8)
    mha_out = mha(input_tensor, input_tensor, input_tensor)
    print(mha_out.shape, mha_out.dtype)
```
### Convolutional neural network
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
### Multilayer perceptron
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

### Recurrent neural networks
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
---

## Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests for improvements.

---

## Reference
[1] _Chiheb Trabelsi, et al. "Deep Complex Networks." arXiv preprint arXiv:1705.09792 (2017)._

[2] _Matthès, Maxime W., et al. "Learning and avoiding disorder in multimode fibers." Physical Review X 11.2 (2021): 021060._

