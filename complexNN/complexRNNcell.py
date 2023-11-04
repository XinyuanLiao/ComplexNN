import numpy as np
import torch
import torch.nn as nn
from complexActivation import complexSigmoid, complexTanh
from complexLayer import complexLinear


class complexRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(complexRNNCell, self).__init__()
        self.Wx = complexLinear(input_size, hidden_size, bias)
        self.Wh = complexLinear(hidden_size, hidden_size, bias)

    def forward(self, x, h_prev):
        h = complexTanh(self.Wx(x) + self.Wh(h_prev))
        return h


class LRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=False,
                 sigma_min=0.9, sigma_max=0.999, phase=np.pi * 2):
        super(LRUCell, self).__init__()
        u1 = np.random.random(size=int(hidden_size))
        u2 = np.random.random(size=int(hidden_size))

        # Prior information
        v = -0.5 * np.log(u1 * (sigma_max ** 2 - sigma_min ** 2) + sigma_min ** 2)
        theta = u2 * phase

        # Unconstrained optimization
        self.v_log = nn.Parameter(torch.tensor(np.log(v), dtype=torch.float32))
        self.theta_log = nn.Parameter(torch.tensor(np.log(theta), dtype=torch.float32))

        # Input matrix
        self.B = complexLinear(input_size, hidden_size, bias=bias)
        # Output matrix
        self.C = complexLinear(hidden_size, output_size, bias=bias)

    def effective_W(self):
        w = torch.exp(-torch.exp(self.v_log) + 1j * torch.exp(self.theta_log))
        return torch.diag(w)  # State matrix

    def forward(self, x, h_prev):
        if not h_prev.dtype == torch.cfloat:
            h_prev = torch.complex(h_prev, torch.zeros_like(h_prev))
        h = self.B(x) + torch.matmul(h_prev, self.effective_W())
        return torch.real(self.C(h))


class complexGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(complexGRUCell, self).__init__()
        self.W_r = complexLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_z = complexLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_h = complexLinear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, h_prev):
        r = complexSigmoid(self.W_r(torch.cat((x, h_prev), 1)))
        z = complexSigmoid(self.W_z(torch.cat((x, h_prev), 1)))
        h_hat = complexTanh(self.W_h(torch.cat((x, r * h_prev), 1)))
        h_new = (1 - z) * h_hat + z * h_prev
        return h_new


class complexLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(complexLSTMCell, self).__init__()
        self.W_i = complexLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_f = complexLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_c = complexLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_o = complexLinear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, hidden):
        h, c = hidden
        i = complexSigmoid(self.W_i(torch.cat((x, h), 1)))
        f = complexSigmoid(self.W_f(torch.cat((x, h), 1)))
        g = complexTanh(self.W_c(torch.cat((x, h), 1)))
        o = complexSigmoid(self.W_o(torch.cat((x, h), 1)))

        c = f * c + i * g
        h = o * complexTanh(c)

        return h, c


if __name__ == '__main__':
    batch_size, input_size, hidden_size = 10, 10, 20
    input_tensor = torch.rand((batch_size, input_size))
    init_state = torch.zeros((batch_size, hidden_size))

    rnn = complexRNNCell(input_size, hidden_size)
    gru = complexGRUCell(input_size, hidden_size)
    lstm = complexLSTMCell(input_size, hidden_size)
    lru = LRUCell(input_size, hidden_size, input_size)

    rnn_out = rnn(input_tensor, init_state)
    gru_out = gru(input_tensor, init_state)
    lstm_out, _ = lstm(input_tensor, (init_state, init_state))
    lru_out = lru(input_tensor, init_state)

    print(rnn_out.shape, gru_out.shape, lstm_out.shape, _.shape, lru_out.shape)
