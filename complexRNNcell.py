import numpy as np
import torch
import torch.nn as nn
from complexActivation import complexSigmoid, complexTanh
from complexLinear import complexLinear


class complexRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(complexRNNCell, self).__init__()
        self.Wx = complexLinear(input_size, hidden_size)
        self.Wh = complexLinear(hidden_size, hidden_size)

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
        h = self.B(x) + torch.matmul(h_prev, self.effective_W())
        return torch.real(self.C(h))


class complexGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(complexGRUCell, self).__init__()
        self.W_r = complexLinear(input_size + hidden_size, hidden_size)
        self.U_r = complexLinear(hidden_size, hidden_size, bias=False)

        self.W_z = complexLinear(input_size + hidden_size, hidden_size)
        self.U_z = complexLinear(hidden_size, hidden_size, bias=False)

        self.W_h = complexLinear(input_size + hidden_size, hidden_size)
        self.U_h = complexLinear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        r = complexSigmoid(self.W_r(torch.cat((x, h_prev), 1)) + self.U_r(h_prev))
        z = complexSigmoid(self.W_z(torch.cat((x, h_prev), 1)) + self.U_z(h_prev))
        h_hat = complexTanh(self.W_h(torch.cat((x, r * h_prev), 1) + self.U_h(h_prev)))
        h_new = (1 - z) * h_prev + z * h_hat
        return h_new


class complexLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(complexLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size, dtype=torch.cfloat))
        self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size, dtype=torch.cfloat))
        self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size, dtype=torch.cfloat))
        self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size, dtype=torch.cfloat))

        self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=torch.cfloat))
        self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=torch.cfloat))
        self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=torch.cfloat))
        self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=torch.cfloat))

        self.bias_i = nn.Parameter(torch.zeros(hidden_size, dtype=torch.cfloat))
        self.bias_f = nn.Parameter(torch.zeros(hidden_size, dtype=torch.cfloat))
        self.bias_c = nn.Parameter(torch.zeros(hidden_size, dtype=torch.cfloat))
        self.bias_o = nn.Parameter(torch.zeros(hidden_size, dtype=torch.cfloat))

        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.W_c)
        nn.init.xavier_uniform_(self.W_o)

        nn.init.orthogonal_(self.U_i)
        nn.init.orthogonal_(self.U_f)
        nn.init.orthogonal_(self.U_c)
        nn.init.orthogonal_(self.U_o)

    def forward(self, x, hidden):
        h, c = hidden
        i = complexSigmoid(x @ self.W_i + h @ self.U_i + self.bias_i)
        f = complexSigmoid(x @ self.W_f + h @ self.U_f + self.bias_f)
        g = complexTanh(x @ self.W_c + h @ self.U_c + self.bias_c)
        o = complexSigmoid(x @ self.W_o + h @ self.U_o + self.bias_o)

        c = f * c + i * g
        h = o * complexTanh(c)

        return h, c
