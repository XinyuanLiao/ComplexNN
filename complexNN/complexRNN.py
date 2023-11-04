import torch
from torch import nn
from complexRNNcell import complexRNNCell, complexGRUCell, complexLSTMCell


class complexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [complexRNNCell(input_size, hidden_size, bias)]
        rnn_cells += [complexRNNCell(hidden_size, hidden_size, bias)
                      for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'RNN takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


class complexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [complexGRUCell(input_size, hidden_size, bias)]
        rnn_cells += [complexGRUCell(hidden_size, hidden_size, bias)
                      for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'GRU takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


class complexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [complexLSTMCell(input_size, hidden_size, bias)]
        rnn_cells += [complexLSTMCell(hidden_size, hidden_size, bias)
                      for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (tuple, shape=([num_layers, batch, hidden_size], [num_layers, batch, hidden_size])) :math:`(h_0, c_0)`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'LSTM takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        h0, c0 = init_states
        for h, c, cell in zip(h0, c0, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h, c = cell(cell_input, (h, c))
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


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
