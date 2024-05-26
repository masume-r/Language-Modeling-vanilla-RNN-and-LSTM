import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.lstm(input, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))





