import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, vocab_size, input_embed_size, rnn_hidden_size, rnn_num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.num_layers = rnn_num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_embed_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h, train=True):

        x = self.embed(x)

        # Forward propagate RNN
        out, h = self.lstm(x, (h0, c0))

        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h
