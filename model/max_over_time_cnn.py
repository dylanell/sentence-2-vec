"""
Max-Over-Time CNN for Sentence-to-Vector encoding.

Ref:
    1. https://arxiv.org/pdf/1408.5882.pdf
    2. https://www.aclweb.org/anthology/W18-5408.pdf
"""


import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad, normalize
from torch.nn import Embedding, Conv2d, Linear, Dropout, Tanh, LeakyReLU


class MaxOverTimeCNN(Module):
    def __init__(
            self, vocab_len: int, wordvec_dim: int, 
            sentvec_dim: int, acceleration: bool=True):
        super(MaxOverTimeCNN, self).__init__()

        # attempt to find a GPU
        if acceleration and torch.cuda.is_available():
            self._device = torch.device('cuda:0')
        else:
            self._device = torch.device('cpu')
        print(f'[INFO]: using {self._device} device')

        # padding token idx
        self._pad_idx = 0

        # make embedding layer and indicate padding idx as <PAD> token idx
        self._embedding = Embedding(
            vocab_len, wordvec_dim, padding_idx=self._pad_idx)

        # make convolutional layers
        self._conv1_1gram = Conv2d(1, 128, (1, wordvec_dim), stride=1)
        self._conv1_2gram = Conv2d(1, 128, (2, wordvec_dim), stride=1)
        self._conv1_3gram = Conv2d(1, 128, (3, wordvec_dim), stride=1)
        self._conv2_1gram = Conv2d(1, 256, (1, 128), stride=1)
        self._conv2_2gram = Conv2d(1, 256, (2, 128), stride=1)
        self._conv2_3gram = Conv2d(1, 256, (3, 128), stride=1)

        # make linear layers
        self._linear = Linear(256*3, sentvec_dim)

        # make dropout layers
        self._dropout = Dropout(p=0.5)

        # make activation layers
        #self._tanh = Tanh()
        self._leaky_relu = LeakyReLU()

    def forward(self, x):
        # pad sequences with padding_value set to <PAD> token idx
        x = pad_sequence(x, batch_first=True, padding_value=self._pad_idx)
        
        # compute embedding outputs
        x = self._embedding(x)

        # pad rows of 0's on top and bottom and unsqueeze channel dim=1
        x_1gram = pad(x, (0, 0, 1, 1), value=0).unsqueeze(1)
        x_2gram = pad(x, (0, 0, 1, 2), value=0).unsqueeze(1)
        x_3gram = pad(x, (0, 0, 2, 2), value=0).unsqueeze(1)

        # compute conv1 layer outputs
        x_1gram = self._leaky_relu(self._conv1_1gram(x_1gram))
        x_2gram = self._leaky_relu(self._conv1_2gram(x_2gram))
        x_3gram = self._leaky_relu(self._conv1_3gram(x_3gram))

        """ Start Conv2 """

        # swap dim1 and dim3 to use filter outputs as new "wordvecs"
        x_1gram = torch.transpose(x_1gram, 3, 1)
        x_2gram = torch.transpose(x_2gram, 3, 1)
        x_3gram = torch.transpose(x_3gram, 3, 1)

        # pad rows of 0's on top and bottom and unsqueeze channel dim=1
        x_1gram = pad(x_1gram, (0, 0, 0, 0), value=0)
        x_2gram = pad(x_2gram, (0, 0, 0, 1), value=0)
        x_3gram = pad(x_3gram, (0, 0, 1, 1), value=0)

        # compute conv2 layer outputs
        x_1gram = self._leaky_relu(self._conv2_1gram(x_1gram))
        x_2gram = self._leaky_relu(self._conv2_2gram(x_2gram))
        x_3gram = self._leaky_relu(self._conv2_3gram(x_3gram))

        """ End Conv2 """

        # squeeze and max-pool over time (sentence length dim)
        x_1gram = torch.max(x_1gram.squeeze(), 2).values
        x_2gram = torch.max(x_2gram.squeeze(), 2).values
        x_3gram = torch.max(x_3gram.squeeze(), 2).values

        # concatenate
        x_concat = torch.cat([x_1gram, x_2gram, x_3gram], dim=1)

        # compute linear layer outputs
        x_out = self._leaky_relu(self._linear(x_concat))

        # normalize output
        x_out = normalize(x_out)

        # add dropout
        x_out = self._dropout(x_out)

        return x_out

    def save(self, path):
        torch.save(
            self.state_dict(), f'{path}/max_over_time_cnn.pt')