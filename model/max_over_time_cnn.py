"""
Max-Over-Time CNN for Sentence-to-Vector encoding.

Ref:
    1. https://arxiv.org/pdf/1408.5882.pdf
    2. https://www.aclweb.org/anthology/W18-5408.pdf
"""


import time
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad, relu
from torch.nn import Embedding, Conv2d, Linear


class MaxOverTimeCNN(torch.nn.Module):
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
        self._conv_1gram = Conv2d(1, 100, (1, wordvec_dim), stride=1)
        self._conv_2gram = Conv2d(1, 100, (2, wordvec_dim), stride=1)
        self._conv_3gram = Conv2d(1, 100, (3, wordvec_dim), stride=1)

        # make linear layers
        self._linear = Linear(300, sentvec_dim)

    def forward(self, x):
        # pad sequences with padding_value set to <PAD> token idx
        x = pad_sequence(x, batch_first=True, padding_value=self._pad_idx)
        
        # compute embedding outputs
        x = self._embedding(x)

        # pad rows of 0's on top and bottom and unsqueeze channel dim=1
        x_1gram = pad(x, (0, 0, 1, 1), value=0).unsqueeze(1)
        x_2gram = pad(x, (0, 0, 1, 2), value=0).unsqueeze(1)
        x_3gram = pad(x, (0, 0, 2, 2), value=0).unsqueeze(1)

        # compute conv layer outputs with relu activation
        x_1gram = relu(self._conv_1gram(x_1gram))
        x_2gram = relu(self._conv_2gram(x_2gram))
        x_3gram = relu(self._conv_3gram(x_3gram))

        # squeeze and max-pool over time (sentence length dim)
        x_1gram = torch.max(x_1gram.squeeze(), 2).values
        x_2gram = torch.max(x_2gram.squeeze(), 2).values
        x_3gram = torch.max(x_3gram.squeeze(), 2).values

        # concatenate
        x_concat = torch.cat([x_1gram, x_2gram, x_3gram], dim=1)

        # compute linear layer outputs
        x_out = relu(self._linear(x_concat))

        return x_out
        



