#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-23 15:57:35
Copyright 2020 liufr
Description: Autoencoder for LSTM
"""
import torch
from torch import nn, optim


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        nn.init.orthogonal_(self.rnn1.weight_ih_l0)
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        nn.init.orthogonal_(self.rnn2.weight_hh_l0)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((-1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, input_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        # x = x.repeat(self.seq_len, self.n_features)
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class RnnAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(
            seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim
        ).to(self.device)
        self.decoder = Decoder(
            seq_len=seq_len, n_features=n_features, input_dim=embedding_dim
        ).to(self.device)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x