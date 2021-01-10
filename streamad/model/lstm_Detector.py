import copy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from streamad.base import BaseDetector
from streamad.model.lstm_AE import RnnAutoencoder
from torch import nn, optim


class LSTMDetector(BaseDetector):
    def __init__(self, trainset_size=5000, window_size=50, embedding_dim=64):
        self.trainset_size = trainset_size
        self.window_size = window_size
        self.record_count = 0
        self.df_train = []
        self.window_data = []
        self.record_train = self.trainset_size * self.window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = 0
        self.embedding_dim = embedding_dim
        self.criterion = nn.L1Loss(reduction="mean").to(self.device)
        # self.scaler = MinMaxScaler()

    def _train_model(self, n_epochs, save_model=True):
        self.model = RnnAutoencoder(self.window_size, self.n_features, 64).to(
            self.device
        )
        learning_rate = 1e-4
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optim_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.9
        )

        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 100000.0
        # self.window_data = self.scaler.fit_transform(self.window_data)

        for i in range(self.trainset_size - self.window_size):
            self.df_train.append(
                torch.tensor(self.window_data[i : i + self.window_size]).float()
            )

        train_df, val_df = train_test_split(self.df_train, test_size=0.1, shuffle=False)

        for epoch in range(n_epochs):
            self.model = self.model.train()

            train_losses = []
            for seq_true in train_df:
                optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                loss = self.criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            if epoch % 5 == 0:
                optim_scheduler.step()

            val_losses = []
            self.model = self.model.eval()

            with torch.no_grad():
                for seq_true in val_df:
                    seq_true = seq_true.to(self.device)
                    seq_pred = self.model(seq_true)

                    loss = self.criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")
            else:
                print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")

        self.model.load_state_dict(best_model_wts)
        if save_model:
            torch.save(self.model, "./model/lstmAE.pth")
        return self

    def fit_partial(self, X: pd.Series, Y: pd.Series):

        self.record_count += 1

        self.window_data.append(X.values)

        if self.record_count == self.trainset_size:
            self.n_features = X.size
            self._train_model(n_epochs=100)
            self.window_data = []

        return self

    def score_partial(self, X):
        if self.record_count <= self.trainset_size:
            return None
        loss = 0
        with torch.no_grad():
            self.model = self.model.eval()
            seq_true = X.values
            seq_pred = self.model(seq_true)
            loss = self.criterion(seq_pred, seq_true)

        return loss
