#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-14 15:33:16
Copyright 2020 liufr
Description: LSTM initialization
"""

from numpy.core.fromnumeric import size
import torch
from torch import nn
from streamingAD.base import BaseDetector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pdb


class LSTMDetector(BaseDetector):
    def __init__(
        self,
        train_set_size=5000,
        step_num=10,
        batch_num=8,
        iteration=300,
        log_path=None,
        hidden_num=25,
    ):
        self.record_count = 0
        self.df_train = []
        self.df_label = []
        self.batch_num = batch_num
        self.step_num = step_num
        self.trainset_size = train_set_size
        self.iteration = iteration
        self.hidden_num = hidden_num

    def _preprocess_train_data(self):
        # scaling
        scaler = MinMaxScaler()
        scaler.fit(self.df_train)
        self.df_train = pd.DataFrame(scaler.transform(self.df_train))
        # print(self.df_train)

        normal_list = []
        anomaly_list = []

        windows = [
            self.df_label.iloc[w * self.step_num : (w + 1) * self.step_num, :]
            for w in range(self.df_label.index.size // self.step_num)
        ]

        for w in windows:
            if w[w["label"] != 0].size == 0:
                normal_list += [i for i in w.index]
            else:
                anomaly_list += [i for i in w.index]

        self.df_normal = self.df_train.iloc[normal_list]
        self.df_anomaly = self.df_train.iloc[anomaly_list]
        del self.df_train

        tmp = self.df_normal.index.size // self.step_num // 10
        assert tmp > 0, (
            "Too small normal dataset with %d rows" % self.df_normal.index.size
        )

        self.sn = self.df_normal.iloc[: tmp * 5 * self.step_num, :]
        self.vn1 = self.df_normal.iloc[
            tmp * 5 * self.step_num : tmp * 8 * self.step_num, :
        ]
        self.vn2 = self.df_normal.iloc[
            tmp * 8 * self.step_num : tmp * 9 * self.step_num, :
        ]
        self.tn = self.df_normal.iloc[tmp * 9 * self.step_num :, :]

        tmp_a = self.df_anomaly.index.size // self.step_num // 2
        self.va = (
            self.df_anomaly.iloc[: tmp_a * self.step_num, :]
            if tmp_a != 0
            else self.df_anomaly
        )
        self.ta = (
            self.df_anomaly.iloc[tmp_a * self.step_num :, :]
            if tmp_a != 0
            else self.df_anomaly
        )

        self.va_labels = self.df_label.iloc[anomaly_list]
        self.va_labels = self.va_labels[: self.va.index.size]
        print("Local preprocessing finished.")

        self.sn_list = [
            self.sn[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.sn.index.size // self.step_num)
        ]
        self.va_list = [
            self.va[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.va.index.size // self.step_num)
        ]
        self.vn1_lsit = [
            self.vn1[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.vn1.index.size // self.step_num)
        ]
        self.vn2_lsit = [
            self.vn1[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.vn2.index.size // self.step_num)
        ]
        self.tn_list = [
            self.tn[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.tn.index.size // self.step_num)
        ]
        self.ta_list = [
            self.ta[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.ta.index.size // self.step_num)
        ]
        self.va_labels_list = [
            self.va_labels[self.step_num * i : self.step_num * (i + 1)].values
            for i in range(self.va_labels.index.size // self.step_num)
        ]
        self.elem_num = self.sn.shape[1]

        return self

    def fit_partial(self, X, Y):
        self.record_count += 1

        if self.record_count <= self.trainset_size:
            self.df_train.append(X.values)
            self.df_label.append(Y.values)
            return self

        self.df_train = pd.DataFrame(self.df_train)
        self.df_label = pd.DataFrame(self.df_label, columns=["label"])
        # print(self.df_train)
        self._preprocess_train_data()
        self._en_decoder()
        # Training

        return True

    def score_partial(self, X):
        return super().score_partial(X)
