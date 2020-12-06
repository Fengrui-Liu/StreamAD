#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-06 19:56:16
Copyright 2020 liufr
Description: [xStream](https://github.com/cmuxstream/cmuxstream-baselines)
"""


import enum
import re
from streamingAD.model.xStream_StreamHash import StreamhashProjector
from streamingAD.base import BaseDetector
import pandas as pd
import numpy as np


class xStreamDetector(BaseDetector):
    def __init__(
        self,
        n_components: int = 50,
        n_chains: int = 100,
        depth: int = 25,
        window_size: int = 25,
    ):
        self.nchains = n_chains
        self.depth = depth
        self.chains = []
        self.projector = StreamhashProjector(
            num_components=n_components, density=1 / 3.0
        )
        self.window_size = window_size
        self.record_count = 0
        self.cur_window = []
        self.ref_window = []
        delta = np.ones(n_components) * 0.5
        self.hs_chains = _hsChains(deltamax=delta, n_chains=n_chains, depth=depth)

    def fit_partial(self, X: pd.Series, Y: pd.Series = None):
        self.record_count += 1

        projected_X = self.projector.fit_transform_partial(X)
        self.cur_window.append(projected_X)
        self.hs_chains.fit(projected_X)

        if self.record_count % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.hs_chains.set_deltamax(deltamax=deltamax)
            self.hs_chains.next_window()

        return self

    def score_partial(self, X):

        X = self.projector.fit_transform_partial(X)

        score = self.hs_chains.score_chains(X).flatten()

        return score

    def _compute_deltamax(self):

        deltamax = np.ptp(self.ref_window, axis=0) / 2.0

        deltamax[deltamax == 0] = 1.0

        return deltamax


class _Chain:
    def __init__(self, deltamax, depth):

        self.depth = depth
        self.deltamax = deltamax
        self.rand = np.random.rand(len(deltamax))
        self.rand_shift = self.rand * deltamax
        self.cmsketch_ref = [{} for _ in range(depth)] * depth
        self.cmsketch_cur = [{} for _ in range(depth)] * depth
        self.is_first_window = True
        self.fs = [np.random.randint(0, len(deltamax)) for _ in range(depth)]

    def bincount(self, X):

        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1
            if depthcount[f] == 1:
                prebins[f] = X[f] + self.rand_shift[f] / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f]

            cmsketch = self.cmsketch_ref[depth]

            for i, prebin in enumerate(prebins):

                l = int(prebin)
                if l in cmsketch:
                    scores[i, depth] = cmsketch[l]
                else:
                    scores[i, depth] = 0.0

        return scores

    def score(self, X):

        scores = self.bincount(X)
        depths = np.arange(1, self.depth + 1)

        scores = np.log2(1.0 + scores) + depths
        return np.min(scores)

    def fit(self, X):

        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[f] = (X[f] + self.rand_shift[f]) / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f]

            if self.is_first_window:

                cmsketch = self.cmsketch_ref[depth]
                for prebin in prebins:

                    l = int(prebin)

                    if l not in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1

                self.cmsketch_ref[depth] = cmsketch
                self.cmsketch_cur[depth] = cmsketch
            else:
                cmsketch = self.cmsketch_cur[depth]
                for prebin in prebins:
                    l = int(prebin)

                    if l not in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1
                self.cmsketch_cur[depth] = cmsketch

        return self

    def next_window(self):
        self.is_first_window = False
        self.cmsketch_ref = self.cmsketch_cur
        self.cmsketch_cur = [{} for _ in range(self.depth)] * self.depth


class _hsChains:
    def __init__(self, deltamax, n_chains: int = 100, depth: int = 25) -> None:
        self.nchains = n_chains
        self.depth = depth
        self.chains = [_Chain(deltamax, depth) for _ in range(n_chains)]

    def score_chains(self, X):

        scores = 0
        for chain in self.chains:
            scores += chain.score(X)

        scores = scores / float(self.nchains)

        return scores

    def fit(self, X):
        for chain in self.chains:
            chain.fit(X)

    def next_window(self):
        for chain in self.chains:
            chain.next_window()

    def set_deltamax(self, deltamax):
        for chain in self.chains:
            chain.deltamax = deltamax
            chain.rand_shift = chain.rand * deltamax