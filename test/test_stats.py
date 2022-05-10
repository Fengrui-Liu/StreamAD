import numpy as np
from streamad.util import (
    MultivariateDS,
    StreamGenerator,
    StreamStatistic,
    UnivariateDS,
)


def test_uni_stats():
    ds = UnivariateDS()
    data = ds.data
    stream = StreamGenerator(data)
    stats = StreamStatistic()

    for X in stream.iter_item():
        stats.update(X)

    assert stats.get_max() == np.max(data)
    assert stats.get_min() == np.min(data)
    assert abs(stats.get_sum() - np.sum(data)) < 0.001
    assert abs(stats.get_mean() - np.mean(data)) < 0.001
    assert abs(stats.get_std() - np.std(data)) < 0.001
    assert abs(stats.get_var() - np.var(data)) < 0.001


def test_multi_stats():
    ds = MultivariateDS()
    data = ds.data
    stream = StreamGenerator(data)
    stats = StreamStatistic()

    for X in stream.iter_item():
        stats.update(X)

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_max(), np.max(data, axis=0))])
        < 0.001
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_min(), np.min(data, axis=0))])
        < 0.001
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_sum(), np.sum(data, axis=0))])
        < 0.001
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_mean(), np.mean(data, axis=0))
            ]
        )
        < 0.001
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_std(), np.std(data, axis=0))])
        < 0.001
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_var(), np.var(data, axis=0))])
        < 0.001
    )
