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
    assert abs(stats.get_sum() - np.sum(data)) < 1e-5
    assert abs(stats.get_mean() - np.mean(data)) < 1e-5
    assert abs(stats.get_std() - np.std(data)) < 1e-5
    assert abs(stats.get_var() - np.var(data)) < 1e-5


def test_multi_stats():
    ds = MultivariateDS()
    data = ds.data
    stream = StreamGenerator(data)
    stats = StreamStatistic()

    for X in stream.iter_item():
        stats.update(X)

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_max(), np.max(data, axis=0))])
        < 1e-5
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_min(), np.min(data, axis=0))])
        < 1e-5
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_sum(), np.sum(data, axis=0))])
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_mean(), np.mean(data, axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_std(), np.std(data, axis=0))])
        < 1e-5
    )

    assert (
        sum([abs(i - j) for i, j in zip(stats.get_var(), np.var(data, axis=0))])
        < 1e-5
    )


def test_windowed_uni_stats():

    ds = UnivariateDS()
    data = ds.data
    stream = StreamGenerator(data)
    stats = StreamStatistic(is_global=False, window_len=10)

    for X in stream.iter_item():
        stats.update(X)

    assert stats.get_max() == np.max(data[-10:])
    assert stats.get_min() == np.min(data[-10:])
    assert stats.get_sum() == np.sum(data[-10:])
    assert stats.get_mean() == np.mean(data[-10:])
    assert stats.get_std() == np.std(data[-10:])
    assert stats.get_var() == np.var(data[-10:])


def test_windows_multi_stats():

    ds = MultivariateDS()
    data = ds.data
    stream = StreamGenerator(data)
    stats = StreamStatistic(is_global=False, window_len=10)

    for X in stream.iter_item():
        stats.update(X)

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_max(), np.max(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_min(), np.min(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_sum(), np.sum(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_mean(), np.mean(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_std(), np.std(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )

    assert (
        sum(
            [
                abs(i - j)
                for i, j in zip(stats.get_var(), np.var(data[-10:], axis=0))
            ]
        )
        < 1e-5
    )
