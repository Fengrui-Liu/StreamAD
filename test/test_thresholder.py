from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import KNNDetector
from streamad.process import ZScoreThresholder, TDigestThresholder


def test_ZScoreThresholder():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    model = ZScoreThresholder(detector=KNNDetector(), sigma=2)

    for x in stream.iter_item():
        score = model.fit_score(x)
        if score is not None:
            assert 0 <= score <= 1


def test_ZScoreThresholder_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    model = ZScoreThresholder(detector=KNNDetector(), sigma=2, is_global=True)

    for x in stream.iter_item():
        score = model.fit_score(x)
        if score is not None:
            assert 0 <= score <= 1


def test_TDigestThresholder():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    model = TDigestThresholder(
        detector=KNNDetector(), percentile_up=93, percentile_down=0
    )

    for x in stream.iter_item():
        score = model.fit_score(x)
        if score is not None:
            assert 0 <= score <= 1


def test_TDigestThresholder_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    model = TDigestThresholder(
        detector=KNNDetector(),
        percentile_up=93,
        percentile_down=0,
        is_global=True,
    )

    for x in stream.iter_item():
        score = model.fit_score(x)
        if score is not None:
            assert 0 <= score <= 1
