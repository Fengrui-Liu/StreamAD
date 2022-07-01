from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import KNNDetector
from streamad.process import ZScoreThresholder, TDigestThresholder


def test_ZScoreThresholder():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    thresholder = ZScoreThresholder(sigma=2)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = thresholder.normalize(score)
        if score is not None:
            assert 0 <= score <= 1


def test_ZScoreThresholder_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    thresholder = ZScoreThresholder(sigma=2, is_global=True)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = thresholder.normalize(score)
        if score is not None:
            assert 0 <= score <= 1


def test_TDigestThresholder():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    thresholder = TDigestThresholder(percentile_up=93, percentile_down=0)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        normalized_score = thresholder.normalize(score)
        if normalized_score is not None:
            assert 0 <= normalized_score <= 1


def test_TDigestThresholder_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    thresholder = TDigestThresholder(
        percentile_up=93, percentile_down=0, is_global=True
    )

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = thresholder.normalize(score)
        if score is not None:
            assert 0 <= score <= 1
