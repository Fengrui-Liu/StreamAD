from streamad.util import StreamGenerator, UnivariateDS, MultivariateDS
from streamad.model import HSTreeDetector


def test_score():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = HSTreeDetector()
    for x in stream.iter_item():
        score = detector.fit_score(x)

        if score is not None:
            assert 0 <= score <= 1


def test_multi_score():
    ds = MultivariateDS()
    stream = StreamGenerator(ds.data)
    detector = HSTreeDetector()
    for x in stream.iter_item():
        score = detector.fit_score(x)

        if score is not None:
            assert 0 <= score <= 1
