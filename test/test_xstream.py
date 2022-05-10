from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import xStreamDetector


def test_score():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = xStreamDetector()
    for x in stream.iter_item():
        score = detector.fit_score(x)

        if score is not None:
            assert 0 <= score <= 1
