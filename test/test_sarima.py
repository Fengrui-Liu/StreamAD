from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import SArimaDetector


def test_sarima():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = SArimaDetector()
    for x in stream.iter_item():
        score = detector.fit_score(x)
        if score is not None:
            assert type(score) is float
