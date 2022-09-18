from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import ArimaDetector


def test_score():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = ArimaDetector()
    for x in stream.iter_item():
        score = detector.fit_score(x)
        #print(score)
        if score is not None:
            assert type(score) is float

#test_score()
