from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import KNNDetector


# Test UnivariateDS
def test_ds():
    model = KNNDetector()
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)

    for x in stream.iter_item():
        model.fit_score(x)


def test_attr():
    model = KNNDetector()
    assert hasattr(model, "data_type")


test_ds()
