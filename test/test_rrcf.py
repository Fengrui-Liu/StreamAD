from streamad.model import rrcf_Detector
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../..")))

from streamad.util import StreamGenerator, StreamStatistic

from streamad.util import UnivariateDS, MultivariateDS
import matplotlib.pylab as pyl


def test_sin():
    n = 730
    A = 50
    center = 100
    phi = 30
    T = 2 * np.pi / 100
    t = np.arange(n)
    sin = A * np.sin(T * t - phi * T) + center
    # add some anomaly data
    sin[235:255] = 80
    sin[600:620] = 0
    y_label = [0] * 727
    y_label = np.array(y_label)
    # change the label
    y_label[235:255] = 1
    y_label[600:620] = 1
    # add some gauss noise for sin
    for i in range(len(sin)):
        sin[i] += 10 * random.gauss(0, 0.1)

    # draw an image for sin
    # x = list(range(0, 730, 1))
    # pyl.plot(x, sin)
    # pyl.show()

    forest = rrcf_Detector.RCForest()

    # can fit data and get score as a whole
    forest.fit(sin)
    y_score = forest.score(sin, 1)

    fpr, tpr, threshold = roc_curve(y_label, y_score)
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    assert roc_auc > 0.8

    # draw an image for roc_curve
    # pyl.figure()
    # lw = 2
    # pyl.figure(figsize=(10, 10))
    # pyl.plot(
    #     fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.4f)" % roc_auc
    # )  ###假正率为横坐标，真正率为纵坐标做曲线
    # pyl.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # pyl.xlim([0.0, 1.0])
    # pyl.ylim([0.0, 1.05])
    # pyl.xlabel("False Positive Rate")
    # pyl.ylabel("True Positive Rate")
    # pyl.title("Receiver operating characteristic example")
    # pyl.legend(loc="lower right")
    # pyl.show()


def test_StreamGenerator():
    ds = UnivariateDS()
    data = ds.data
    label = ds.label
    stream = StreamGenerator(data, label, shuffle=False)
    forest = rrcf_Detector.RCForest()

    # can fit data and get score one-by-one
    for X, z in stream.iter_item():
        forest.fit(X)
    y_score = forest.score(stream, 1)
    # when all the data fitted less than shingle_size(default is 4), we can't get scores,
    # so must delete the first several labels.
    while label.size > forest.index:
        label = np.delete(label, 0)

    fpr, tpr, threshold = roc_curve(label, y_score)
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    assert roc_auc > 0.8

    # draw an image for roc_curve
    # pyl.figure()
    # lw = 2
    # pyl.figure(figsize=(10, 10))
    # pyl.plot(
    #     fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.4f)" % roc_auc
    # )  ###假正率为横坐标，真正率为纵坐标做曲线
    # pyl.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # pyl.xlim([0.0, 1.0])
    # pyl.ylim([0.0, 1.05])
    # pyl.xlabel("False Positive Rate")
    # pyl.ylabel("True Positive Rate")
    # pyl.title("Receiver operating characteristic example")
    # pyl.legend(loc="lower right")
    # pyl.show()


# test_sin()
# test_StreamGenerator()
