import os
import sys

from sklearn.metrics import silhouette_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../../")))

import numpy as np
import pandas as pd
from streamad.util import MultivariateDS, UnivariateDS
from streamad.util import StreamGenerator
from streamad.model import *
from streamad.util import StreamStatistic
from streamad.util import AUCMetric
from scipy import sparse

eval = AUCMetric()



ds = MultivariateDS()
data = pd.DataFrame(ds.data)
print(data.shape[1])
label = pd.DataFrame(ds.label)
stream = StreamGenerator(data,label,shuffle=False)
data = data.values
print(data.shape[1])
data = sparse.csr_matrix(data)
model = RShashDetector(data)

Sscore = []
Y = []
for X,y in stream.iter_item():
    score = model.fit_score(X)
    Sscore.append(score)
    Y.append(y)
    eval.update(y,score)
    print("\r Anomaly score: {} \n".format(score), end="",flush="True")

print('\n AUC_ROC metrics evaluation:',eval.evaluate())
