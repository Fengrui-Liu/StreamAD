import os
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from streamad.base import BaseDetector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../../")))

def compute_statistics(scores, labels):
    avg_precision = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, avg_precision


class RShashDetector(BaseDetector):
    """Multivariate RShash Detector."""

    def __init__(
        self,
        data,
        sampling_points=1000,
        decay=0.015,
        num_components=100,
        num_hash_fns=4,
    ):
        """RShash detector for multivariate data.

        Args:
            sampling_points (int, optional):  Sample size. Defaults to 1000.
            decay (float, optional):  Decay rate. Defaults to 0.015.
            num_components (int, optional): Number of ensemble components. Defaults to 100.
            num_hash_fns (int, optional): Number of hash tables. Defaults to 4.
        """
        self.m = num_components
        self.w = num_hash_fns
        self.s = sampling_points
        self.Xall = data
        self.n = self.Xall.shape[0]
        self.dim = self.Xall.shape[1]
        self.count = 0
        self.decay = decay
        self.scores = []
        self.buf = []
        self.num_hash = num_hash_fns
        self.cmsketches = []
        self.effS = max(1000, 1.0 / (1 - np.power(2, -self.decay)))
        print ("setting s to:" + str(self.s))
        print ("setting decay to:" + str(self.decay))
        print ("Effective S=" + str(self.effS))
        self._preprocess()

    def _preprocess(self):
        """Preprocess to get the initial parameter of the model.
        """
        for i in range(self.num_hash):
            self.cmsketches.append({})
        print ("Number of Sketches=" + str(self.cmsketches))
        print ("PreProcessing...")
        self._getMinMax()
        print ("Min=" + str(self.minimum.shape) + " Max=" + str(self.maximum.shape))
        self._sample_f()
        print ("Sampled quantity f ->" + str(self.f))
        self._sample_dims()
        print ("Sample Dimensions:" + str(self.V))
        self._sample_shifts()
        print ("Sampling Shifts:" + str(len(self.alpha)))

    def _getMinMax(self):
        """Get the min and max of the data.
        """
        print(type(self.Xall))
        self.pp_data = self.Xall[: self.s, :]
        self.minimum = np.min(self.pp_data, axis=0).toarray()[0]
        self.maximum = np.max(self.pp_data, axis=0).toarray()[0]
        print (
            "Min Shape="
            + str(self.minimum.shape)
            + " and max shape="
            + str(self.maximum.shape)
        )

    def _sample_dims(self):
        """Get the sampling dimension of the data.
        """
        max_term = np.max((2 * np.ones(self.f.size), list(1.0 / self.f)), axis=0)
        common_term = np.log(self.effS) / np.log(max_term)
        low_value = 1 + 0.5 * common_term
        high_value = common_term
        print ("low_value=" + str(low_value))
        print ("high value=" + str(high_value))
        self.r = np.empty(
            [
                self.m,
            ],
            dtype=int,
        )
        self.V = []
        for i in range(self.m):
            if np.floor(low_value[i]) == np.floor(high_value[i]):
                print (low_value[i], high_value[i], i)
                self.r[i] = 1
            else:
                self.r[i] = min(
                    np.random.randint(low=low_value[i], high=high_value[i]), self.dim
                )
            all_feats = np.array(range(self.pp_data.shape[1]))
            choice_feats = all_feats[
                np.where(self.minimum[all_feats] != self.maximum[all_feats])
            ]
            sel_V = np.random.choice(choice_feats, size=self.r[i], replace=False)
            self.V.append(sel_V)

    def _normalize(self):
        """Nomalizition of X.
        """
        self.X_normalized = (self.pp_data - self.minimum) / (
            self.maximum - self.minimum
        )
        self.X_normalized[np.abs(self.X_normalized) == np.inf] = 0

    def _sample_shifts(self):
        """Get the parameter alpha.
        """
        self.alpha = []
        for r in range(self.m):
            self.alpha.append(
                np.random.uniform(low=0, high=self.f[r], size=len(self.V[r]))
            )

    def _sample_f(self):
        """Get the parameter f.
        """
        self.f = np.random.uniform(
            low=1.0 / np.sqrt(self.effS),
            high=1 - (1.0 / np.sqrt(self.effS)),
            size=self.m,
        )

    def score(self, X:np.ndarray)-> float:
        """Score the current observation.

        Args:
            X (np.ndarray): Current observation.

        Returns:
            float: Anomaly probability.
        """

        if self.count < self.s - 1:
            return 0
        if self.count == self.s:
            burnscore = self.burn_in()
            print(burnscore)
            return 0
        index = self.count - self.s + 1
        X_normalized = (X - self.minimum) / (self.maximum - self.minimum)
        X_normalized[np.abs(X_normalized) == np.inf] = 0
        X_normalized = np.asarray(X_normalized).ravel()
        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            Y[range(len(self.V[r]))] = np.floor(
                (X_normalized[np.array(self.V[r])] + np.array(self.alpha[r]))
                / float(self.f[r])
            )

            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int))

            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError as e:
                    value = (index, 0)

                # Scoring the Instance
                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (index - tstamp))
                c.append(new_wt)

                # Update the instance
                new_tstamp = index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt + 1)

            min_c = min(c)
            c = np.log(1 + min_c)
            if c < 0:
                print ("Wrong here")
                print (c)
                print (mod_entry)
                print ("STOP")
            score_instance = score_instance + c

        score = score_instance / self.m
        if score < 0:
            print ("SOME error @")
            print (index)
        if np.isinf(score):
            print (score_instance, self.m)
            print ("HEY")
        return score/8

    def burn_in(self):
        """Initialization of each component.
        """
        # pp_data has the sample.
        # Normalize the data
        print ("Normalizing")
        self._normalize()
        print ("Normalized")
        for r in range(self.m):
            for i in tqdm(range(self.pp_data.shape[0]), desc="burnin"):
                Y = -1 * np.ones(len(self.V[r]))
                Y[range(len(self.V[r]))] = np.floor(
                    (
                        self.X_normalized[i, np.array(self.V[r])]
                        + np.array(self.alpha[r])
                    )
                    / float(self.f[r])
                )
                mod_entry = np.insert(Y, 0, r)
                mod_entry = tuple(mod_entry.astype(np.int))

                for w in range(len(self.cmsketches)):
                    try:
                        value = self.cmsketches[w][mod_entry]
                    except KeyError as e:
                        value = (0, 0)

                    # Setting Timestamp explicitly to 0
                    value = (0, value[1] + 1)
                    self.cmsketches[w][mod_entry] = value

        print ("Scoring....")

        scores = np.zeros(self.pp_data.shape[0])
        for r in range(self.m):
            for i in range(self.pp_data.shape[0]):
                Y = -1 * np.ones(len(self.V[r]))
                Y[range(len(self.V[r]))] = np.floor(
                    (
                        self.X_normalized[i, np.array(self.V[r])]
                        + np.array(self.alpha[r])
                    )
                    / float(self.f[r])
                )

                mod_entry = np.insert(Y, 0, r)
                mod_entry = tuple(mod_entry.astype(np.int))
                c = []
                for w in range(len(self.cmsketches)):
                    try:
                        value = self.cmsketches[w][mod_entry]
                    except KeyError as e:
                        print ("Something is Wrong. This should not have happened")
                        print ("stop")

                    c.append(value[1])

                c = np.log2(min(c))
                scores[i] = scores[i] + c

        scores = scores / self.m
        return scores/8

    def fit(self, X: np.ndarray):
        """Record and analyse the current observation from the stream. Use the first 1000 sample points as the initialization sample set. Detector collect the init data firstly, and further score observation base on the observed data.

        Args:
            X (np.ndarray): Current observation.
        """
        self.count += 1
        self.n += 1
        self.buf.append(X)
        self.dim = X.shape[0]
        return self
