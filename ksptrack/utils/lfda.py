from metric_learn import LFDA
import numpy as np
from sklearn.decomposition import PCA
from ksptrack.utils import my_utils as utls


class myLFDA(LFDA):
    def __init__(self,
                 n_components=None,
                 k=None,
                 embedding_type='weighted'):

        super(myLFDA, self).__init__(n_components=n_components,
                                     k=k,
                                     embedding_type=embedding_type)

    def fit(self, X, y, threshs, n_samp):
        """
        X: features matrix
        y: probability values
        thresh: [low_thr, high_thr]
        n_samp: number of samples to select randomly from each class
        clean_zeros: remove feature that are zeros on all samples
        """

        rand_descs, rand_y = utls.sample_features(X, y, threshs, n_samp)

        super(myLFDA, self).fit(rand_descs.astype(np.float64), rand_y)

    

    def transform(self, x):

        # return super(myLFDA, self).transform(x[..., self.valid_components])
        return super(myLFDA, self).transform(x)
