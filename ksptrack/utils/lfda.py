from metric_learn import LFDA
import numpy as np
from sklearn.decomposition import PCA
from ksptrack.utils import my_utils as utls


class myLFDA(LFDA):
    def __init__(self,
                 n_components=None,
                 n_components_prestage=None,
                 k=None,
                 n_bins=40,
                 embedding_type='weighted'):

        super(myLFDA, self).__init__(n_components=n_components,
                                     k=k,
                                     embedding_type=embedding_type)

        self.n_components_prestage = n_components_prestage
        if(n_components_prestage is not None):
            self.prestage = PCA(n_components_prestage)
        self.prestage_components_ = None

        self.n_bins = n_bins


    def fit(self, X, y, threshs, n_samp):
        """
        X: features matrix
        y: probability values
        thresh: [low_thr, high_thr]
        n_samp: number of samples to select randomly from each class
        clean_zeros: remove feature that are zeros on all samples
        """

        rand_descs, rand_y = utls.sample_features(X, y, threshs, n_samp,
                                                  n_bins=self.n_bins)

        if(self.n_components_prestage is not None):
            self.prestage_components_ = self.prestage.fit(rand_descs).components_.T
            rand_descs = np.dot(rand_descs, self.prestage_components_)
        else:
            self.prestage_components_ = np.eye(rand_descs.shape[1])

        super(myLFDA, self).fit(rand_descs.astype(np.float64), rand_y)

        self.components_ = np.dot(self.prestage_components_, self.components_.T).T

    

    def transform(self, x):

        # return super(myLFDA, self).transform(x[..., self.valid_components])
        return super(myLFDA, self).transform(x)
