from metric_learn import LFDA
import numpy as np


class myLFDA(LFDA):
    def __init__(self, num_dims=None, k=None, embedding_type='weighted'):

        super(myLFDA, self).__init__(num_dims, k, embedding_type)

    def fit(self, X, y, thresh, n_samp, clean_zeros=True):
        """
        X: features matrix
        y: probability values
        thresh: threshold above which sample is considered positive
        n_samp: number of samples to select randomly from each class
        clean_zeros: remove feature that are zeros on all samples
        """

        n_samp = min((np.sum(y > thresh), n_samp))

        if (clean_zeros):
            X, unq_idx = np.unique(X, axis=0, return_index=True)
            y = (y > thresh).astype(int)[unq_idx]
        else:
            y = (y > thresh).astype(int)

        rand_idx_pos = np.random.choice(
            np.where(y > 0)[0], replace=False, size=n_samp)
        rand_idx_neg = np.random.choice(
            np.where(y == 0)[0], replace=False, size=n_samp)
        rand_X_pos = X[rand_idx_pos, :]
        rand_X_neg = X[rand_idx_neg, :]
        rand_y_pos = y[rand_idx_pos]
        rand_y_neg = y[rand_idx_neg]
        rand_descs = np.concatenate((rand_X_pos, rand_X_neg), axis=0)

        # Check for samples with all zeros
        inds_ = np.where(np.sum(rand_descs, axis=1) != 0)[0]
        rand_descs = rand_descs[inds_, :]

        # Check for components with all zeros
        self.valid_components = np.where(np.sum(rand_descs, axis=0) != 0)[0]
        rand_descs = rand_descs[..., self.valid_components]

        rand_y = np.concatenate((rand_y_pos, rand_y_neg), axis=0)

        return super(myLFDA, self).fit(rand_descs, rand_y)

    def transform(self, x):

        return super(myLFDA, self).transform(x[..., self.valid_components])
