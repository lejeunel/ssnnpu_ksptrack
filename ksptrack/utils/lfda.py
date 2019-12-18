from metric_learn import LFDA
import numpy as np


class myLFDA(LFDA):
    def __init__(self, n_components=None, k=None, embedding_type='weighted'):

        super(myLFDA, self).__init__(n_components=n_components,
                                     k=k,
                                     embedding_type=embedding_type)

    def fit(self, X, y, threshs, n_samp):
        """
        X: features matrix
        y: probability values
        thresh: threshold above which sample is considered positive
        n_samp: number of samples to select randomly from each class
        clean_zeros: remove feature that are zeros on all samples
        """

        n_samp = min((np.sum(y < threshs[0]), np.sum(y > threshs[1]), n_samp))

        replace = False
        if((y < threshs[0]).sum() < n_samp):
            replace = True
        rand_idx_neg = np.random.choice(
            np.where(y < threshs[0])[0], replace=replace, size=n_samp)

        replace = False
        if((y > threshs[1]).sum() < n_samp):
            replace = True
        rand_idx_pos = np.random.choice(
            np.where(y > threshs[1])[0], replace=False, size=n_samp)

        rand_X_pos = X[rand_idx_pos, :]
        rand_X_neg = X[rand_idx_neg, :]
        rand_y_pos = y[rand_idx_pos]
        rand_y_neg = y[rand_idx_neg]
        rand_descs = np.concatenate((rand_X_pos, rand_X_neg), axis=0)

        # Check for samples with all zeros
        # inds_ = np.where(np.sum(rand_descs, axis=1) != 0)[0]
        # rand_descs = rand_descs[inds_, :]

        # Check for components with all zeros
        # self.valid_components = np.where(np.sum(rand_descs, axis=0) != 0)[0]

        # rand_descs = rand_descs[..., self.valid_components]
        rand_y = np.concatenate((np.ones_like(rand_y_pos), np.zeros_like(rand_y_neg)), axis=0)
        # rand_y = rand_y[self.valid_components]

        return super(myLFDA, self).fit(rand_descs, rand_y)

    def transform(self, x):

        # return super(myLFDA, self).transform(x[..., self.valid_components])
        return super(myLFDA, self).transform(x)
