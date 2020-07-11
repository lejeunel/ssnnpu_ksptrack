import imgaug as ia
from imgaug.augmentables import normalize_shape
import operator


class KeypointsOnLabelMap(ia.KeypointsOnImage):
    def __init__(self, keypoints, labels):

        self.check_shape(labels)

        super().__init__(keypoints=keypoints, shape=labels[0].shape)
        self.keypoints = keypoints

        assert isinstance(labels, list), ("Expected list of arrays, got %s." %
                                          (type(labels), ))

        self.shape = normalize_shape(labels[0].shape[:2])
        self.update_labels(labels)

    def check_shape(self, labels):
        for i, l in enumerate(labels):
            assert l.ndim == 2, ("Expected 2D array, got %s." % (l.ndim, ))
            shape = l.shape
            if (i > 0):
                assert l.shape == shape, ("Expected all arrays of equal shape")

    def update_labels(self, labels):
        self.check_shape(labels)

        self.labels = [
            l[k.y_int, k.x_int] for k, l in zip(self.keypoints, labels)
        ]
