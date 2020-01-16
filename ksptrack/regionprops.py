import numpy as np
from ksptrack.utils import my_utils as utls
import os
import tqdm
from skimage.measure._regionprops import RegionProperties, PROPS, PROP_VALS
from skimage.measure import _regionprops
from scipy import ndimage as ndi
from warnings import warn
import matplotlib.pyplot as plt


PROPS['ForwardHoof'] = 'forward_hoof'
PROPS['BackwardHoof'] = 'backward_hoof'


class MyRegionProperties(RegionProperties):
    def __init__(self, slice, label, label_image, intensity_image,
                 cache_active,
                 forward_flow, backward_flow, n_bins_hoof=30):
        super().__init__(slice, label, label_image, intensity_image,
                         cache_active)

        self.forward_flow = forward_flow
        self.backward_flow = backward_flow
        self.n_bins_hoof = n_bins_hoof
        self.bins_hoof = np.linspace(-np.pi, np.pi, self.n_bins_hoof + 1)

    @property
    def forward_hoof(self):
        return 1

    def __iter__(self):
        props = PROP_VALS

        if self._intensity_image is None:
            unavailable_props = ('intensity_image',
                                 'max_intensity',
                                 'mean_intensity',
                                 'min_intensity',
                                 'weighted_moments',
                                 'weighted_moments_central',
                                 'weighted_centroid',
                                 'weighted_local_centroid',
                                 'weighted_moments_hu',
                                 'weighted_moments_normalized')

        if self.forward_flow is None:
            unavailable_props += ('forward_hoof')

        if self.backward_flow is None:
            unavailable_props += ('backward_hoof')

            props = props.difference(unavailable_props)

        return iter(sorted(props))

    def __getitem__(self, key):
        value = getattr(self, key, None)
        if value is not None:
            return value
        else:  # backwards compatibility
            return getattr(self, PROPS[key])

    def hoof(self, direction):
        if(direction == 'forward'):
            flow_x = self.forward_flow[0]
            flow_y = self.forward_flow[1]
        else:
            flow_x = self.backward_flow[0]
            flow_y = self.backward_flow[1]

        r = self.coords[:, 0]
        c = self.coords[:, 1]
        angle = np.arctan2(flow_x[r, c], flow_y[r, c])
        norm = np.linalg.norm(
            np.stack((flow_x[r, c],
                      flow_y[r, c])),
            axis=0).ravel()

        # get angle-bins for each pixel
        b_angles = np.digitize(angle, self.bins_hoof).ravel()

        # get bin index for each pixel
        b_mask = [
            np.where(b_angles == b)[0].tolist()
            for b in range(1, len(self.bins_hoof))
        ]
        
        # Sum norms for each bin and each label
        hoof__ = np.asarray([np.sum(norm[b]) for b in b_mask])

        # Normalize w.r.t. L1-norm
        l1_norm = np.sum(hoof__)
        hoof__ = np.nan_to_num(hoof__ / l1_norm)

        return hoof__

    @property
    def backward_hoof(self):
        return self.hoof('backward')

    @property
    def forward_hoof(self):
        return self.hoof('forward')

def regionprops(label_image, intensity_image=None, cache=True,
                coordinates=None,
                backward_flow=None,
                forward_flow=None,
                n_bins_hoof=30):
    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integer type.')

    if coordinates is not None:
        if coordinates == 'rc':
            msg = ('The coordinates keyword argument to skimage.measure.'
                   'regionprops is deprecated. All features are now computed '
                   'in rc (row-column) coordinates. Please remove '
                   '`coordinates="rc"` from all calls to regionprops before '
                   'updating scikit-image.')
            warn(msg, stacklevel=2, category=FutureWarning)
        else:
            msg = ('Values other than "rc" for the "coordinates" argument '
                   'to skimage.measure.regionprops are no longer supported. '
                   'You should update your code to use "rc" coordinates and '
                   'stop using the "coordinates" argument, or use skimage '
                   'version 0.15.x or earlier.')
            raise ValueError(msg)

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = MyRegionProperties(sl, label, label_image,
                                   intensity_image=intensity_image,
                                   cache_active=cache,
                                   backward_flow=backward_flow,
                                   forward_flow=forward_flow,
                                   n_bins_hoof=30)
        regions.append(props)

    return regions
