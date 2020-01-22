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
                 flow, n_bins_hoof=30,
                 no_motion_thr=1e-2):
        super().__init__(slice, label, label_image, intensity_image,
                         cache_active)

        self.flow = flow
        self.n_bins_hoof = n_bins_hoof
        self.bins_hoof = np.arange(-np.pi, np.pi+1e-6, 2*np.pi/self.n_bins_hoof)
        self.no_motion_thr = no_motion_thr

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

    @property
    def hoof(self):
        # y axis points downwards with pyflow...
        flow_x = self.flow[0]
        flow_y = -self.flow[1]

        r = self.coords[:, 0]
        c = self.coords[:, 1]
        angle = np.arctan2(flow_y[r, c], flow_x[r, c])
        norm = np.linalg.norm(
            np.stack((flow_x[r, c],
                      flow_y[r, c])),
            axis=0).ravel()
        if(norm.sum() / self.area < self.no_motion_thr):
            return np.ones(self.n_bins_hoof) / self.n_bins_hoof
        _mag_greater_zero = norm > 0.0
        pruned_angle = angle[_mag_greater_zero]
        hist, bin_edges = np.histogram(pruned_angle.flatten(),
                                       bins=self.bins_hoof)
        
        hist = hist.astype(np.float32) / (np.sum(_mag_greater_zero) + 1e-6)
        return hist

def regionprops(label_image, intensity_image=None, cache=True,
                coordinates=None,
                flow=None,
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
                                   flow=flow,
                                   n_bins_hoof=n_bins_hoof)
        regions.append(props)

    return regions
