from torch.utils.data import DataLoader
from labeling.utils import my_utils as utls
from skimage.transform import resize
from skimage.morphology import binary_dilation
from skimage.morphology import square
import numpy as np

class PatchExtractor(DataLoader):

    def __init__(self, img_path, sp_labels, patch_size, ratio_width=1.0):

        self.img = utls.imread(img_path)
        self.sp_labels = sp_labels
        self.ratio_width = ratio_width
        self.unique_labels = np.unique(sp_labels)
        self.patch_size = patch_size

    def __len__(self):
        'Denotes the total number of samples'
        return self.unique_labels.size

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        print('PathExtractor getitem')
        selem = square(2)
        mask = self.sp_labels == self.unique_labels[index]
        i_mask, j_mask = np.where(mask)
        w = max(j_mask) - min(j_mask)
        h = max(i_mask) - min(i_mask)
        if(w < h):
            cols_to_add = h-w+1
            idx_i = np.arange(min(i_mask), max(i_mask) + 1).astype(int)
            idx_j = np.arange(min(j_mask) - np.floor(cols_to_add/2), max(j_mask) + np.ceil(cols_to_add/2)).astype(int)
        elif(w > h):
            rows_to_add = w-h+1
            idx_i = np.arange(min(i_mask)-np.floor(rows_to_add/2), max(i_mask) + np.ceil(rows_to_add/2)).astype(int)
            idx_j = np.arange(min(j_mask), max(j_mask) + 1).astype(int)
        else:
            idx_i = np.arange(min(i_mask), max(i_mask) + 1)
            idx_j = np.arange(min(j_mask), max(j_mask) + 1)

        patch = resize(
            image.take(idx_i,
                       mode='wrap',
                       axis=0).take(idx_j,
                                    mode='wrap',
                                    axis=1),
                            (self.patch_size,
                             self.patch_size)).astype(np.float32)
        print('done')

        return (patch, self.unique_labels[index])
