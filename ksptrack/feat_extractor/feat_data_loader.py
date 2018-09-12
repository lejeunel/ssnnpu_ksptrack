import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG
import os
from ksptrack.utils import my_utils as utls
from PIL import Image
from skimage.transform import resize
from skimage.morphology import binary_dilation
from skimage.morphology import square
from progressbar import ProgressBar as pbar
import numpy as np
import matplotlib.pyplot as plt

class PatchDataLoader(Dataset):

    def __init__(self,
                 img_path,
                 sp_labels,
                 patch_size,
                 transform,
                 batch_size = 10):

        self.patch_size = patch_size
        self.sp_labels = sp_labels
        self.img_path = img_path
        self.transform = transform
        self.unique_sp_labels = np.unique(self.sp_labels)
        self.data_loader = DataLoader(self, batch_size=batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        # stuff
        return self.extract_patch(index)

    def __len__(self):
        return self.unique_sp_labels.size

    def extract_patch(self, index):
        """Load an image and convert it to a torch tensor."""

        img = utls.imread(self.img_path)
        selem = square(2)
        this_mask = self.sp_labels == self.unique_sp_labels[index]
        i_mask, j_mask = np.where(this_mask)
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

        patch = resize(img.take(idx_i,
                                mode='wrap',
                                axis=0).take(idx_j,
                                            mode='wrap',
                                            axis=1),
                    (self.patch_size, self.patch_size)).astype(np.float32)

        # Convert to PIL and apply torch transform
        patch = (patch * 255 / np.max(patch)).astype('uint8')
        patch = Image.fromarray(patch)
        patch = self.transform(patch).squeeze()

        return (patch, self.unique_sp_labels[index])

    @staticmethod
    def patch_dict_to_tensor(patches):

        patches_list = [patches[k] for k in patches.keys()]
        patches_tensor = torch.stack(patches_list).squeeze()

        return patches_tensor, patches.keys()
            
