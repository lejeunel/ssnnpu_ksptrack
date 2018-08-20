import os
import glob
import requests
from labeling.feat_extractor.feat_data_loader import FeatDataLoader
from labeling.feat_extractor.myvgg16 import MyVGG16
from scipy import (ndimage,io)
from torchvision import transforms
from torch.utils.data import DataLoader

root_dir = os.path.join('/home',
                          'laurent.lejeune',
                          'medical-labeling',
                          'Dataset00')

image_paths = glob.glob(os.path.join(root_dir,
                                     'input-frames',
                                     '*.png'))

labels = io.loadmat(os.path.join(root_dir,
                              'EE',
                              'sp_labels_ml.mat'))['labels']

image_paths = image_paths[0:10]

f_ind = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])

mypatchloader = FeatDataLoader(image_paths[f_ind],
                               labels[..., f_ind],
                               224,
                               transform)

dataset_loader = DataLoader(dataset=mypatchloader, batch_size=10)

myvgg16 = MyVGG16(cuda=False)

feats = list()
for (patches, labels) in dataset_loader:
    feats.append(myvgg16.get_features(patches))
