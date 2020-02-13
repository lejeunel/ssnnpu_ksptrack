from torch.utils.data import Dataset, DataLoader
from ksptrack.siamese.loader import Loader, scale_boxes
import torch 
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, draw

dl = Loader(
    '/home/ubelix/lejeune/data/medical-labeling/Dataset30')

label = 700
frame = 40
scale_factor = 2

sample = dl[frame]
im = sample['image']
labels = sample['labels'][..., 0]
sp = labels == label
contour = segmentation.find_boundaries(labels)
im[contour, ...] = (255, 0, 0)

bbox = sample['bboxes'][label]
rr, cc = draw.rectangle_perimeter(start=(bbox[1], bbox[0]),
                                end=(bbox[3], bbox[2]),
                                shape=labels.shape)
im[rr, cc, ...] = (0, 0, 255)

bbox_scaled = scale_boxes(sample['bboxes'], scale_factor)[label]
rr, cc = draw.rectangle_perimeter(start=(bbox_scaled[1], bbox_scaled[0]),
                                end=(bbox_scaled[3], bbox_scaled[2]),
                                shape=labels.shape)
im[rr, cc, ...] = (0, 255, 0)

print(bbox)
print(bbox_scaled)

plt.subplot(121)
plt.imshow(im)
plt.subplot(122)
plt.imshow(sp)
plt.show()
