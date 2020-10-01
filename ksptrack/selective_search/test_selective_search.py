import time
import glob
import numpy
from skimage import io, draw, segmentation
import features
import color_space
import selective_search
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from os.path import join as pjoin
import os
from unet_region import utils as utls

def save_frames(R, F, out_dir):
    if (os.path.exists(out_dir)):
        print('Deleting content of dir: ' + out_dir)
        fileList = os.listdir(out_dir)
        for fileName in fileList:
            #os.remove(out_dir+"/"+fileName)
            os.remove(os.path.join(out_dir, fileName))

    if (not os.path.exists(out_dir)):
        print('output directory does not exist... creating')
        os.mkdir(out_dir)

    print('Saving images to dir: ' + str(out_dir))
    colors = generate_color_table(R)
    pbar = tqdm.tqdm(total=len(F))
    for depth, label in enumerate(F):
        result = colors[label]
        result = (result * alpha + image * (1. - alpha)).astype(numpy.uint8)
        fn = "%s_%04d.png" % (os.path.splitext(im_name)[0], depth)
        fn = os.path.join(out_dir, fn)
        io.imsave(fn, result)
        pbar.update(1)

    pbar.close()

def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors


data_root = '/home/ubelix/data/medical-labeling/'
data_frame_dir = 'input-frames'
out_dir = 'results'

data_dir = 'Dataset00'
f_num = 0
p_x, p_y = 512, 305

# data_dir = 'Dataset21'
# f_num = 5
# p_x, p_y = 374, 230

# data_dir = 'Dataset30'
# f_num = 18
# p_x, p_y = 480, 258

im_name = sorted(glob.glob(pjoin(data_root, data_dir, data_frame_dir, '*.png')))[f_num]

color = 'rgb'
feature = ['size', 'color', 'texture', 'fill']
output = 'result'
k = 50
alpha = 1.0

label = np.load(pjoin(data_root, data_dir, 'precomp_desc',
                      'sp_labels.npz'))['sp_labels'][..., f_num]

label_contours = segmentation.find_boundaries(label, mode='thick')
image = io.imread(pjoin(data_root, data_dir, data_frame_dir, im_name))[..., 0:3]

img_point = image.copy()
rr, cc = draw.circle(p_y, p_x, 5, shape=img_point.shape)
img_point[rr, cc, :] = (0, 255, 0)
img_point[label_contours, ...] = (255, 0, 0)

prior = utls.make_2d_gauss(img_point.shape[:2], 25, (p_y, p_x))
prior = prior / prior.max()

start_t = time.time()
mask = features.SimilarityMask('size' in feature, 'color' in feature,
                               'texture' in feature, 'fill' in feature)
#R: stores region label and its parent (empty if initial).
# record merged region (larger region should come first)
R, F, g, h = selective_search.hierarchical_segmentation(image,
                                                       k=k,
                                                       feature_mask=mask,
                                                       F0=label,
                                                       to_maxpool=prior)

# suppress warning when saving result images

end_t = time.time()
print('Built hierarchy in ' + str(end_t - start_t) + ' secs')

h = np.array(h)

h_pool = np.mean(h, axis=0)
        
# save_frames(R, F, out_dir)
thrs = [0.9, 0.8, 0.7]
fig, ax = plt.subplots(2, 3)
ax = ax.flatten()
ax[0].imshow(img_point)
ax[1].imshow(prior)
ax[1].set_title('prior')
ax[2].imshow(h_pool)
ax[2].plot(p_x, p_y, 'ro')
ax[2].set_title('h_pool')
ax[3].imshow(h_pool > thrs[0])
ax[3].plot(p_x, p_y, 'ro')
ax[3].set_title(thrs[0])
ax[4].imshow(h_pool > thrs[1])
ax[4].plot(p_x, p_y, 'ro')
ax[4].set_title(thrs[1])
ax[5].imshow(h_pool > thrs[2])
ax[5].plot(p_x, p_y, 'ro')
ax[5].set_title(thrs[2])
fig.show()
