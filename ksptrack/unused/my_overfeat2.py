import overfeat
import os
import numpy
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize


def is_big_enough(img):
    h = img.shape[0]
    w = img.shape[1]
    return (h >= 231) and (w >= 231)


def resize_img(img):
    # resize and crop into a 231x231 image
    h0 = img.shape[0]
    w0 = img.shape[1]
    d0 = float(min(h0, w0))
    img = img[int(round((h0-d0)/2.)):int(round((h0-d0)/2.)+d0), int(round((w0-d0)/2.)):int(round((w0-d0)/2.)+d0), :]
    img = imresize(img, (231, 231))
    return img


def prepare_img(img_file):
    img = imread(img_file)
    #if not is_big_enough(img):
    #    img = resize_img(img)
    img = resize_img(img)
    img = img.astype(numpy.float32)

    # numpy loads img with colors as last dimension, transpose tensor
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    img = img.reshape(w*h, c)
    img = img.transpose()
    img = img.reshape(c, h, w)
    return img

# initialize overfeat. Note that this takes time, so do it only once if possible
overfeat.init('/home/laurent.lejeune/Documents/overfeat/data/default/net_weight_1', 1)

features = []
classes = []

for file_ in os.listdir('.'):
    if file_.endswith('.jpg'):
        print(file_)
        image = prepare_img(file_)
        b = overfeat.fprop(image)
        f = numpy.copy(overfeat.get_output(22))
        features.append(f)
        clazz = 0 if 'cat' in file_ else 1
        classes.append(clazz)

for feat in features:
    print(feat.shape)

print(classes)

with open('features.pickle', 'w') as f:
    pickle.dump(features, f)

with open('classes.pickle', 'w') as f:
    pickle.dump(classes, f)

