from unet_obj_prior.unet_feat_extr import UNetFeatExtr
from pytorch_utils.dataset import Dataset
from pytorch_utils import utils as ptu
from pytorch_utils import im_utils as ptiu
from ksptrack.utils import my_utils as utls
import os
import matplotlib.pyplot as plt

data_root = '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling'
dataset_dir = 'Dataset24'
frame_dir = 'input-frames'
save_dir = os.path.join(data_root, dataset_dir, 'precomp_descriptors')

cp_fname = 'checkpoint.pth.tar'
bm_fname = 'best_model.pth.tar'

params = {
    'batch_size': 1,
    'cuda': False,
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay_adam': 0,
    'num_epochs': 150,
    'out_dir': save_dir,
    'cp_fname': cp_fname,
    'bm_fname': bm_fname
}

model = UNetFeatExtr(params)

model.model = ptu.load_checkpoint(
    os.path.join(save_dir, bm_fname), model.model, params['cuda'])

fnames = utls.get_images(os.path.join(data_root, dataset_dir, frame_dir))

orig_shape = utls.imread(fnames[0]).shape

in_shape = ptu.comp_unet_input_shape(
    orig_shape, model.model.depth, max_shape=(600, 600))

dataset = Dataset(in_shape, im_paths=fnames, mode='eval', cuda=params['cuda'])

model.model.eval()

idx = [0, 20, 40, 60, 80, 100]
feats = []
ims = []
for i in idx:
    print(i)
    im, prior, truth, im_orig = dataset[i]

    f_ = model.get_features_layer(im.unsqueeze(0)).squeeze(0).detach().numpy()
    f_ = f_.transpose((1, 2, 0))
    feats.append(f_)
    ims.append(ptiu.img_tensor_to_img(im))

idx_ = 4
plt.subplot(121)
plt.imshow(ims[idx_])
plt.subplot(122)
plt.imshow(feats[idx_][..., 10])
plt.show()
