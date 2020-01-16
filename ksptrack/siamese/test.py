from siamese_sp import utils as utls
from siamese_sp.loader import Loader
from siamese_sp.siamese import Siamese
from siamese_sp.my_augmenters import rescale_augmenter
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from torch.nn import functional as F
from imgaug import augmenters as iaa


path = '/home/ubelix/lejeune/data/medical-labeling/Dataset30/'
transf = iaa.Sequential([
    iaa.Resize(224),
    rescale_augmenter])

dl = Loader(path, augmentation=transf)

dataloader_prev = DataLoader(dl,
                             batch_size=1,
                             shuffle=True,
                             collate_fn=dl.collate_fn,
                             num_workers=0)

model = Siamese()

for data in dataloader_prev:

    edges_to_pool = [[e for e in g.edges] for g in data['graph']]
    res, y = model(data['image'], data['graph'], data['labels'],
                   edges_to_pool)
    fig = utls.make_grid_rag(data, [F.sigmoid(r) for r in res])

    # fig.show()
    fig.savefig('test.png', dpi=200)
    break
