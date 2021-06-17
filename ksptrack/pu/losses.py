import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ksptrack.pu.utils import df_to_tgt


def cross_entropy_logits(in_, tgt, reduction='none'):
    # in_ : tensor (without logit)
    # tgt : integer (0 or 1)
    tgt = torch.ones_like(in_) * tgt

    return F.binary_cross_entropy_with_logits(in_, tgt, reduction=reduction)


class BalancedBCELoss(nn.Module):
    """
    """
    def __init__(self, pi=0.25):
        super(BalancedBCELoss, self).__init__()
        self.pi = pi

    def forward(self, input, target, pi=None, pi_mul=1.):
        """
        """
        if pi is None:
            pi = self.pi

        if not isinstance(pi, list):
            pi = np.unique(target['frame']).shape[0] * [pi]

        loss = 0
        for in_, tgt, pi_ in zip(input, target, pi):
            target_pos, target_neg, target_aug = df_to_tgt(tgt)

            in_p_plus = in_[(target_pos + target_aug).bool()]
            Rp_plus = cross_entropy_logits(in_p_plus, 1)

            in_u_minus = in_[target_neg.bool()]
            Ru_minus = cross_entropy_logits(in_u_minus, 0)

            loss_p_plus = pi_mul * pi_ * Rp_plus.sum() / target_pos.sum()
            loss_u_minus = (1 -
                            pi_mul * pi_) * Ru_minus.sum() / target_neg.sum()

            loss += loss_p_plus + loss_u_minus

        return loss / len(input)


class PULoss(nn.Module):
    """
    https://arxiv.org/abs/2002.04672
    """
    def __init__(self, pi=0.25, do_ascent=False, pxls=False, beta=0):
        super(PULoss, self).__init__()

        self.pi = pi
        self.beta = beta
        self.pxls = pxls
        self.do_ascent = do_ascent

    def forward(self, input, target, pi=None, pi_mul=1., epoch=0):
        """
        """

        if pi is None:
            pi = self.pi

        if not isinstance(pi, list):
            pi = np.unique(pd.concat(target)['frame']).shape[0] * [pi]

        pos_risk = torch.tensor([0.]).to(input[0].device)
        neg_risk = torch.tensor([0.]).to(input[0].device)

        for in_, tgt, pi_ in zip(input, target, pi):
            target_pos, target_neg = df_to_tgt(tgt)

            in_p_plus = in_[(target_pos).bool()]
            in_u_minus = in_[target_neg.bool()]

            Rp_plus = cross_entropy_logits(in_p_plus, 1)
            Ru_minus = cross_entropy_logits(in_u_minus, 0)

            in_p_minus = in_[target_pos.bool()]
            Rp_minus = cross_entropy_logits(in_p_minus, 0)

            loss_u_minus = Ru_minus
            loss_p_minus = pi_mul * pi_ * Rp_minus

            if target_pos.sum() > 0:
                pos_risk += pi_mul * pi_ * Rp_plus.mean()
                neg_risk += loss_u_minus.mean() - loss_p_minus.mean()
            else:
                neg_risk += loss_u_minus.mean()

        if self.do_ascent and (neg_risk < -self.beta):
            loss = pos_risk - neg_risk

        else:
            loss = pos_risk
            loss += F.relu(neg_risk)

        loss = loss / len(input)
        neg_risk = neg_risk / len(input)
        pos_risk = pos_risk / len(input)

        return {'loss': loss, 'neg_risk': neg_risk, 'pos_risk': pos_risk}


if __name__ == "__main__":
    from ksptrack.pu.set_explorer import SetExplorer
    from torch.utils.data import DataLoader
    from imgaug import augmenters as iaa
    from ksptrack.pu.utils import df_to_tgt
    import matplotlib.pyplot as plt

    transf = iaa.Sequential([
        iaa.OneOf([
            iaa.BilateralBlur(d=8,
                              sigma_color=(100, 150),
                              sigma_space=(100, 150)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.06 * 255)),
            iaa.GammaContrast((1., 2.))
        ])
        # iaa.Flipud(p=0.5),
        # iaa.Fliplr(p=.5),
        # iaa.Rot90((1, 3))
    ])

    dl = SetExplorer('/home/ubelix/lejeune/data/medical-labeling/Dataset00',
                     augmentations=transf,
                     normalization='rescale',
                     resize_shape=512)
    criterion = PULoss(pxls=True)
    dl = DataLoader(dl, collate_fn=dl.collate_fn)

    inp = torch.randn((1, 1, 512, 512))

    for s in dl:
        target = [
            s['annotations'][s['annotations']['frame'] == f]
            for f in s['frame_idx']
        ]
        target_pos, target_neg, target_aug = df_to_tgt(target[0])
        criterion(inp, target)

        plt.subplot(121)
        plt.imshow(np.moveaxis(s['image'][0].detach().numpy(), 0, -1))
        plt.subplot(122)
        plt.imshow(target_pos.squeeze().detach().numpy())
        plt.show()
