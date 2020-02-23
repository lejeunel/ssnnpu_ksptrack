import numpy as np
from imgaug import augmenters as iaa
from ksptrack.siamese.loader import Loader
import torch
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.my_augmenters import Normalize
from torch.utils.data import DataLoader
import tqdm
from ksptrack.siamese.distrib_buffer import target_distribution


class LinkAgentModel(LinkAgentRadius):
    def __init__(self,
                 csv_path,
                 data_path,
                 model,
                 L,
                 entrance_radius=None,
                 cuda=False):

        super().__init__(csv_path, data_path, entrance_radius=entrance_radius)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.L = torch.tensor(L).float().to(self.device)

        self.prepare_feats()

    def prepare_feats(self):
        print('preparing features for linkAgentModel')

        batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        transf = iaa.Sequential(
            [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dl = Loader(self.data_path, normalization=transf)
        dl = DataLoader(dl, collate_fn=dl.collate_fn)

        self.feats = []

        pbar = tqdm.tqdm(total=len(dl))
        for i, data in enumerate(dl):
            data = batch_to_device(data)
            with torch.no_grad():
                res = self.model.dec(data, do_assign=False)
                self.feats.append(res['pooled_aspp_feats'])
            pbar.update(1)
        pbar.close()

    def get_proba(self, f0, l0, f1, l1, *args):

        feat0 = self.feats[f0][l0]
        feat1 = self.feats[f1][l1]

        X = torch.stack((feat0, feat1)).unsqueeze(1)

        proba = self.model.get_probas(X).detach().cpu().numpy()

        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)
        return proba

    def get_proba_(self, f0, l0, f1, l1, *args):

        feat0 = self.feats[f0][l0]
        feat1 = self.feats[f1][l1]

        X = torch.stack((feat0, feat1))

        proba = self.model.get_probas(X).detach().cpu().numpy()

        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)
        return proba

    def get_proba_entrance(self, sp, *args):

        label_user = self.get_closest_label(sp)
        if(label_user is not None):
            return self.get_proba(sp['frame'], label_user, sp['frame'], sp['label'])
        else:
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2, *args):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(frame_1, label_1, frame_2, label_2)

        return proba
