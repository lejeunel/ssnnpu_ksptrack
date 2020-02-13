import tqdm
import collections
import numpy as np
import logging


class TrackletManager:
    # Makes dictionary of tracklets with in_frame and out_frame keys

    def __init__(self,
                 sps_man,
                 direction,
                 tls,
                 n_frames):

        self.n_frames = 0
        self.dict_in = dict()
        self.dict_out = dict()
        self.logger = logging.getLogger('TrackletManager')
        self.direction = direction
        self.make_dict(tls, n_frames)
        self.sps_man = sps_man

    def make_dict(self, tls, n_frames):

        self.n_frames = n_frames
        dict_in = collections.defaultdict(list)
        dict_out = collections.defaultdict(list)

        frames = np.arange(0, n_frames)

        self.logger.info('Building input frames dictionary')

        bar = tqdm.tqdm(total=n_frames)
        for i in frames:
            tls_i = [
                t for t in tls
                if ((t.get_in_frame() == i) and (t.blocked == False))
            ]
            dict_in[i] = tls_i
            bar.update(1)
        bar.close()

        self.logger.info('Building output frames dictionary')
        bar = tqdm.tqdm(total=n_frames)
        for i in frames:
            tls_o = [
                t for t in tls
                if ((t.get_out_frame() == i) and (t.blocked == False))
            ]
            dict_out[i] = tls_o
            bar.update(1)
        bar.close()

        self.dict_in = dict_in
        self.dict_out = dict_out

    def get_linkables(self, t_arg, rel_radius, hoof_tau_u,
                      direction='forward'):
        """
        t_arg: Tracklet for which we want linkable tracklets
        mode:
            'head': Will find linkables to head of t_arg
            'tail': Will find linkables to tail of t_arg
        """
        t_linkable = []

        # Get superpixels candidates
        sps = self.sps_man.graph[(t_arg.get_out_frame(),
                                    t_arg.get_out_label())]
        if(direction == 'forward'):
            keys = [k for k in sps.keys() if(k[0] > t_arg.get_out_frame())]
        else:
            keys = [k for k in sps.keys() if(k[0] < t_arg.get_out_frame())]

        sps = {key: value for key, value in sps.items() if(key in keys)}

        # filter sps that don't overlap or use distance
        if(rel_radius == 0):
            sps = {k: v for k, v in sps.items() if(sps[k]['overlap'])}
        else:
            sps = {k: v for k, v in sps.items() if(sps[k]['dist'] < rel_radius)}

        sps = {k: v for k, v in sps.items() if(sps[k][direction] > hoof_tau_u)}

        # check hoof / radius / overlap conditions
        labels_filt = []
        for s, val in sps.items():
            if(direction in val.keys()):
                labels_filt.append(s[1])

        # Get tracklet
        if (direction == 'forward'):
            t_candidates = self.dict_in[t_arg.get_head_frame(self.n_frames)]
            t_linkable = [
                t for t in t_candidates if (t.get_in_label() in labels_filt)
            ]
        else:
            t_candidates = self.dict_in[t_arg.get_head_frame(self.n_frames)]
            t_linkable = [
                t for t in t_candidates if (t.get_in_label() in labels_filt)
            ]

        return t_linkable

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
