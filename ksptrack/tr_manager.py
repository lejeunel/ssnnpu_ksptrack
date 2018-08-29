import progressbar
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

        #in_frames = np.unique([t.get_in_frame() for t in tls])
        #head_frames = np.unique([t.get_head_frame(n_frames) for t in tls])
        frames = np.arange(0,n_frames)

        self.logger.info('Building input frames dictionary')
        with progressbar.ProgressBar(maxval=n_frames) as bar:
            for i in frames:
                bar.update(i)
                tls_i = [t for t in tls if((t.get_in_frame() == i) and (t.blocked == False))]
                dict_in[i] = tls_i

        self.logger.info('Building output frames dictionary')
        with progressbar.ProgressBar(maxval=n_frames) as bar:
            for i in frames:
                bar.update(i)
                tls_o = [t for t in tls if((t.get_out_frame() == i) and (t.blocked == False))]
                dict_out[i] = tls_o

        self.dict_in = dict_in
        self.dict_out = dict_out

    def get_linkables(self, t_arg, tau_u, mode='head', direction='forward'):

        """
        t_arg: Tracklet for which we want linkable tracklets
        mode:
            'head': Will find linkables to head of t_arg
            'tail': Will find linkables to tail of t_arg
        """
        t_linkable = []

        # Get superpixels candidates
        if((mode == 'head')):
            sps = self.sps_man.graph[(t_arg.get_out_frame(),
                                      t_arg.get_out_label())]
            if(direction == 'forward'):
                keys = [k for k in sps.keys() if(k[0] > t_arg.get_out_frame())]
            else:
                keys = [k for k in sps.keys() if(k[0] < t_arg.get_out_frame())]
                
        elif((mode == 'tail')):
            sps = self.sps_man.graph[(t_arg.get_in_frame(),
                                      t_arg.get_in_label())]
            if(direction == 'forward'):
                keys = [k for k in sps.keys() if(k[0] < t_arg.get_in_frame())]
            else:
                keys = [k for k in sps.keys() if(k[0] > t_arg.get_in_frame())]

        sps = { key:value for key, value in sps.items() if(key in keys)}

        # check HOOF intersection
        labels_filt = []
        for s, val in sps.items():
            if(val[direction] > tau_u): 
                labels_filt.append(s[1])

        # Get tracklet
        if((mode == 'head') & (direction == 'forward')):
            t_candidates = self.dict_in[t_arg.get_head_frame(self.n_frames)]
            t_linkable = [t for t in t_candidates if(t.get_in_label() in labels_filt)]
        elif((mode == 'tail') & (direction == 'forward')):
            t_candidates = self.dict_out[t_arg.get_tail_frame(self.n_frames)]
            t_linkable = [t for t in t_candidates if(t.get_out_label() in labels_filt)]
        elif((mode == 'head') & (direction == 'backward')):
            t_candidates = self.dict_in[t_arg.get_head_frame(self.n_frames)]
            t_linkable = [t for t in t_candidates if(t.get_in_label() in labels_filt)]
        elif((mode == 'tail') & (direction == 'backward')):
            t_candidates = self.dict_out[t_arg.get_tail_frame(self.n_frames)]
            t_linkable = [t for t in t_candidates if(t.get_in_label() in labels_filt)]

        return t_linkable
