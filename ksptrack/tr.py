import numpy as np


class Tracklet:
    # Tracklet object

    def __init__(self,
                 id_,
                 in_id,
                 out_id,
                 sps=None,
                 df_ix=None,
                 direction='forward',
                 scale=1,
                 length=1,
                 blocked=True,
                 marked=False):

        self.sps = sps  #List of (frame,sp_label) tuples
        self.proba = None
        self.df_ix = df_ix  #List of indices of sp_pm_df and sp_desc_df.
        self.id_ = id_  #index for identification
        self.in_id = in_id
        self.out_id = out_id
        self.direction = direction  #This is used for merging
        self.scale = scale
        self.length = length
        self.marked = marked
        self.blocked = blocked

    def get_sp_list(self, with_id=True):
        out = []
        for i in range(len(self.sps)):
            if (type(self.sps[i][0][1]) is np.ndarray):
                for j in range(self.sps[i][0][1].shape[0]):
                    if (with_id):
                        out.append((self.id_, self.get_in_frame(),
                                    self.sps[i][0][1][j]))
                    else:
                        out.append((self.get_in_frame(), self.sps[i][0][1][j]))
            else:
                if (with_id):
                    out.append((self.id_, self.get_in_frame(),
                                self.sps[i][0][1]))
                else:
                    out.append((self.get_in_frame(), self.sps[i][0][1]))

        return out

    def set_id(self, new_id):
        self.id_ = new_id

    def get_in(self):
        return self.sps[0]

    def get_out(self):
        return self.sps[-1]

    def get_out_frame(self):
        return self.sps[-1][0][0]

    def get_in_frame(self):
        return self.sps[0][0][0]

    def get_in_label(self):
        return [s[1] for s in self.sps[0]][0]

    def get_out_label(self):
        return [s[1] for s in self.sps[-1]][0]

    def get_head_frame(self, n_frames):
        #Returns frame num that can be linked to head of tracklet
        #n_frames gives number of frames of sequence for border conditions
        if (self.direction == 'forward'):
            hf = self.sps[-1][0][0] + 1
            if (hf > n_frames - 1): return np.NAN
            return hf
        else:
            hf = self.sps[-1][0][0] - 1
            if (hf < 0): return np.NAN
            return hf

    def get_tail_frame(self, n_frames):
        #Returns frame num that can be linked to tail of tracklet
        #n_frames gives number of frames of sequence for border conditions
        if (self.direction == 'forward'):
            tf = self.sps[0][0][0] - 1
            if (tf < 0): return np.NAN
            elif (tf > n_frames - 1): return np.NAN
            return tf
        else:
            tf = self.sps[0][0][0] + 1
            if (tf > n_frames - 1): return np.NAN
            return tf
