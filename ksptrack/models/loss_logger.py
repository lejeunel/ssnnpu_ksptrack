import numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class LossLogger:
    def __init__(self,
                 phase,
                 batch_size,
                 n_samples,
                 save_path,
                 print_mode=1,
                 print_loss_every=15):

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.save_path = save_path
        self.print_mode = print_mode
        self.phase = phase
        self.print_loss_every = print_loss_every

        # This will be a dict with epoch as keys
        # values are (batch, loss) tuples
        self.loss = dict()

    def print_loss_batch(self, epoch, batch, loss):
        print('[{}] batch {}/{}: running_loss: {:.8f}'.format(
            self.phase, batch, self.n_samples, loss))

    def save(self, fname, avg_batch=True):
        l_ = self.loss
        if(avg_batch):
            l_ = {k: [np.mean(v)] for k,v in l_.items()}
        df = pd.DataFrame(l_)
        df.to_csv(os.path.join(self.save_path, fname))

    def update(self, epoch, batch, loss):

        if (epoch not in self.loss.keys()):
            self.loss[epoch] = list()
        self.loss[epoch].append(loss)

        if (self.print_mode == 1):
            losses = np.asarray(self.loss[epoch])
            if (losses.size % self.print_loss_every == 0):
                self.print_loss_batch(epoch, batch, np.mean(loss))

    def get_best_epoch(self):

        epochs = [(k, np.mean(np.asarray(self.loss[k])))
                  for k in self.loss.keys()]

        epochs = np.asarray(epochs)

        min_col = np.argmin(epochs[:, 1])

        return min_col, int(epochs[min_col, 0])

    def get_last_loss(self):

        if(len(self.loss.keys()) == 0):
            return None
        else:
            epoch = max(self.loss.keys())
            return self.loss[epoch][-1]

    def get_last_loss_as_str(self):

        loss = self.get_last_loss()
        if(loss is None):
            return ''
        else:
            return 'Curr. loss: {}'.format(loss)
        
    def get_loss(self, epoch):

        if (epoch in self.loss.keys()):
            return np.mean(np.asarray(self.loss[epoch]))
        else:
            return None

    def print_epoch(self, epoch):

        if (epoch in self.loss.keys()):
            print('[{}] batch_size: {}, epoch: {}, loss: {}'.format(
                self.phase, self.batch_size, epoch, self.get_loss(epoch)))

    #def save(self, fname):
