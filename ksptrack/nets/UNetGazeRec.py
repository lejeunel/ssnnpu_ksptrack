from .UNetImpl import UNetImpl
import numpy as np
import scipy.misc as scm
from progressbar import Bar, Percentage, ProgressBar


class UNetGazeRec(UNetImpl):

    def train(self, conf, im_list, locs2d, save_examples=False):
        """
        Start training with Gaze-prior (U-Net Gaze Rec). The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """

        im_shape = scm.imread(im_list[0], mode='L').shape

        self.gazeGaussStd = (conf.unet_gaze_gaussian_std / 100.0) * im_shape[1]

        gz_labels = self.__gen_gaze_labels(locs2d,
                                           (self.inputDimY,
                                            self.inputDimX))

        self.logger.info('Start training model: "U-Net Gaze Rec"')
        # forward the call using mean squared error loss
        super(UNetGazeRec, self)._train(conf,
                                        im_list,
                                        loss_name='msegaze',
                                        gaze_labels=gz_labels,
                                        save_examples=save_examples)

    def forward_prop(self, conf, feat_path, im_list):
        """
        Applies forward propagation to list of images and save features.

        :param conf:      Configurations (Bunch object)
        :param feat_path: Array of images (normalized through ...)
        :param im_list:   Indices needed for reshuffeling
        :return:          None
        """
        super(UNetGazeRec, self)._forward_prop(conf, feat_path, im_list, loss_name='msegaze')

    def forward_prop_output(self, conf, im_list):
        """
        Applies forward propagation to list of images and save features.

        :param conf:      Configurations (Bunch object)
        :param im_list:   Indices needed for reshuffeling
        :return:          None
        """
        return super(UNetGazeRec, self)._forward_prop_output(conf, im_list, loss_name='msegaze')
    ####################################################################################################################
    # (PRIVATE) MEMBER FUNCTIONS
    ####################################################################################################################
    def __gen_gaze_labels(self, locs2d, shape):
        """
        Generate the gaze labels.

        :param locs2d: gaze locations
        :param shape:       (tuple) Shape of the image (height, width)
        :param verbose;     (optional) 0: show nothing in stdout, 1: show state info
        :return:            Gaze labels as a list (per image one item)
        """

        # make kernel of size 6 times the std (odd value)
        kernel_size = np.ceil(self.gazeGaussStd * 6) // 2 * 2 + 1

        unique_frames = np.unique(locs2d[:, 0])
        n_frames = int(np.max(unique_frames)+1)

        # Initialize to uniform (for missing frames)
        gaze_labels = [np.ones(shape)/np.prod(shape) for i in range(n_frames)]

        for idx, gz in enumerate(locs2d.tolist()):
            # do only add gausian if the gaze location is not zero (meaning object not present)
            # inpaint gaussian

            cx = round(gz[-2] * shape[1])
            cy = round(gz[-1] * shape[0])

            g = make_2d_gauss(shape, self.gazeGaussStd, (cy, cx))

            gaze_labels[int(gz[0])] += g

        return gaze_labels

########################################################################################################################
# NON-MEMBER FUNCTIONS
########################################################################################################################
def gaussian_2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    :param shape: (tuple) Shape of the 2D map (height, width)
    :param sigma: Gaussian's standard deviation
    :return:      2D map
    """

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5*((x-x0)/std)**2)

    return y/np.sum(y)
    

def make_2d_gauss(shape, std, center):

    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1,1), (1, shape[1]))

    g = g_x*g_y

    return g/np.sum(g)
