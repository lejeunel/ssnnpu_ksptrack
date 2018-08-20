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

        if len(im_list) != len(locs2d):
            self.logger.error('Number of gaze locations (%i) must be equal to number of images (%i)' %
                              (len(locs2d), len(im_list)))

        im_shape = scm.imread(im_list[0], mode='L').shape

        self.gazeGaussStd = (conf.unet_gaze_gaussian_std / 100.0) * im_shape[1]

        gz_labels = self.__gen_gaze_labels(locs2d[:, 3:], (self.inputDimY, self.inputDimX))

        # normalize to max value = 1
        gz_labels = [i / i.max() for i in gz_labels]

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
    def __gen_gaze_labels(self, gaze_labels, shape, verbose=1):
        """
        Generate the gaze labels.

        :param gaze_labels: Gaze labels as a n x 2 numpy array
        :param shape:       (tuple) Shape of the image (height, width)
        :param verbose;     (optional) 0: show nothing in stdout, 1: show state info
        :return:            Gaze labels as a list (per image one item)
        """

        # make kernel of size 6 times the std (odd value)
        kernel_size = np.ceil(self.gazeGaussStd * 6) // 2 * 2 + 1
        gauss_kernel = gaussian_2D((kernel_size, kernel_size), self.gazeGaussStd)

        gaze_lab = list()
        pbar = None
        if verbose > 0:
            self.logger.info('Build 2D-Gaussions maps with std = %i px' % self.gazeGaussStd)
            pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(gaze_labels)).start()

        for idx, gz in enumerate(gaze_labels):
            # do only add gausian if the gaze location is not zero (meaning object not present)
            if not np.array_equal(gz, np.array((0.0, 0.0))):
                # inpaint gaussian
                gauss_arr = np.zeros(shape)

                c_x = round(gz[0] * shape[1])
                c_y = round(gz[1] * shape[0])

                edge_x = int(c_x - kernel_size // 2)
                edge_y = int(c_y - kernel_size // 2)

                v_range1 = slice(max(0, edge_y), max(min(edge_y + gauss_kernel.shape[0], gauss_arr.shape[0]), 0))
                h_range1 = slice(max(0, edge_x), max(min(edge_x + gauss_kernel.shape[1], gauss_arr.shape[1]), 0))

                v_range2 = slice(max(0, -edge_y), min(-edge_y + gauss_arr.shape[0], gauss_kernel.shape[0]))
                h_range2 = slice(max(0, -edge_x), min(-edge_x + gauss_arr.shape[1], gauss_kernel.shape[1]))

                gauss_arr[v_range1, h_range1] += gauss_kernel[v_range2, h_range2]
            else:
                # if gaze loc is zero (no object) we use uniform distribution
                gauss_arr = np.ones(shape) / (shape[0] * shape[1])

            gaze_lab.append(gauss_arr)

            if verbose > 0:
                pbar.update(idx)

        if verbose > 0:
            pbar.finish()

        return gaze_lab


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
