import os
import scipy.misc as scm
import numpy as np
import keras.optimizers as kopt
import keras.losses as kloss
import keras.metrics as kmet
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import random as rd
import h5py
from progressbar import Bar, Percentage, ProgressBar
import pandas as pd
from multiprocessing import Pool
from functools import partial
import logging
from . import utils as utls
from .UNetCallback import UNetCallback
from .UNetModels import  UNetModels


class UNetImpl:

    def __init__(self, conf, unet_path, sample_img_path, img_height=None, img_width=None, n_chans_output=None, locs2d=None):
        """
        Initialization of the U-Net Implementation class

        :param conf:            Global configurations (Bunch object)
        :param unet_path:       Path where stuff are saved
        :param sample_img_path: Path to a sample image to retrieve its properties
        :param img_height:      (optional) Desired image height
        :param img_width:       (optional) Desired image width
        """

        self.logger = logging.getLogger('UNet')

        # to start off, set learning phase to Test
        K.set_learning_phase(0)

        # standard deviation for dataset augmentation function (initially set to zero)
        self.gausStdForDataAugm = 0

        self.locs2d=locs2d

        # check if dataset is grayscale
        sample_im = scm.imread(sample_img_path, mode='RGB')
        # if grayscale, all three channels should be equal

        if np.all(sample_im[:,:,0] == sample_im[:,:,1]):
            self.nbrChannels = 1
        else:
            self.nbrChannels = 3

        if(n_chans_output is None):
            self.nbrChannels_out = self.nbrChannels
        else:
            self.nbrChannels_out = n_chans_output


        # check if we got a weightfile already
        self.weightFilePath, self.logFilePath, self.dataAugPreviewDir = utls.get_file_paths(unet_path, conf)

        # in case there is no resolution specified, use the one of the sample image
        self.imHeight = sample_im.shape[0]
        if img_height==None:
            img_height = self.imHeight
            div_var = 2 ** UNetModels.MODEL_DEPTH
            # check if we can achieve the last layer without ending up in decimal size
            if (float(img_height) / div_var) % 1 != 0:
                # back-calculation of image height
                img_height = int(int(float(img_height) / div_var) * div_var)
                self.logger.info('Cannot take image height, use ' + str(img_height) + ' instead')

        self.imWidth = sample_im.shape[1]
        if img_width == None:
            img_width = self.imWidth
            div_var = 2 ** UNetModels.MODEL_DEPTH
            # check if we can achieve the last layer without ending up in decimal size
            if (float(img_width) / div_var) % 1 != 0:
                # back-calculation of image width
                img_width = int(int(float(img_width) / div_var) * div_var)
                self.logger.info('Cannot take image width, use ' + str(img_width) + ' instead')

        self.inputDimY = img_height
        self.inputDimX = img_width

        self.logger.info('Setup net using:')
        self.logger.info('input size ' + str(self.inputDimY) + ' x ' +
              str(self.inputDimX) + ' x ' + str(self.nbrChannels))
        self.logger.info('num. of output channels ' + str(self.nbrChannels_out))

    def interp_features(self, conf, feat_path, labels, n_jobs=4):
        """
        Generate mean interpolated features from features of middle layer on superpixels

        :param conf:    Configurations (Bunch object)
        :param im_list: List of image-paths
        :param labels:  Array of labels of superpixels
        :return:        Pandas frame with columns {frame, label, descriptor}
        """

        labels = [labels[...,f] for f in range(labels.shape[-1])]


        # initialize hdf5 files
        hdInFile = h5py.File(feat_path, 'r')
        imFeat = hdInFile['raw_feat'][...]
        hdInFile.close()

        # get feat properties
        feat_h = imFeat.shape[1]
        feat_w = imFeat.shape[2]
        feat_d = imFeat.shape[3]
        # feature [0,0] corresponds to this location (x,y) from the input vector
        patDist = (2 ** (UNetModels.MODEL_DEPTH - 1)) - 0.5
        # calculate distance back to the original image resolution using scaling before
        # input to the U-Net
        patDist_y = patDist * (self.imHeight / self.inputDimY)
        patDist_x = patDist * (self.imWidth / self.inputDimX)
        # calculate now the points that correspond to the first, resp. last feature entry (x,y)
        y_s = patDist_y
        y_e = self.inputDimY - 1 - patDist_y
        x_s = patDist_x
        x_e = self.inputDimX - 1 - patDist_x
        # get the vectors
        yy = np.linspace(y_s, y_e, feat_h)
        xx = np.linspace(x_s, x_e, feat_w)

        threadSize = int((len(imFeat) / n_jobs) + 0.999)

        # make feat segment dictionary list
        featSegmList = list()
        i_s = 0
        i_e = 0
        cnt = 0
        while i_e < len(imFeat):
            i_e = i_s + threadSize
            if i_e >= len(imFeat):
                i_e = len(imFeat)
            featSegmList.append(dict())
            featSegmList[cnt]['feat'] = imFeat[i_s:i_e, ...]
            featSegmList[cnt]['segm'] = labels[i_s:i_e]
            cnt += 1
            i_s = i_e

        #x_vec = [utls.upscale_feat(xx, yy, feat_d, featSegmList[i])
        #         for i in range(len(featSegmList))]

        # append interpolating threads
        func = partial(utls.upscale_feat, xx, yy, feat_d)
        with Pool(processes=n_jobs) as pool:
            x_vec = pool.map(func, featSegmList)

        # flatten the list
        x_vec = [val for sublist in x_vec for val in sublist]

        out = []
        for f in range(len(labels)):
            unique_labels = np.unique(labels[f])
            for s in range(len(unique_labels)):
                out.append((f,unique_labels[s],x_vec[f][s,...]))

        out = pd.DataFrame(
            out, columns=["frame", "sp_label", "desc"])
        out.sort_values(['frame', 'sp_label'], inplace=True)

        return out


    def get_input_shape(self):
        """
        Retrieve the input shape (2-dimensions) of the model (#channels not included)

        :return: (tuple) input shape (height, width)
        """
        return (self.inputDimY, self.inputDimX)


    def get_weights_path(self):
        """
        Retrieve weight path

        :return: Path to the weights file
        """
        return self.weightFilePath


    ####################################################################################################################
    # (PRIVATE) MEMBER FUNCTIONS
    ####################################################################################################################
    def _train(self, conf, im_list, loss_name,
               gaze_labels=None, save_examples=False):
        """
        Stores the weight file and the feature at path conf.unet_path

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param loss_name:     Name of the loss ('mse', 'mae', 'msegaze', 'maegaze')
        :param gaze_labels:   (optional) In case of "U-Net Gaze Rec" give here the 2D-Gaussian probability maps
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """

        self.logger.info('Preprocessing ' + str(len(im_list)) + ' images...')
        imgs, imgs_mask, mean, std = self.preprocess_and_normalize_imgs(im_list, gaze_labels=gaze_labels)

        # shuffle dataset forward (can be reverted)
        imgs, order = utls.shuffle_forward(imgs, conf.unet_seed_val)
        imgs_mask, _ = utls.shuffle_forward(imgs_mask, order=order)

        # if validation split, monitor validation loss
        monitorloss = 'loss'

        # set the std for gaussian noise adding function based on the dataset std
        self.gausStdForDataAugm = (conf.unet_data_gaussian_noise_std / std)

        # train datagenerator: use rotation, shearing, shift and gaussian noise
        datagenTrain = ImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          rotation_range=conf.unet_data_rot_range,
                                          width_shift_range=conf.unet_data_width_shift,
                                          height_shift_range=conf.unet_data_height_shift,
                                          shear_range=np.radians(conf.unet_data_shear_range),
                                          zoom_range=0.,
                                          channel_shift_range=0.,
                                          fill_mode='nearest',
                                          cval=0.,
                                          horizontal_flip=False,
                                          vertical_flip=False,
                                          rescale=None,
                                          preprocessing_function=self.data_augmentation_gaussian,
                                          data_format=K.image_data_format())

        if (save_examples):
            self.logger.info('Save some example images of dataset augmentation...')
            i = 0
            idx = int(len(imgs) / 2)
            for _ in datagenTrain.flow(imgs[idx:(idx + 1), ...], batch_size=1, save_to_dir=self.dataAugPreviewDir,
                                       save_prefix='img', save_format='png'):
                i += 1
                if i > 20:
                    break

        # define callbacks
        model_callbacks = list()
        # keras callback, used to save the weights
        model_callbacks.append(ModelCheckpoint(self.weightFilePath, monitor=monitorloss,
                                               mode='min', save_best_only=True))
        # customized
        model_callbacks.append(UNetCallback(self.logFilePath, self.weightFilePath))

        self.logger.info('Compile model for train mode...')

        self.create_and_compile_model(conf, loss_name, learning_phase='train')

        ind_train = np.arange(len(imgs))

        # fit the model
        if conf.unet_data_use_generator:
            # set up generators for training
            trainGenerator = datagenTrain.flow(imgs[ind_train], imgs_mask[ind_train],
                                               batch_size=conf.unet_batchsize, shuffle=True)

            self.model.fit_generator(trainGenerator,
                                     steps_per_epoch=conf.unet_data_steps_per_epoch // conf.unet_batchsize,
                                     epochs=conf.unet_nbr_epochs,
                                     callbacks=model_callbacks,
                                     verbose=1)
        else:
            self.model.fit(imgs[ind_train], imgs_mask[ind_train],
                           batch_size=conf.unet_batchsize,
                           epochs=conf.unet_nbr_epochs,
                           callbacks=model_callbacks)

    def _forward_prop(self, conf, feat_path, im_list, loss_name):
        """
        Applies forward propagation to list of images and save features.

        :param conf:      Configurations (Bunch object)
        :param feat_path: Array of images (normalized through ...)
        :param im_list:   Indices needed for reshuffeling
        :param loss_name: Name of the loss ('mse', 'mae', 'msegaze', 'maegaze')
        :return:          None
        """

        # if in train phase, we need to change to test
        self.logger.info('Recompile model for test-mode...')
        self.create_and_compile_model(conf, loss_name, learning_phase='test')

        # retrieve the features at the middle layer
        model_get_feat = K.function([self.model.layers[0].input],
                                    [self.model.layers[self.layerNbr].output])


        # preprocess the data for latter forward propagation
        imgs, imgs_mask, mean, std = self.preprocess_and_normalize_imgs(im_list)

        self.logger.info('Load model weights...')
        self.model.load_weights(self.weightFilePath)

        # prepare output array of forward propagation
        div_var = 2 ** UNetModels.MODEL_DEPTH
        im_feat = np.zeros((len(imgs), int(self.inputDimY / div_var), int(self.inputDimX / div_var),
                            UNetModels.FEATURE_DIM), dtype='float32')

        # batch wise forward propagation
        ind_s = 0
        ind_e = conf.unet_batchsize
        self.logger.info('Start forward propagation of %i images...' % len(imgs))
        while ind_s < len(imgs):
            # apply forward propagation to batch
            im_feat[ind_s:ind_e, ...] = model_get_feat([imgs[ind_s:ind_e, ...]])[0]
            ind_s += conf.unet_batchsize
            ind_e += conf.unet_batchsize
            if ind_e >= len(imgs):
                ind_e = len(imgs)

        self.logger.info('Saving (downscaled) features to ' + feat_path)
        hd5_file = h5py.File(feat_path, 'w')
        seq = hd5_file.create_dataset('raw_feat', (im_feat.shape), dtype='float32',
                                    compression='gzip', compression_opts=2)
        seq[...] = im_feat
        hd5_file.flush()
        hd5_file.close()

    def _forward_prop_output(self, conf, im_list, loss_name):
        """
        Applies forward propagation to list of images and get output

        :param conf:      Configurations (Bunch object)
        :param im_list:   Indices needed for reshuffeling
        :param loss_name: Name of the loss ('mse', 'mae', 'msegaze', 'maegaze')
        :return:          None
        """

        # if in train phase, we need to change to test
        self.logger.info('Recompile model for test-mode...')
        self.create_and_compile_model(conf, loss_name, learning_phase='test')

        # retrieve the features at the middle layer
        #model_get_output = K.function([self.model.layers[0].input],
        #                            [self.model.layers[self.layerNbr].output])

        # preprocess the data for latter forward propagation
        imgs, imgs_mask, mean, std = self.preprocess_and_normalize_imgs(im_list)


        self.logger.info('Load model weights...')
        self.model.load_weights(self.weightFilePath)

        model_output = self.model.predict(imgs, batch_size=imgs.shape[0], verbose=0, steps=None)

        return model_output


    def preprocess_imgs(self, im_list, post_height, post_width, nbr_channels):
        """
        Read images from list using the according mode and rescale it if necessary

        :param im_list:      Image paths as list
        :param post_height:  Requested height
        :param post_width:   Requested width
        :param nbr_channels: Number of requested color channels (1 = greyscale, 3 = RGB)
        :return:             List of images
        """

        if not isinstance(im_list, list):
            tmp = im_list
            im_list = list()
            im_list.append(tmp)

        # read image according to number of channels
        read_mode = 'RGB'
        img_dim = [len(im_list), post_height, post_width, nbr_channels]
        if nbr_channels == 1:
            read_mode = 'L'
            # if only one dimension, do not take last dimension for image processing
            img_dim = img_dim[:3]

        # prepare return array
        imgs_p = np.ndarray(img_dim, dtype=np.uint8)

        # visualize progress using progressbar
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(im_list)).start()
        for i, item in enumerate(im_list):
            img = scm.imread(item, mode=read_mode)
            if img.shape[0] == post_height and img.shape[1] == post_width:
                imgs_p[i] = img
            else:
                # interp: 'nearest' 'lanczos' 'bilinear' 'bicubic' 'cubic'
                imgs_p[i] = scm.imresize(img, (post_height, post_width, nbr_channels), interp='bilinear')
            pbar.update(i)
        pbar.finish()

        # add an additional axis if grayscale
        if nbr_channels == 1:
            imgs_p = imgs_p[..., np.newaxis]

        return imgs_p


    def preprocess_and_normalize_imgs(self, im_list, gaze_labels=None):
        """
        Preprocess and normalize the images by providing the image path as list and optionally gaze labels

        :param im_list:     List of images as paths
        :param labels: (otional) In case of "U-Net Gaze Rec" give here the 2D-Gaussian probability maps
        :return:            (Tuple) \n
                            imgs:       Preprocessed and normalized images \n
                            imgs_mask:  Image labels squeezed to range [0,1] \n
                            mean:       Mean value over all images \n
                            std:        Standard deviation over all images \n
        """

        # preprocess images by means of loading and resizing
        imgs = self.preprocess_imgs(im_list, self.inputDimY, self.inputDimX, self.nbrChannels)
        imgs = imgs.astype('float32')

        if gaze_labels == None:
            # (U-Net Reconstruct) calculate masks (masks are input images)
            imgs_mask = np.array(imgs)
            imgs_mask /= 255 # scale masks to [0,1]
        else:
            imgs_gauss = np.asarray(gaze_labels)
            imgs_gauss = imgs_gauss[..., np.newaxis]

            # (U-Net Gaze Reconstruct) add gaussian as the last axis
            imgs_mask = np.array(imgs)
            imgs_mask /= 255  # scale masks to [0,1]
            imgs_mask = np.concatenate((imgs_mask, imgs_gauss), axis=-1).astype('float32')
            del imgs_gauss

        # normalize input data
        mean = np.mean(imgs)  # mean for data centering
        std = np.std(imgs)  # std for data normalization
        imgs -= mean
        imgs /= std

        return imgs, imgs_mask, mean, std


    def create_and_compile_model(self,
                                 conf,
                                 loss_name,
                                 learning_phase='train',
                                 mode='features'):
        """
        Create and compile the model (self.model) for given phase (Test, Train)

        :param conf:           Configuration (Bunch object)
        :param loss_name:      Loss ('mse', 'mae', 'msegaze', 'maegaze')
        :param learning_phase: 'Train' or 'Test'
        :return:               None
        """

        # learning phase 0 = test, 1 = train
        K.set_learning_phase(1 if learning_phase == 'train' else 0)

        self.modelClass = UNetModels(self.inputDimY,
                                     self.inputDimX,
                                     self.nbrChannels,
                                     self.nbrChannels_out)

        # generate the requested model
        if(mode == 'features'):
            self.model, self.layerNbr = self.modelClass.gen_model_features()
        else:
            self.model = self.modelClass.gen_model_standard((self.inputDimY,
                                                       self.inputDimX,
                                                       self.nbrChannels))

        # read the loss from the configuration file
        loss = None
        loss_metrics = None
        if loss_name == 'mse':
            loss = kloss.mean_squared_error
            loss_metrics = [kmet.mse, kmet.mae, kmet.binary_crossentropy]
        elif loss_name == 'mae':
            loss = kloss.mean_absolute_error
            loss_metrics = [kmet.mse, kmet.mae, kmet.binary_crossentropy]
        elif loss_name == 'bce':
            loss = kloss.binary_crossentropy
            loss_metrics = [kmet.mse, kmet.mae, kmet.binary_crossentropy]
        elif loss_name == 'msegaze':
            from .UNetLosses import mean_squared_error_gaze, mean_absolute_error_gaze
            loss = mean_squared_error_gaze
            loss_metrics = [mean_squared_error_gaze, mean_absolute_error_gaze]
        elif loss_name == 'maegaze':
            from .UNetLosses import mean_squared_error_gaze, mean_absolute_error_gaze
            loss = mean_absolute_error_gaze
            loss_metrics = [mean_squared_error_gaze, mean_absolute_error_gaze]

        # read the optimizer method from the configuration file
        optimizer = None
        if conf.unet_optimizer == 'sgd':
            optimizer = kopt.SGD(lr=conf.unet_sgd_learning_rate,
                                 momentum=conf.unet_sgd_momentum,
                                 decay=conf.unet_sgd_decay,
                                 nesterov=bool(conf.unet_sgd_nesterov))
        elif conf.unet_optimizer == 'adagrad':
            optimizer = kopt.Adagrad(lr=conf.unet_adagrad_learning_rate,
                                     epsilon=conf.unet_adagrad_epsilon,
                                     decay=conf.unet_adagrad_decay)
        elif conf.unet_optimizer == 'adam':
            optimizer = kopt.Adam(lr=conf.unet_adam_learning_rate,
                                  beta_1=conf.unet_adam_beta1,
                                  beta_2=conf.unet_adam_beta2,
                                  epsilon=conf.unet_adam_epsilon,
                                  decay=conf.unet_adam_decay)
        elif conf.unet_optimizer == 'adamax':
            optimizer = kopt.Adamax(lr=conf.unet_adamax_learning_rate,
                                    beta_1=conf.unet_adamax_beta1,
                                    beta_2=conf.unet_adamax_beta2,
                                    epsilon=conf.unet_adamax_epsilon,
                                    decay=conf.unet_adamax_decay)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=loss_metrics)

    def data_augmentation_gaussian(self, img):
        """
        Adds gaussian noise to an image using standard deviation self.gausStdForDataAugm

        :param img: image of shape (n x m x 3) or (n x m x 1)
        :return:    image with gaussian noise added
        """

        return (img + np.random.normal(0, self.gausStdForDataAugm, img.shape))
