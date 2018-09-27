from .UNetCallback import UNetCallback
from . import utils as utls
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import backend as K
import numpy as np
import tensorflow as tf
import os
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import scipy.misc as scm
from .utils import EvaluateCallback
from keras.applications import VGG16
import logging
import keras.metrics as kmet

class VGG16Basic:

    def __init__(self,conf, dir_, im_sample_path):
        #UNetImpl.__init__(self, conf, dir_, im_sample_path, n_chans_output=1)

        """
        Initialization of the U-Net Implementation class

        :param conf:            Global configurations (Bunch object)
        :param unet_path:       Path where stuff are saved
        :param sample_img_path: Path to a sample image to retrieve its properties
        :param img_height:      (optional) Desired image height
        :param img_width:       (optional) Desired image width
        """

        self.logger = logging.getLogger('VGG16')

        # to start off, set learning phase to Test
        K.set_learning_phase(0)

        # standard deviation for dataset augmentation function (initially set to zero)
        self.gausStdForDataAugm = 0
        n_chans_output = 1

        # check if dataset is grayscale
        sample_im = scm.imread(im_sample_path, mode='RGB')
        # if grayscale, all three channels should be equal

        if np.all(sample_im[:,:,0] == sample_im[:,:,1]):
            self.nbrChannels = 1
        else:
            self.nbrChannels = 3

        if(n_chans_output is None):
            self.nbrChannels_out = self.nbrChannels
        else:
            self.nbrChannels_out = n_chans_output

        img_height = None
        img_width = None

        #self.weightFilePath, self.logFilePath, self.dataAugPreviewDir = utls.get_file_paths(dir_, conf)
        self.logFilePath = os.path.join(dir_, 'log.csv')

    def train(self,
              conf,
              dims,
              im_list,
              labels_list,
              checkpoint_path,
              n_epochs,
              initial_epoch=0,
              dir_eval_clbk=None,
              save_examples=False,
              resume_model=None):
        """
        Start training. The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """

        self.logger.info('Preprocessing ' + str(len(im_list)) + ' images...')
        imgs, _, _, std = utls.preprocess_and_normalize_imgs(im_list,
                                                             dims)
        gts = np.asarray([scm.imread(labels_list[i])/255 for i in range(len(labels_list))])
        gts = gts[...,0]
        gts = gts[...,np.newaxis].astype(int)

        # shuffle dataset forward (can be reverted)
        imgs, order = utls.shuffle_forward(imgs, conf.seed_val)
        gts, _ = utls.shuffle_forward(gts, order=order)

        # if validation split, monitor validation loss
        monitorloss = 'loss'

        # set the std for gaussian noise adding function based on the dataset std
        self.gausStdForDataAugm = (conf.data_gaussian_noise_std / std)

        # train datagenerator: use rotation, shearing, shift and gaussian noise
        datagenTrain = ImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          rotation_range=conf.data_rot_range,
                                          width_shift_range=conf.data_width_shift,
                                          height_shift_range=conf.data_height_shift,
                                          shear_range=np.radians(conf.data_shear_range),
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
        checkpoint_path = os.path.join(checkpoint_path,
                                       'model-{epoch:02d}-{val_loss:.2f}.hdf5')
        model_callbacks.append(ModelCheckpoint(filepath=checkpoint_path,
                                               monitor=monitorloss,
                                               mode='min',
                                               save_best_only=True,
                                               period=conf.checkpoint_period,
                                               verbose=True))

        early_stop_clbk = EarlyStopping(monitor=monitorloss,
                                        patience=conf.early_stopping_patience,
                                        mode='auto')
        model_callbacks.append(early_stop_clbk)
        # customized
        model_callbacks.append(UNetCallback(self.logFilePath, None))

        if(resume_model is not None):
          self.logger.info('Will resume model: ' + resume_model)
          self.model = load_model(resume_model)
        else:
          self.logger.info('Compile model for train mode...')
          self.model = VGG16()
          #UNetImpl.create_and_compile_model(self,
          #                                  conf,
          #                                  conf.unet_loss,
          #                                  learning_phase='train',
          #                                  mode='standard')

        n_samp_train = np.floor(len(imgs)*(1-conf.validation_split)).astype(int)
        ind_train = np.arange(len(imgs))[0:n_samp_train]
        ind_val = np.arange(len(imgs))[n_samp_train:]

        ind_eval_clbk = ind_val[0]
        im_eval_clbk = imgs[ind_eval_clbk,...]
        gt_eval_clbk = gts[ind_eval_clbk,...]

        #Set evaluate callback function
        if(dir_eval_clbk is not None):
            eval_clbk = EvaluateCallback(im_eval_clbk,
                                        gt_eval_clbk,
                                        out_path=dir_eval_clbk,
                                        verbose=False)
        else:
            eval_clbk = None

        model_callbacks.append(eval_clbk)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        #Weight classes?
        if(conf.class_weight):
            self.logger.info('Weighting of classes (balanced mode)')
            weights_arr = class_weight.compute_class_weight('balanced'
                                                            ,np.unique(gts)
                                                            ,gts.ravel())
        else:
            self.logger.info('No weighting of classes (assumed already balanced)')
            weights_arr = None

        # fit the model
        self.logger.info('Start training model: "VGG16"')
        self.logger.info('Total num. of epochs: ' + str(n_epochs))
        self.logger.info('Initial num. of epochs: ' + str(initial_epoch))
        self.logger.info('Additional num. of epochs: '
                         + str(n_epochs - initial_epoch))
        if conf.data_use_generator:
            # set up generators for training
            trainGenerator = datagenTrain.flow(imgs[ind_train], gts[ind_train],
                                               batch_size=conf.batchsize, shuffle=True)

            valGenerator = datagenTrain.flow(imgs[ind_val], gts[ind_val],
                                               batch_size=conf.batchsize, shuffle=True)

            self.model.fit_generator(trainGenerator,
                                     steps_per_epoch=conf.data_steps_per_epoch // conf.batchsize,
                                     epochs=n_epochs,
                                     initial_epoch=initial_epoch,
                                     callbacks=model_callbacks,
                                     validation_data=valGenerator,
                                     validation_steps=ind_val.shape[0]//conf.batchsize,
                                     class_weight=weights_arr,
                                     verbose=1)
        else:
            self.model.fit(imgs[ind_train], gts[ind_train],
                           batch_size=conf.batchsize,
                           epochs=n_epochs,
                           initial_epoch=initial_epoch,
                           validation_split=conf.validation_split,
                           class_weight=weights_arr,
                           callbacks=model_callbacks)

        # forward the call using mean squared error loss

    def eval(self,
             conf,
             weights_path,
             im_list,
             mean,
             std):
        """
        Start training. The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """
        self.logger.info('Preprocessing ' + str(len(im_list)) + ' images...')
        imgs = utls.preprocess_imgs(im_list,
                                    self.inputDimY,
                                    self.inputDimX,
                                    self.nbrChannels)
        imgs = imgs.astype('float32')
        imgs -= mean
        imgs /= std

        self.model = VGG16.VGG16()

        self.model.load_weights(weights_path)
        preds = self.model.predict(imgs, conf.unet_batchsize)

        return preds.transpose(1,2,3,0)


    def data_augmentation_gaussian(self, img):
        """
        Adds gaussian noise to an image using standard deviation self.gausStdForDataAugm

        :param img: image of shape (n x m x 3) or (n x m x 1)
        :return:    image with gaussian noise added
        """

        return (img + np.random.normal(0, self.gausStdForDataAugm, img.shape))
