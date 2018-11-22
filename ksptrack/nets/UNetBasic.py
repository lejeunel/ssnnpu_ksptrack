from .UNetImpl import UNetImpl
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


class UNetBasic(UNetImpl):
    def __init__(self, conf, dir_, im_sample_path):
        UNetImpl.__init__(self, conf, dir_, im_sample_path, n_chans_output=1)

    def data_augmentation_gaussian(self, imgs):
        return super(UNetBasic, self).data_augmentation_gaussian(imgs)

    def train(self,
              conf,
              im_list,
              labels_list,
              checkpoint_path,
              n_epochs,
              initial_epoch=0,
              dir_eval_clbk=None,
              save_examples=True,
              resume_model=None):
        """
        Start training. The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """

        #Read all images
        input_y = self.inputDimY
        input_x = self.inputDimX
        n_chans = self.nbrChannels
        imgs = UNetImpl.preprocess_imgs(self, im_list, input_y, input_x,
                                        n_chans)
        gts = np.asarray([
            scm.imread(labels_list[i]) / 255 for i in range(len(labels_list))
        ])
        gts = gts[..., 0]
        gts = gts[..., np.newaxis].astype(int)

        # Shuffle
        shuff_inds = np.arange(imgs.shape[0])
        np.random.shuffle(shuff_inds)
        imgs = imgs[shuff_inds, ...]
        gts = gts[shuff_inds, ...]

        # Get train and validation indices
        n_samp_train = np.floor(
            len(im_list) * (1 - conf.unet_validation_split)).astype(int)
        ind_train = shuff_inds[0:n_samp_train]
        ind_val = shuff_inds[n_samp_train:]

        # Split sets
        imgs_train = imgs[ind_train, ...].astype('float32')
        gts_train = gts[ind_train, ...].astype('float32')
        imgs_val = imgs[ind_val, ...].astype('float32')
        gts_val = gts[ind_val, ...].astype('float32')

        # Normalize
        mean_train = np.mean(imgs_train.reshape(-1, 3), axis=0)
        std_train = np.std(imgs_train.reshape(-1, 3), axis=0)
        imgs_train = utls.normalize_imgs(imgs_train, mean_train, std_train)
        imgs_val = utls.normalize_imgs(imgs_val, mean_train, std_train)

        gts_val = gts[ind_val, ...]

        # if validation split, monitor validation loss
        monitorloss = 'loss'

        # set the std for gaussian noise adding function based on the dataset std
        self.gausStdForDataAugm = (
            conf.unet_data_gaussian_noise_std / std_train)

        # train datagenerator: use rotation, shearing, shift and gaussian noise
        datagenTrain = ImageDataGenerator(
            featurewise_center=False,
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
            self.logger.info(
                'Save some example images of dataset augmentation...')
            i = 0
            idx = int(len(imgs_train) / 2)
            for _ in datagenTrain.flow(
                    imgs_train[idx:(idx + 1), ...],
                    batch_size=1,
                    save_to_dir=self.dataAugPreviewDir,
                    save_prefix='img',
                    save_format='png'):
                i += 1
                if i > 20:
                    break

        # define callbacks
        model_callbacks = list()

        # keras callback, used to save the weights
        checkpoint_path = os.path.join(
            checkpoint_path, 'model-{epoch:02d}-{val_loss:.2f}.hdf5')
        model_callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitorloss,
                mode='min',
                save_best_only=True,
                period=conf.checkpoint_period,
                verbose=True))

        early_stop_clbk = EarlyStopping(
            monitor=monitorloss,
            patience=conf.early_stopping_patience,
            mode='auto')
        model_callbacks.append(early_stop_clbk)
        # customized
        model_callbacks.append(
            UNetCallback(self.logFilePath, self.weightFilePath))

        if (resume_model is not None):
            self.logger.info('Will resume model: ' + resume_model)
            self.model = load_model(resume_model)
        else:
            self.logger.info('Compile model for train mode...')
            UNetImpl.create_and_compile_model(
                self,
                conf,
                conf.unet_loss,
                learning_phase='train',
                mode='standard')

        ind_eval_clbk = 0
        im_eval_clbk = imgs_val[ind_eval_clbk, ...]
        gt_eval_clbk = gts_val[ind_eval_clbk, ...]

        #Set evaluate callback function
        if (dir_eval_clbk is not None):
            eval_clbk = EvaluateCallback(
                im_eval_clbk,
                gt_eval_clbk,
                out_path=dir_eval_clbk,
                verbose=False)
        else:
            eval_clbk = None

        model_callbacks.append(eval_clbk)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        #Weight classes?
        if (conf.unet_class_weight):
            self.logger.info('Weighting of classes (balanced mode)')
            weights_arr = class_weight.compute_class_weight(
                'balanced', np.unique(gts), gts.ravel())
        else:
            self.logger.info(
                'No weighting of classes (assumed already balanced)')
            weights_arr = None

        # fit the model
        self.logger.info('Start training model: "U-Net"')
        self.logger.info('Total num. of epochs: ' + str(n_epochs))
        self.logger.info('Initial num. of epochs: ' + str(initial_epoch))
        self.logger.info('Additional num. of epochs: ' +
                         str(n_epochs - initial_epoch))
        if conf.unet_data_use_generator:
            # set up generators for training
            trainGenerator = datagenTrain.flow(
                imgs_train,
                gts_train,
                batch_size=conf.unet_batchsize,
                shuffle=True)

            valGenerator = datagenTrain.flow(
                imgs_val,
                gts_val,
                batch_size=conf.unet_batchsize,
                shuffle=True)

            self.model.fit_generator(
                trainGenerator,
                steps_per_epoch=conf.unet_data_steps_per_epoch //
                conf.unet_batchsize,
                epochs=n_epochs,
                initial_epoch=initial_epoch,
                callbacks=model_callbacks,
                validation_data=valGenerator,
                validation_steps=ind_val.shape[0] // conf.unet_batchsize,
                class_weight=weights_arr,
                verbose=1)
        else:
            self.model.fit(
                imgs[ind_train],
                gts[ind_train],
                batch_size=conf.unet_batchsize,
                epochs=n_epochs,
                initial_epoch=initial_epoch,
                validation_split=conf.unet_validation_split,
                class_weight=weights_arr,
                callbacks=model_callbacks)

        # forward the call using mean squared error loss

    def eval(self, conf, weights_path, im_list, mean, std):
        """
        Start training. The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """
        self.logger.info('Preprocessing ' + str(len(im_list)) + ' images...')
        imgs = UNetImpl.preprocess_imgs(self, im_list, self.inputDimY,
                                        self.inputDimX, self.nbrChannels)
        imgs = imgs.astype('float32')
        imgs_norm = utls.normalize_imgs(imgs, mean, std)

        UNetImpl.create_and_compile_model(
            self, conf, conf.unet_loss, learning_phase='eval', mode='standard')

        self.model.load_weights(weights_path)
        preds = self.model.predict(imgs, conf.unet_batchsize)
        #preds = self.model.predict(imgs_norm[0, ...], conf.unet_batchsize)

        return preds.transpose(1, 2, 3, 0)

    def preprocess_and_normalize_imgs(self, im_list):
        return UNetImpl.preprocess_and_normalize_imgs(self, im_list)
