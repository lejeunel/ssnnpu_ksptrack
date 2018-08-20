from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, \
                         UpSampling2D, BatchNormalization, Activation,\
                         Cropping2D, ZeroPadding2D, Flatten, Dense
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers


class UNetModels(object):
    # constant definition
    MODEL_DEPTH = 4
    FEATURE_DIM = 512

    def __init__(self,
                 dimY,
                 dimX,
                 nbrInCh,
                 nbrOutCh=None,
                 kernel_init_relu='he_normal',
                 kernel_init_sigmoid='glorot_uniform'):
        """
        Constructor for model class, initializes some parameters.

        :param dimY:          Input height in pixel
        :param dimX:          Input width in pixel
        :param nbrInCh:       Nbr of channels input
        :param nbrOutCh:      (optional) Nbr of channels output, default or ('None' = for same as input)
        """
        self.dimY = dimY
        self.dimX = dimX
        self.nbrChannels = nbrInCh
        self.kernel_init_sigmoid = kernel_init_sigmoid
        self.kernel_init_relu = kernel_init_relu

        if nbrOutCh == None:
            self.nbrOutChannels = self.nbrChannels
        else:
            self.nbrOutChannels = nbrOutCh

    def get_crop_shape(self, target, refer):

        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def gen_model_features(self):
        """
        Generate the U-Net model

        Structure = Conv -> BN -> ReLu -> Conv -> BN -> ReLu -> Maxpool
                    Skip connections attached to second ReLu
        Depth     = 4

        :return: (tuple) \n
                 Reference to model \n
                 Layer number of deepest layer for feature extraction
        """

        # at which layer the features will be extracted
        feat_layer = 34

        inputs = Input((self.dimY, self.dimX, self.nbrChannels))

        conv1 = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(32, (3, 3), padding='same', activation=None)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu', name='enc_0')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), padding='same', activation=None)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(64, (3, 3), padding='same', activation=None)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu', name='enc_1')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), padding='same', activation=None)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(128, (3, 3), padding='same', activation=None)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu', name='enc_2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), padding='same', activation=None)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(256, (3, 3), padding='same', activation=None)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu', name='enc_3')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), padding='same', activation=None)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same', activation=None)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu', name='feat')(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = Conv2D(256, (3, 3), padding='same', activation=None)(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same', activation=None)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu', name='dec_3')(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(128, (3, 3), padding='same', activation=None)(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same', activation=None)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu', name='dec_2')(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv2D(64, (3, 3), padding='same', activation=None)(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same', activation=None)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu', name='dec_1')(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv2D(32, (3, 3), padding='same', activation=None)(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same', activation=None)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu', name='dec_0')(conv9)

        conv10 = Conv2D(self.nbrOutChannels, (1, 1), activation='sigmoid')(conv9)

        tmp_model = Model(inputs=[inputs], outputs=[conv10])

        # comment this in to get an idea about the layer numbers
        # from keras.utils import plot_model
        # for idx, tp in enumerate(tmp_model.layers):
        #     print('%-55s shape: %-21s layer: %i' % (type(tp), str(tmp_model.layers[idx].output_shape), idx))
        # plot_model(tmp_model,show_shapes=True)

        return tmp_model, feat_layer

    def gen_model_standard(self, img_shape, num_class=1):

        concat_axis = 3
        inputs = Input(shape = img_shape)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1', kernel_initializer=self.kernel_init_relu)(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=self.kernel_init_relu)(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        #conv10 = Conv2D(num_class, (1, 1), activation='relu', kernel_initializer=self.kernel_init)(conv9)
        conv10 = Conv2D(num_class, (1, 1), activation='sigmoid', kernel_initializer=self.kernel_init_sigmoid)(conv9)

        return Model(inputs=inputs, outputs=conv10)

    def gen_model_standard_bkp(self, img_shape, num_class=1):

        concat_axis = 3
        inputs = Input(shape = img_shape)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)

        return Model(inputs=inputs, outputs=conv10)
