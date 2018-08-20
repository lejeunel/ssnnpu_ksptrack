from .UNetImpl import UNetImpl


class UNetRec(UNetImpl):

    def train(self, conf, im_list, save_examples=False):
        """
        Start training in autoencoder fashion (U-Net Rec). The model weights will be stored to the path given in the
        configuration (conf).

        :param conf:          Configuration (Bunch object)
        :param im_list:       List of paths of images
        :param save_examples: (optional) Save example of transformed images (augmentation)
        :return:              None
        """

        self.logger.info('Start training model: "U-Net Rec"')
        # forward the call using mean squared error loss
        super(UNetRec, self)._train(conf, im_list, loss_name='mse', gaze_labels=None, save_examples=save_examples)


    def forward_prop(self, conf, feat_path, im_list):
        """
        Applies forward propagation to list of images and save features.

        :param conf:      Configurations (Bunch object)
        :param feat_path: Array of images (normalized through ...)
        :param im_list:   Indices needed for reshuffeling
        :return:          None
        """
        super(UNetRec, self)._forward_prop(conf, feat_path, im_list, loss_name='mse')
