import os

def cfg():
    """ Builds default configuration
    Returns Bunch object
    """

    unet_seed_val = 0                # use model number 5, it is the best (U-Net, depth 4 with BN)
    unet_model_nbr = 5                # use model number 5, it is the best (U-Net, depth 4 with BN)
    unet_batchsize = 4                # depending on your data size and GPU memory
    n_epochs = 40              # number of epochs to train
    unet_curr_trial = 0               # current trial (to test reproducebility)
    unet_loss = 'bce'
    unet_validation_split = 0.05
    unet_save_only_best_weight = 1    # ADVANCED, do not change
    unet_save_epoch_interval = 1      # ADVANCED, do not change

    # UNet Rec config
    unet_seed_val = None                 # seed random shuffeling of images (None for no seeding)
    unet_optimizer = 'sgd'              # use adam optimizer
    unet_sgd_nesterov = True
    unet_sgd_momentum = 0.9
    unet_sgd_decay = 1e-6
    unet_sgd_learning_rate = 1e-2
    #unet_kernel_init = 'he_normal'
    unet_kernel_init_sigmoid = 'glorot_uniform'
    unet_kernel_init_relu = 'he_normal'
    unet_adam_learning_rate = 10e-7
    #unet_adam_learning_rate = 0.000001
    unet_adam_beta1 = 0.9
    unet_adam_beta2 = 0.999
    unet_adam_epsilon = 1e-8
    unet_adam_decay = 0.0
    #unet_data_use_generator = 1          # use data generator
    unet_data_use_generator = 1          # use data generator
    unet_data_steps_per_epoch = 2000     # 2000 steps per epoch
    #unet_data_gaussian_noise_std = 13    # std of applied gaussian noise
    unet_data_gaussian_noise_std = 5    # std of applied gaussian noise
    #unet_data_rot_range = 11.25          # max rotation in degrees (+/-)
    unet_data_rot_range = 5.25          # max rotation in degrees (+/-)
    unet_data_width_shift = 0.2          # max x-shift by ratio
    unet_data_height_shift = 0.2         # max y-shift by ratio
    #unet_data_shear_range = 22.5         # max shearing in degrees (+/-)
    unet_data_shear_range = 12.5         # max shearing in degrees (+/-)
    unet_gaze_gaussian_std = 30          # std dev for creating 2D Gaussian map

    checkpoint_period = 5

    early_stopping_patience = 5

    unet_im_size = 512

    unet_class_weight = True

    return locals()

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
