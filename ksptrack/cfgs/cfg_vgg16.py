import os

def cfg():
    """ Builds default configuration
    Returns Bunch object
    """

    seed_val = 0                # use model number 5, it is the best (U-Net, depth 4 with BN)
    model_nbr = 5                # use model number 5, it is the best (U-Net, depth 4 with BN)
    batchsize = 4                # depending on your data size and GPU memory
    n_epochs = 80              # number of epochs to train
    curr_trial = 0               # current trial (to test reproducebility)
    loss = 'bce'
    validation_split = 0.05
    save_only_best_weight = 1    # ADVANCED, do not change
    save_epoch_interval = 1      # ADVANCED, do not change

    # UNet Rec config
    seed_val = None                 # seed random shuffeling of images (None for no seeding)
    optimizer = 'sgd'              # use adam optimizer
    sgd_nesterov = True
    sgd_momentum = 0.9
    sgd_decay = 1e-6
    sgd_learning_rate = 1e-2
    adam_learning_rate = 10e-7
    #unet_adam_learning_rate = 0.000001
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    adam_decay = 0.0
    #unet_data_use_generator = 1          # use data generator
    data_use_generator = 1          # use data generator
    data_steps_per_epoch = 2000     # 2000 steps per epoch
    data_gaussian_noise_std = 13    # std of applied gaussian noise
    data_rot_range = 11.25          # max rotation in degrees (+/-)
    data_width_shift = 0.2          # max x-shift by ratio
    data_height_shift = 0.2         # max y-shift by ratio
    data_shear_range = 22.5         # max shearing in degrees (+/-)
    gaze_gaussian_std = 30          # std dev for creating 2D Gaussian map

    checkpoint_period = 5

    early_stopping_patience = 5

    im_size = 224

    class_weight = True

    return locals()

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
