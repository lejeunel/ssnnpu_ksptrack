import os
from ruamel import yaml
import warnings
import munch

seq_type_dict = {'Tweezer' : ['Dataset00','Dataset01','Dataset02','Dataset03'],
                'Brain' : ['Dataset30','Dataset31','Dataset32','Dataset33'],
                 'Slitlamp' : ['Dataset20','Dataset21','Dataset22','Dataset23'],
                'Cochlea' : ['Dataset10','Dataset11','Dataset12','Dataset13']
}

def dict_to_munch(a_dict):

    return munch.Munch(a_dict)

def load_and_convert(path):
    # Load yml file and convert to munch

    #warnings.simplefilter('ignore',
    #                      ruamel.yaml.error.MantissaNoDotYAML1_1Warning)
    #yaml = yaml.YAML(typ='safe')

    with open(path, 'r') as myfile:
        out = yaml.load(myfile)

    return munch.Munch(out)


def datasetdir_to_type(dir_):
    """ Get sequence type from directory name
    """

    if((dir_ == 'Dataset00') or (dir_ == 'Dataset01') or(dir_ == 'Dataset02') or (dir_ == 'Dataset03')):
        seq_type = 'Tweezer'
    elif((dir_ == 'Dataset30') or (dir_ == 'Dataset31') or(dir_ == 'Dataset32') or (dir_ == 'Dataset33')):
        seq_type = 'Brain'
    elif((dir_ == 'Dataset20') or (dir_ == 'Dataset21') or(dir_ == 'Dataset22') or (dir_ == 'Dataset23')):
        seq_type = 'Slitlamp'
    elif((dir_ == 'Dataset10') or (dir_ == 'Dataset11')or (dir_ == 'Dataset12') or (dir_ == 'Dataset13')):
        seq_type = 'Cochlea'
    else:
        seq_type = 'Unknown'

    return seq_type

def cfg():
    """ Builds default configuration
    Returns Bunch object
    """

    #Paths, dirs, names ...

    precomp_desc_path = '.'
    dataInRoot = '/home/laurent.lejeune/medical-labeling/'
    dataOutRoot = '/home/laurent.lejeune/medical-labeling/'
    dataOutResultDir = ''
    dataOutImageResultDir = 'results' # Where segmentations are saved
    make_datetime_dir = True
    resultDir = 'results'
    gazeDir = 'gaze-measurements'
    gtFrameDir = 'ground_truth-frames'
    dataSetDir = 'Dataset00'
    fileOutPrefix = 'exp'
    framePrefix = 'frame_'
    frameExtension = '.png'
    frameDigits = 4
    frameDir = 'input-frames'
    csvFileName_fg = '2dlocs.csv'
    csvFileType = 'anna'
    #csvFileType = 'pandas'
    csvName_fg = os.path.join(dataInRoot,dataSetDir,gazeDir, csvFileName_fg)

    comment = 'Comment of experiment'

    seqStart = None
    seqEnd = None

    #Descriptors/codebooks ready-to-load.
    feats_files_dir = 'precomp_descriptors'
    feats_compute = True #This value should not be changed here.

    calc_superpix = False
    calc_sp_feats = False
    calc_sp_feats_unet_rec = False
    calc_sp_feats_unet_gaze_rec = False
    calc_entrance = False
    calc_linking = False
    use_hoof = False
    calc_pm = False
    calc_pm = False
    calc_ss = False
    relabel_who = ['']
    force_relabel = False

    #Superpixel segmentation
    compactness = 10.0
    reqdsupervoxelsize = 20000

    # Superpixel transitions
    sp_trans_init_mode = 'overlap'

    #Optical flow
    oflow_alpha = 0.012
    oflow_ratio = 0.75
    oflow_minWidth = 50.
    oflow_nOuterFPIterations = 7.
    oflow_nInnerFPIterations = 1.
    oflow_nSORIterations = 30.

    # Segmentation from TPS
    labelMatPath = ''

    #flow parameters
    sig_a = 0.5
    sig_r_in = 0.4
    sig_r_trans = 0.3

    gaze_radius = 20
    max_paths = None #Set to None for min cost

    n_iter_ksp = 10
    n_iter_lp = 1
    n_iter_lp_gd = 10

    monitor_score = False

    # Bagging---------------
    T = 500
    n_bins = 100
    max_n_feats = 0.013
    max_samples = 2000
    max_depth = None
    bagging_jobs = 4
    #-----------------------

    # Random Forest
    max_n_feats_rf = 'sqrt'
    max_depth_rf = 3
    # ----------------------

    tau_u = 0.45
    n_bins_hoof = 100

    # Metric learning
    n_comp_pca = 3 # for vgg16!
    lfda_k = 7
    lfda_dim = 5
    lfda_n_samps = 1000
    lfda_thresh = 0.8
    pca = False #Overrides LFDA and computes PCA with n_components=lfda_dims

    #Graph parameters
    normNeighbor = 0.08 #Cut edges outside neighborhood
    normNeighbor_in = 0.05 #Cut edges outside neighborhood
    thresh_aux = [] #Cut auxiliary edges below this proba value
    thresh_aux_fix = 0.5
    ksp_tol = 0 #Tolerance of KSP (normalized)

    #Plotting
    roc_xlim = [0,0.4]
    pr_rc_xlim = [0.6,1.]
    testing = False #Won't save results if True

    pm_thr = 0.8 #Threshold on KSP+SS PM for positives selection

    #Unet config-------
    unet_path = 'unet'

    feats_graph = 'unet_gaze_cov'
    #feats_graph = 'unet_tmp'
    feats_pm = 'hsv'

    feat_extr_algorithm = 'unet_gaze'      #{unet, unet_gaze, scp}
    #feat_extr_algorithm = 'scp'      # set unet as feautre extractor algorithm
    unet_seed_val = 0                # use model number 5, it is the best (U-Net, depth 4 with BN)
    unet_model_nbr = 5                # use model number 5, it is the best (U-Net, depth 4 with BN)
    unet_batchsize = 4                # depending on your data size and GPU memory
    unet_nbr_epochs = 20              # number of epochs to train
    #unet_nbr_epochs = 1              # number of epochs to train
    unet_curr_trial = 0               # current trial (to test reproducebility)
    unet_loss = 'mse'                 # use MSE loss
    unet_validation_split = 0         # do not use validation for training
    unet_save_only_best_weight = 1    # ADVANCED, do not change
    unet_save_epoch_interval = 1      # ADVANCED, do not change
    unet_optimizer = 'adam'           # use adam optimizer
    # UNet Rec config
    unet_rec_path = 'unet_rec'           # path for "U-Net Reconstruct"
    unet_gaze_rec_path = 'unet_gaze_rec' # path for "U-Net Gaze Reconstruct"

    unet_seed_val = None                 # seed random shuffeling of images (None for no seeding)
    unet_optimizer = 'adam'              # use adam optimizer
    unet_adam_learning_rate = 0.0001
    unet_adam_beta1 = 0.9
    unet_adam_beta2 = 0.999
    unet_adam_epsilon = 1e-8
    unet_adam_decay = 0.0
    unet_data_use_generator = 1          # use data generator
    unet_data_steps_per_epoch = 2000     # 2000 steps per epoch
    unet_data_gaussian_noise_std = 13    # std of applied gaussian noise
    unet_data_rot_range = 11.25          # max rotation in degrees (+/-)
    unet_data_width_shift = 0.2          # max x-shift by ratio
    unet_data_height_shift = 0.2         # max y-shift by ratio
    unet_data_shear_range = 22.5         # max shearing in degrees (+/-)
    unet_interp_n_jobs = 1               # Num of jobs to interpolate unet features back to orig size
    unet_gaze_gaussian_std = 30          # std dev for creating 2D Gaussian map

    vgg16_batch_size = 16
    vgg16_cuda = False
    vgg16_size = 224

    # ---- Baselines -----

    # Vilarino baseline
    calc_training_patches = True
    patch_size = 128
    scale_factor = 0.25
    overlap_ratio = 0.5
    gamma = 0.0005
    C = 0.01
    vilar_jobs = 2
    vilar_batch = 10
    vilar_joblib_temp_folder = '/tmp'
    vilar_n_frames = -1

    # gaze2segment baseline
    calc_saliencies = False
    calc_gradients = False
    R = [1., 0.8, 0.5, 0.3]
    Rq = [1., 0.5, 0.25]
    c = 3
    K = 64
    patchSize = 7
    overlap = 0.5
    g2s_mph = 0.8
    g2s_radius = 0.08 # Radius around gaze-point to look for max saliency (foreground)
    g2s_pos_thr = 0.5 # Ratio of positive gt pixel in SP

    return locals()
