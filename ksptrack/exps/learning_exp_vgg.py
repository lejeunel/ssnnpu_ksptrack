from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import (color, io, segmentation, transform)
import my_utils as utls
import gazeCsv as gaze
import learning_dataset
from itertools import cycle
import features
import matplotlib.gridspec as gridspec
import logging

"""
Computes segmentation frames, ROC curves on single sequence
"""

def get_all_scores(y, y_pred, n_points):

    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_= auc(np.asarray(fpr).ravel(), np.asarray(tpr).ravel())
    pr, rc, _ = precision_recall_curve(y, y_pred)
    probas_thr = np.linspace(0,1,n_points)
    f1 = 2 * (pr * rc) / (pr + rc)

    return(fpr, tpr, auc_, pr, rc, f1, probas_thr)

def get_pm_array(labels,descs,probas,idx=None):

    if(idx is None):
        idx = np.arange(labels.shape[-1])

    descs_aug = descs
    descs_aug['proba'] = pd.Series(probas, index=descs_aug.index)
    scores = labels.copy().astype(float)
    with progressbar.ProgressBar(
            maxval=scores.shape[2]) as bar:
        for i in idx:
            bar.update(i)
            this_frame_pm_df = descs_aug[descs_aug['frame'] == i]
            dict_keys = this_frame_pm_df['sp_label']
            dict_vals = this_frame_pm_df['proba']
            dict_map = dict(zip(dict_keys, dict_vals))
            for k, v in dict_map.items():
                scores[scores[...,i]==k,i] = v

    return scores

def write_frames_train(X,y,root_dir,logger):

    dir_img = os.path.join(root_dir,'img')
    dir_gt = os.path.join(root_dir,'gt')
    imgs_paths = []
    gts_paths = []

    if( not os.path.exists(root_dir)):
        os.makedirs(root_dir)

    for i in range(X.shape[-1]):
        path_ = os.path.join(dir_img, 'frame'+str(i)+'.png')
        imgs_paths.append(path_)

    for i in range(y.shape[-1]):
        path_ = os.path.join(dir_gt, 'frame'+str(i)+'.png')
        gts_paths.append(path_)

    if( not os.path.exists(dir_img)):
        os.makedirs(dir_img)
        for i in range(X.shape[-1]):
            io.imsave(imgs_paths[i], X[...,i])

    if( not os.path.exists(dir_gt)):
        os.makedirs(dir_gt)
        for i in range(y.shape[-1]):
            io.imsave(gts_paths[i], y[...,i])

    return imgs_paths, gts_paths

def write_frames_pred(y,root_dir,logger,nchans=3):

    dir_img = os.path.join(root_dir,'img')
    imgs_paths = []

    if( not os.path.exists(root_dir)):
        os.makedirs(root_dir)

    for i in range(y.shape[-1]):
        path_ = os.path.join(dir_img, 'frame'+str(i)+'.png')
        imgs_paths.append(path_)

    if(not os.path.exists(dir_img)):
        os.makedirs(dir_img)
    for i in range(y.shape[-1]):
        if(nchans == 3):
            io.imsave(imgs_paths[i], y[...,i])
        if(nchans == 1):
            io.imsave(imgs_paths[i], np.tile(y[...,i],(1,1,3)))

    return imgs_paths

def get_newest_exp_dirs(dir_root,res_dir,dataset_dirs):
    """ Parses directories dir_root/dataset_dirs[i]/results.
    Returns last modified experiment for all i
    """

    all_exp_dirs = []
    for d in dataset_dirs:
        exp_path = os.path.join(dir_root,d,'results')
        exp_dirs = [os.path.join(exp_path,d) for d in os.listdir(exp_path)]
        latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
        all_exp_dirs.append(latest_exp_dir)

    return all_exp_dirs

def resize_stack(X,shape):

    n_chans = X[...,0].shape[-1]
    return  np.asarray([transform.resize(X[...,i], (shape,shape,n_chans)) for i in range(X.shape[-1])]).transpose(1,2,3,0)

def resize_datasets(X,shape):

    return np.concatenate([resize_stack(X[i], shape) for i in range(len(X))], axis=3)

def sigmoid(x):

    return 0.5 * (1 + x / (1 + abs(x)))

def main(confs,
         confvgg,
         out_dir=None,
         train=True,
         pred=True,
         score=True,
         resume_model=None):

    alpha = 0.3
    n_points = 2000
    seq_type = confs[0].seq_type

    if(out_dir is None):
        now = datetime.datetime.now()
        dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(confs[0].dataOutRoot,
                               'learning_exps',
                            'learning_' + confs[0].seq_type + '_' + dateTime)

    dir_in = [c.dataOutDir for c in confs]


    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(confvgg, outfile, default_flow_style=True)

    datasets = []
    utls.setup_logging(out_dir)
    logger = logging.getLogger('learning_exp_vgg')

    logger.info('Starting learning experiment on:')
    logger.info(dir_in)
    logger.info('Gaze file: ' + str(confs[0].csvFileName_fg))
    logger.info('')

    if(not os.path.exists(os.path.join(out_dir,'datasets.npz'))):
        logger.info('Building target vectors')
        for i in range(len(dir_in)):
            with open(os.path.join(dir_in[i], 'cfg.yml'), 'r') as outfile:
                conf = yaml.load(outfile)

            logger.info('Dataset: ' + str(i+1) + '/' + str(len(dir_in)))

            dataset = learning_dataset.LearningDataset(conf)

            npz_file = np.load(os.path.join(dir_in[i],'results.npz'))

            #seeds = np.asarray(utls.get_node_list_tracklets(npz_file['list_ksp'][-1]))
            dataset.scores = npz_file['ksp_scores_mat'].astype(bool)
            datasets.append(dataset)

        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)

        logger.info('saving datasets to: ' + out_dir)
        np.savez(os.path.join(out_dir,'datasets.npz'),**{'datasets': datasets})

    dir_my_root = os.path.join(out_dir, 'my')
    dir_true_root = os.path.join(out_dir, 'true')
    n_folds = 4

    if(train):
        from nets import VGG16Basic

        logger.info('Loading datasets...')
        datasets = np.load(os.path.join(out_dir,'datasets.npz'))['datasets']
        fold_ids = np.arange(0,4)[::-1]

        for i in range(n_folds):

            logger.info('-----------------')
            pred_fold = i
            train_folds = np.asarray([fold_ids[j] for j in range(n_folds) if(fold_ids[j] != pred_fold)])

            logger.info('train_folds: ' + str(train_folds))
            logger.info('pred_folds: ' + str(pred_fold))
            logger.info('-----------------')
            logger.info('Extracting X')
            X = (resize_datasets([datasets[train_folds[j]].X_all_images for j in range(train_folds.shape[0])], confvgg.im_size)*255).astype(np.uint8)
            logger.info('Extracting y')
            y = (resize_datasets([np.tile(datasets[train_folds[j]].scores[:,:,np.newaxis,:].astype(np.uint8)*255,(1,1,3,1)) for j in range(train_folds.shape[0])], confvgg.im_size)*255).astype(np.uint8)

            # Set dirs
            dir_my = os.path.join(dir_my_root, 'fold_'+str(pred_fold))
            dir_my_train = os.path.join(dir_my, 'train')
            logger.info('Writing _my_ train images/gts to disk...')
            ims_my, gts_my = write_frames_train(X,
                                                y,
                                                dir_my_train,
                                                logger)

            logger.info('Training VGG on my segmentation...')
            vgg_my = VGG16Basic.VGG16Basic(confvgg,
                                          dir_my_train,
                                          ims_my[0])
            if(resume_model is not None):
                model_path_my = get_model_path(dir_my_train,
                                               take_min_loss=False)
                initial_epoch_my = 0

                if(len(model_path_my) == 0):
                    n_epochs_my = confvgg.n_epochs
                    model_path_my = None
                    initial_epoch_my = 0
                else:
                    n_epochs_my = confvgg.n_epochs
                    initial_epoch_my = int(os.path.split(model_path_my)[-1][6:8])
            else:
                model_path_my = None
                n_epochs_my = confvgg.n_epochs
                initial_epoch_my = 0

            dims = (confvgg.im_size,
                    confvgg.im_size,
                    3)

            vgg_my.train(confvgg,
                         dims,
                         ims_my,
                         gts_my,
                         dir_my_train,
                         n_epochs_my,
                         initial_epoch=initial_epoch_my,
                         dir_eval_clbk=dir_my,
                         resume_model=model_path_my)

            logger.info('Extracting y')
            y = (resize_datasets([np.tile(datasets[train_folds[j]].gt[:,:,np.newaxis,:].astype(np.uint8)*255,(1,1,3,1)) for j in range(train_folds.shape[0])], confvgg.im_size)*255).astype(np.uint8)

            dir_true = os.path.join(dir_true_root, 'fold_'+str(pred_fold))
            dir_true_train = os.path.join(dir_true, 'train')

            logger.info('Writing _true_ train images/gts to disk...')
            ims_true, gts_true = write_frames_train(X,
                                                    y,
                                                    dir_true_train,
                                                    logger)

            logger.info('Training U-Net on true segmentation...')
            vgg_true = VGG16Basic.VGG16Basic(confvgg,
                                            dir_true_train,
                                            ims_true[0])
            if(resume_model is not None):
                model_path_true = get_model_path(dir_true_train,
                                                 take_min_loss=False)
                initial_epoch_true = 0
                if(len(model_path_true) == 0):
                    n_epochs_true = confvgg.n_epochs
                    model_path_true = None
                    initial_epoch_true = 0
                else:
                    n_epochs_true = confvgg.n_epochs
                    initial_epoch_true = int(os.path.split(model_path_true)[-1][6:8])
            else:
                model_path_true = None
                n_epochs_true = confvgg.n_epochs
                initial_epoch_true = 0

            vgg_true.train(confvgg,
                           dims,
                           ims_true,
                           gts_true,
                           dir_true_train,
                           n_epochs_true,
                           initial_epoch=initial_epoch_true,
                           dir_eval_clbk=dir_true,
                           resume_model=model_path_true)

        else:
            logger.info('Results directory')
            logger.info(dir_my_root)
            logger.info(dir_true_root)
            logger.info('Exist. Delete and re-compute')

    logger.info('-----------------')
    logger.info('Loading datasets...')
    datasets = np.load(os.path.join(out_dir,'datasets.npz'))['datasets']
    if(pred):
        from nets import VGG16Basic
        fold_ids = np.arange(0,4)[::-1]

        for i in range(n_folds):

            pred_fold = i
            logger.info('Predicting on fold_' + str(pred_fold))

            dir_my = os.path.join(dir_my_root, 'fold_'+str(pred_fold))
            dir_true = os.path.join(dir_true_root, 'fold_'+str(pred_fold))

            #model_path_my = os.path.join(dir_my, 'train', 'weights.h5')
            model_path_my = get_model_path(os.path.join(dir_my,'train'))

            #model_path_true = os.path.join(dir_true, 'train', 'weights.h5')
            #model_path_true = get_model_path(dir_true)
            model_path_true = get_model_path(os.path.join(dir_true,'train'))
            dir_true_pred = os.path.join(dir_true, 'pred')
            dir_my_pred = os.path.join(dir_my, 'pred')
            dir_true_pred_res = os.path.join(dir_true, 'pred_res')
            dir_my_pred_res = os.path.join(dir_my, 'pred_res')

            logger.info('Will use models:')
            logger.info(model_path_my)
            logger.info(model_path_true)

            logger.info('Extracting y_my/y_true')
            X = datasets[pred_fold].X_all_images
            X = resize_stack(X, confvgg.im_size)

            logger.info('Writing _my_ pred images to disk...')
            ims_my = write_frames_pred(X, dir_my_pred, logger)

            logger.info('Writing _true_ pred images to disk...')
            ims_true = write_frames_pred(X, dir_true_pred, logger)

            vgg_true = VGG16Basic.VGG16Basic(confvgg, dir_true_pred, ims_true[0])
            vgg_my = VGG16Basic.VGG16Basic(confvgg, dir_my_pred, ims_my[0])

            # Get normalization parameters of training set
            logger.info('Extracting X (train) normalization factors')

            train_folds = np.asarray([fold_ids[j]
                                      for j in range(n_folds) if(fold_ids[j] != pred_fold)])

            im_list_train = [datasets[train_folds[j]].conf.frameFileNames for j in range(train_folds.shape[0])]
            im_list_train = [item for sublist in im_list_train for item in sublist]
            _, _, mean, std = vgg_my.preprocess_and_normalize_imgs(im_list_train)

            logger.info('Predicting on my segmentation...')
            preds_my = vgg_my.eval(confvgg,
                                    model_path_my,
                                    ims_my,
                                    mean,
                                    std)

            logger.info('Predicting on true segmentation...')
            preds_true = vgg_true.eval(confvgg,
                                        model_path_true,
                                        ims_true,
                                        mean,
                                        std)

            logger.info('Writing _my_ pred results images to disk...')
            ims_my = write_frames_pred(preds_my,
                                    dir_my_pred_res,
                                    logger,
                                    nchans=1)

            logger.info('Writing _true_ pred results images to disk...')
            ims_true = write_frames_pred(preds_true,
                                        dir_true_pred_res,
                                        logger,
                                        nchans=1)

    if(score):
        for i in range(n_folds):
            score_dict = dict()
            pred_fold = i
            logger.info('Scoring on fold_' + str(pred_fold))

            dir_my = os.path.join(dir_my_root, 'fold_'+str(pred_fold))
            dir_true = os.path.join(dir_true_root, 'fold_'+str(pred_fold))

            dir_true_pred_res = os.path.join(dir_true, 'pred_res', 'img')
            dir_my_pred_res = os.path.join(dir_my, 'pred_res', 'img')

            fnames_true_pred_res = glob.glob(os.path.join(dir_true_pred_res,
                                                          '*.png'))
            fnames_true_pred_res = sorted(fnames_true_pred_res)

            fnames_my_pred_res = glob.glob(os.path.join(dir_my_pred_res,
                                                          '*.png'))
            fnames_my_pred_res = sorted(fnames_my_pred_res)

            my_preds = np.asarray([utls.imread(f) for f in fnames_my_pred_res]).transpose(1,2,3,0)/255
            true_preds = np.asarray([utls.imread(f) for f in fnames_true_pred_res]).transpose(1,2,3,0)/255
            gts = (resize_datasets([np.tile(datasets[pred_fold].gt[:,:,np.newaxis,:],(1,1,3,1))], confvgg.im_size)>0).astype(np.uint8)

            vals_gt = gts.ravel()
            vals_my = my_preds.ravel()
            vals_true = true_preds.ravel()

            logger.info('Calculating metrics on my... ')
            all_scores = get_all_scores(vals_gt,
                                        vals_my,
                                        n_points)

            score_dict['conf'] = datasets[pred_fold].conf
            score_dict['fold'] = pred_fold
            score_dict['fpr_my'] = all_scores[0]
            score_dict['tpr_my'] = all_scores[1]
            score_dict['auc_my'] = all_scores[2]
            score_dict['pr_my'] = all_scores[3]
            score_dict['rc_my'] = all_scores[4]
            score_dict['f1_my'] = all_scores[5]
            score_dict['thr_my'] = all_scores[6]

            logger.info('Calculating metrics on true... ')
            all_scores = get_all_scores(vals_gt,
                                        vals_true,
                                        n_points)

            score_dict['fpr_true'] = all_scores[0]
            score_dict['tpr_true'] = all_scores[1]
            score_dict['auc_true'] = all_scores[2]
            score_dict['pr_true'] = all_scores[3]
            score_dict['rc_true'] = all_scores[4]
            score_dict['f1_true'] = all_scores[5]
            score_dict['thr_true'] = all_scores[6]

            score_dict['my_preds'] = my_preds
            score_dict['true_preds'] = true_preds

            logger.info('Saving results on fold: ' + str(pred_fold))
            file_out = os.path.join(out_dir,
                                    'scores_' + str(pred_fold) + '.npz')
            np.savez(file_out, **score_dict)


def get_model_path(path_, ext='model-*.hdf5', default='weights.h5', take_min_loss=True):

    files = glob.glob(os.path.join(path_, ext))
    print('files:')
    print(files)

    if(len(files) < 1):
        #return os.path.join(path_, default)
        return []
    else:
        files = [os.path.split(f)[1] for f in files]
        if(take_min_loss):
            files = sorted(files, key=lambda name: float(name[9:13]))
            return os.path.join(path_, files[0])
        else:
            files = sorted(files)
            return os.path.join(path_, files[-1])
