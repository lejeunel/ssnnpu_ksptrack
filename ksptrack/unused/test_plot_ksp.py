import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import my_utils as utls
from sklearn.metrics import f1_score,confusion_matrix,auc
from sklearn.metrics import confusion_matrix
import progressbar
import gazeCsv as gaze

dir_in_root = '/home/laurent.lejeune/medical-labeling'

#dir_result = 'Dataset12/results/2017-06-14_14-47-27_exp/'
dir_results = 'Dataset2/results/2017-06-14_22-26-11_exp'
pm_scores = []
ksp_scores = []
iters = np.arange(10)
for i in iters:
    this_npz_file = np.load(os.path.join(dir_in_root,dir_results,'pm_scores_iter_' + str(i) + '.npz'))
    pm_scores.append(this_npz_file['pm_scores_fg'])
    ksp_scores.append(this_npz_file['ksp_scores_mat'])

with open(os.path.join(dir_in_root,dir_results, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)

#Plot n_pix ksp per iteration
n_pix = []
for i in iters:
    n_pix.append(np.sum(ksp_scores[i]))

plt.plot(iters,n_pix,'bo-')
plt.xlabel('iterations')
plt.ylabel('num. of pixels of ksp')
plt.title(dir_results)
plt.savefig(os.path.join(dir_in_root,dir_results,'pix_per_iter.png'),dpi=200)

the_gaze = conf.myGaze_fg

max_n_frames = 50
f_ind = np.linspace(0,len(conf.frameFileNames)-1,np.min((max_n_frames,len(conf.frameFileNames)))).astype(int)

for i in iters:
    with progressbar.ProgressBar(maxval=iters.shape[0]) as bar:
        for i in iters:
            bar.update(i)
            this_npz_file = np.load(os.path.join(dir_in_root,dir_results,'pm_scores_iter_' + str(i) + '.npz'))
            pm_scores = this_npz_file['pm_scores_fg']
            ksp_scores = this_npz_file['ksp_scores_mat']
            for f in f_ind:
                plt.subplot(131)
                plt.imshow(ksp_scores[f,...])
                plt.title('KSP scores. Iter.: ' + str(i))
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(pm_scores[...,f])
                plt.title('PM before Iter.: ' + str(i))
                plt.axis('off')
                this_im = utls.imread(conf.frameFileNames[f])
                this_im =  gaze.drawGazePoint(the_gaze,f,this_im,radius=12)
                plt.subplot(133)
                plt.imshow(this_im)
                plt.suptitle('frame ' + str(f+1) + '/' + str(len(conf.frameFileNames)))
                this_fname = os.path.split(conf.frameFileNames[f])[1]
                #plt.show()
                this_subdir = 'all_frames_iter_' + str(i)
                this_full_path = os.path.join(dir_in_root,dir_results,this_subdir,'pm_ksp_iter_' + str(i) + '_' + this_fname)
                if(not os.path.exists(os.path.join(dir_in_root,dir_results,this_subdir))):
                    os.mkdir(os.path.join(dir_in_root,dir_results,this_subdir))
                plt.savefig(os.path.join(this_full_path),dpi=100)
#plt.show()

gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
gtFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.frameDigits, conf.truth_dir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

gt_positives = utls.getPositives(gtFileNames)

conf_mats_ksp = []
conf_mats = []
f_scores = []
f_scores_ksp = []
tpr_fpr = []
thr = np.linspace(0.1,0.9,15)
for i in range(iters.shape[0]):
#for i in range(1):
    conf_mats.append([])
    f_scores.append([])
    this_conf_mat_ksp = confusion_matrix(gt_positives.astype(int).ravel(),(npz_file[iters[i]]['ksp_scores_mat']).astype(int).ravel())
    conf_mats_ksp.append(this_conf_mat_ksp)
    this_f1_ksp = utls.conf_mat_to_f1(this_conf_mat_ksp)
    f_scores_ksp.append(this_f1_ksp)
    print('iter: ' + str(i))
    for t in thr:
        print('    thr: ' + str(t))
        this_conf_mat = confusion_matrix(gt_positives.astype(int).ravel(),
                                         (npz_file[iters[i]]['pm_scores_fg'] > t).astype(int).ravel())
        #test = utls.conf_mat_to_tpr_fpr(this_conf_mat)
        conf_mats[i].append(this_conf_mat)

        f_scores[i].append(utls.conf_mat_to_f1(this_conf_mat))
    this_iter_tpr_fpr = [utls.conf_mat_to_tpr_fpr(conf_mats[i][j]) for j in range(len(conf_mats[i]))]
    tpr_fpr.append(this_iter_tpr_fpr)

tpr_fpr_arr = np.asarray(tpr_fpr)
all_iter_auc = auc(tpr_fpr_arr[:,0],tpr_fpr_arr[:,1])

last_auc = auc(last_tpr_fpr[:,1],last_tpr_fpr[:,0])
last_tpr_fpr = np.asarray(tpr_fpr[-1])
plt.plot(last_tpr_fpr[:,1],last_tpr_fpr[:,0]); plt.show()


#my_legend = []
#max_f = []
#for i in range(iters.shape[0]):
#    print('iter: ' + str(i))
#    plt.plot(false_pos_rate_pm[i],true_pos_rate_pm[i])
#    max_f.append(utls.get_max_f_score(gt_positives,
#                                      npz_file[iters[i]]['pm_scores_fg'],
#                                      thr=np.linspace(0.1,1,15)))
#    this_legend = 'iter: ' + str(i) + '. fmax: ' + str(max_f[-1])
#    #this_conf_mat = confusion_matrix(gt_positives.ravel(), gc_maps.ravel())
#    my_legend.append(this_legend)
#
#plt.legend(my_legend)
#plt.show()
