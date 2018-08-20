import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from skimage.draw import circle
from skimage.color import color_dict
from skimage import color
import gazeCsv as gaze
import my_utils as utls
from operator import itemgetter
import pandas as pd
import datetime
import os

#path_ = '/home/krakapwa/otlshare/medical-labeling/Dataset00/results/2017-11-27_11-53-28_coverage'
#
#npzfile = np.load(os.path.join(path_, 'info.npz'))
#df = pd.DataFrame(data=[npzfile['coverage']], columns=['coverage'])
#df.to_csv(os.path.join(path_, 'coverage.csv'))

def calc_coverage(pos_labels,
                  labels_clicked):

    # Remove duplicates
    labels_clicked = [l[-1] for l in labels_clicked]
    labels_clicked = list(set(labels_clicked))

    inters = list(set(labels_clicked) & set(pos_labels))

    return len(inters)/len(pos_labels)

def make_frame(idx,
               ref_idx,
               frames,
               labels,
               label_contours,
               gts,
               labels_clicked,
               mouse_clicks,
               mode='right',
               highlight_idx=None):

    if(mode == 'right'):
        the_idx = idx
    else:
        the_idx = ref_idx

    shape = labels[:,:,0].shape

    colors_ = [color_dict['red'],
               color_dict['green'],
               color_dict['blue'],
               color_dict['magenta'],
               color_dict['white']]

    mask = np.zeros(shape, dtype=np.uint8)

    img = utls.imread(frames[the_idx])
    if(img.shape[2] > 3): img = img[:,:,0:3]
    if(label_contours is not None):
        idx_contours = np.where(label_contours[the_idx,:,:])
        img[idx_contours[0],idx_contours[1],:] = (255,255,255)
        l_ = (utls.imread(gts[the_idx])>0)[...,0].astype(np.uint8)
        l_idx = np.where(l_)
    mask[l_idx[0], l_idx[1]] = 1

    #Draw set of tracked labels
    if(mode == 'right'):
        labels_to_draw = [l[-1] for l in labels_clicked if(l[1] == idx)]
    else:
        print('labels_clicked: ' + str(labels_clicked))
        print('idx: ' + str(idx))
        labels_to_draw = [l[-1] for l in labels_clicked if(l[0] == ref_idx)]
        print('labels_to_draw: ' + str(labels_to_draw))


    if(len(labels_to_draw) > 0):

        #print(self.curr_idx)
        for i in range(len(labels_to_draw)):
            this_label = labels_to_draw[i]
            mask_tmp = labels[:,:,the_idx] == this_label

            l_ = mask_tmp.astype(np.uint8)
            l_idx = np.where(l_)
            mask[l_idx[0], l_idx[1]] = 2

    if(mode == 'right'):
        mouse_clicks_to_draw = [r for r in mouse_clicks if(r[1] == idx)]
        #print(mouse_clicks_to_draw)
    else:
        mouse_clicks_to_draw = [r for r in mouse_clicks if(r[0] == ref_idx)]
        #mouse_clicks_to_draw = mouse_clicks

    if(len(mouse_clicks_to_draw) > 0):
        for i in range(len(mouse_clicks_to_draw)):
            x = mouse_clicks_to_draw[i][-2]
            y = mouse_clicks_to_draw[i][-1]
            g_i, g_j = gaze.gazeCoord2Pixel(x,
                                        y,
                                        mask.shape[1],
                                        mask.shape[0])
            rr, cc = circle(g_i, g_j, radius=7)
            mask[rr, cc] = 3

    #mask = np.zeros(shape, dtype=np.uint8)

    #if(highlight_idx is not None):
    #    mouse_clicks_to_draw = [r for r in mouse_clicks if(r[0] in highlight_idx)]
    #    for i in range(len(mouse_clicks_to_draw)):
    #        x = mouse_clicks_to_draw[i][3]
    #        y = mouse_clicks_to_draw[i][4]
    #        g_i, g_j = gaze.gazeCoord2Pixel(x,
    #                                    y,
    #                                    mask.shape[1],
    #                                    mask.shape[0])
    #        rr, cc = circle(g_i, g_j, radius=7)
    #        mask[rr, cc] = 4

    img = color.label2rgb(mask,
                          img,
                          alpha=.2,
                          bg_label=0,
                          colors=colors_)

    img = np.rot90(img)[::-1,:,:]

    return img

def sort_list(list_, dim):
    return sorted(list_, key=itemgetter(dim))

def update_mouse_list(idx,
                      ref_idx,
                      mouse_list,
                      x,
                      y,
                      operation='overwrite',
                      mode='right'):

    if(mouse_list is not None):

        if(mode == 'right'):
            if(operation == 'overwrite'):
                frames_already_there = [m[1] for m in mouse_list]
                #Already exist?
                if(idx in frames_already_there):
                    #Removes clicked label if it already exist
                    #print('frame already there... removing it')
                    mouse_list = [mouse_list[i] for i in range(len(mouse_list)) if(mouse_list[i][1] != idx)]
                #Add it
                mouse_list.append([ref_idx,
                                   idx,
                                   0,
                                   1,
                                   x,
                                   y])
            elif(operation == 'remove'):
                    mouse_list = [m for m in mouse_list if(m[1] != idx)]
            elif(operation == 'add'):
                mouse_list.append([ref_idx,
                                   idx,
                                   0,
                                   1,
                                   x,
                                   y])
        elif(mode == 'ref'):
            if(operation == 'overwrite'):
                frames_already_there = [m[0] for m in mouse_list]
                #Already exist?
                if(idx in frames_already_there):
                    #Removes clicked label if it already exist
                    #print('frame already there... removing it')
                    mouse_list = [mouse_list[i] for i in range(len(mouse_list)) if(mouse_list[i][0] != idx)]
                #Add it
                mouse_list.append([ref_idx,
                                   idx,
                                   0,
                                   1,
                                   x,
                                   y])
            elif(operation == 'remove'):
                    mouse_list = [m for m in mouse_list if(m[1] != idx)]
            elif(operation == 'add'):
                mouse_list.append([ref_idx,
                                   idx,
                                   0,
                                   1,
                                   x,
                                   y])

    return sort_list(mouse_list, 1)

def update_label_list(idx,
                      ref_idx,
                      labels,
                      labels_list,
                      x,
                      y,
                      operation='overwrite',
                      mode='right'):

    if(mode == 'right'):
        the_idx = idx
    else:
        the_idx = ref_idx
    this_label = [ref_idx, idx, labels[int(y), int(x), the_idx]]
    if(labels_list is not None):

        if(mode == 'right'):
            if(operation == 'overwrite'):
                #Already exist?
                frames_already_there = [l[1] for l in labels_list]
                #print(this_label)
                #print(frames_already_there)
                if(this_label[1] in frames_already_there):
                    #Removes clicked label if it already exist
                    #print('frame already tracked... removing it')
                    labels_list = [labels_list[i] for i in range(len(labels_list)) if(labels_list[i][1] != this_label[1])]
                #Add it
                labels_list.append(this_label)
            elif(operation == 'add'):
                labels_list.append(this_label)
            elif(operation == 'remove'):
                labels_list = [l for l in labels_list if(l[1] != idx)]
        elif(mode == 'ref'):
            if(operation == 'overwrite'):
                #Already exist?
                frames_already_there = [l[0] for l in labels_list]
                #print(this_label)
                #print(frames_already_there)
                if(this_label[0] in frames_already_there):
                    #Removes clicked label if it already exist
                    #print('frame already tracked... removing it')
                    labels_list = [labels_list[i] for i in range(len(labels_list)) if(labels_list[i][0] != this_label[0])]
                #Add it
                labels_list.append(this_label)
            elif(operation == 'add'):
                labels_list.append(this_label)
            elif(operation == 'remove'):
                labels_list = [l for l in labels_list if(l[1] != idx)]


    return sort_list(labels_list, 0)

def get_pos_sps(labels, gt, thr=0.5):

    pos_labels = []
    for l in np.unique(labels):
        mask = labels == l
        inters = mask*gt
        ratio = np.sum(inters)/np.sum(mask)
        if(ratio > thr):
            pos_labels.append(l)

    return np.asarray(pos_labels)

def isKeyPressed(gaze,frameNum):
    return bool(gaze[frameNum,2])

def save_all(save_dir,
             dataset_dir,
             ref_frame,
             coverage,
             labels_list,
             mouse_list,
             labels_list_ref,
             mouse_list_ref):

    now = datetime.datetime.now()
    dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
    dir_out = os.path.join(save_dir, dateTime + '_coverage')
    if(not os.path.exists(dir_out)):
        os.makedirs(dir_out)

    labels_f = os.path.join(dir_out,'labels.csv')
    labels_ref_f = os.path.join(dir_out,'labels_ref.csv')
    mouse_f = os.path.join(dir_out,'mouse.csv')
    mouse_ref_f = os.path.join(dir_out,'mouse_ref.csv')
    info_f = os.path.join(dir_out,'info')
    coverage_f = os.path.join(dir_out,'coverage.csv')

    labels_list = [[l[1], l[2]] for l in labels_list]
    df_labels = pd.DataFrame(data=labels_list,
                             columns=['frame', 'sp_label'])
    df_labels.to_csv(labels_f)

    labels_list_ref = [[l[0], l[2]] for l in labels_list_ref]
    df_labels_ref = pd.DataFrame(data=labels_list_ref,
                             columns=['frame', 'sp_label'])
    df_labels_ref.to_csv(labels_ref_f)

    mouse_list = [m[1:] for m in mouse_list]
    print(mouse_list)
    df_mouse = pd.DataFrame(data=mouse_list,
                             columns=['frame', 'time','visible','x','y'])
    df_mouse.to_csv(mouse_f)

    mouse_list_ref = [[m[0], m[2], m[3], m[4], m[5]] for m in mouse_list_ref]
    print(mouse_list_ref)
    df_mouse_ref = pd.DataFrame(data=mouse_list_ref,
                             columns=['frame', 'time','visible','x','y'])
    df_mouse_ref.to_csv(mouse_ref_f)

    df_coverage = pd.DataFrame(data=[coverage],
                               columns=['coverage'])
    df_coverage.to_csv(coverage_f)

    info = dict()
    info['dataset_dir'] = dataset_dir
    info['ref_frame'] = ref_frame
    info['coverage'] = coverage
    np.savez(info_f, **info)

    print('Saved data to: ' + dir_out)

    #df = pd.DataFrame(data=data, index=I)
class SliderWithText(QtGui.QWidget):
    def __init__(self, *var, **kw):
        QtGui.QWidget.__init__(self, *var, **kw)
        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.label = QtGui.QLabel(self)
        # QVBoxLayout the label above; could use QHBoxLayout for
        # side-by-side
        layout = QtGui.QVBoxLayout()
        #layout.setMargin(0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        #self.style_off = '<span style=" font-size:8pt; font-weight:600; color:#aa0000;">'
        #self.style_on = '<span style=" font-size:8pt; font-weight:600; color:#00aa00;">'
    def set_text(self,string,mode):
        if(mode): #Display on style
            self.label.setStyleSheet("font-weight: bold; color: green");
            #self.label.setText(self.style_on + string + '</span>')
            self.label.setText(string)
        else: #Display off style
            self.label.setStyleSheet("font-weight: bold; color: red");
            #self.label.setText(self.style_off + string + '</span>')
            self.label.setText(string)
