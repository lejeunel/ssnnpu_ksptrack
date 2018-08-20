import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import os.path
import sys
import gazeCsv as gaze
import my_utils as utls
import glob
import viewer_utls as vutls


class Window(QtGui.QMainWindow):
    def __init__(self,
                 frame_dir,
                 dataset_dir,
                 save_dir,
                 frames,
                 gts,
                 ref_frame,
                 pos_sps,
                 dir_clicks,
                 label_contours=None,
                 labels=None,
                 parent=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle('Mes jolies images')
        self.window = pg.ImageView(view=pg.PlotItem())
        self.slider_text = vutls.SliderWithText()
        self.label_contours = label_contours
        self.labels = labels

        self.setCentralWidget(self.window)
        self.slider_text.slider.setMinimum(0)
        self.slider_text.slider.setMaximum(len(frames))
        self.slider_text.slider.setTickInterval(1)
        self.slider_text.slider.setSingleStep(1)

        self.label = QtGui.QLabel(self)
        #self.label.setGeometry(160, 40, 80, 30)

        self.savebutton = QtGui.QPushButton("Save")

        self.frames = frames
        self.ref_frame = ref_frame
        self.frame_dir = frame_dir
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.gts = gts
        self.pos_sps = pos_sps
        self.dir_clicks = dir_clicks

        self.curr_img = utls.imread(os.path.join(self.frame_dir,self.frames[0]))
        self.mouse_list_ref = []
        self.mouse_list = []
        self.labels_list_ref = []
        self.labels_list = []

        # add the menubar with the method createMenuBar()
        self.createMenuBar()
        # add the dock widget with the method createDockWidget()
        self.createDockWidget()
        #
        # first set the default value to a
        self.curr_idx = 0
        self.max_idx = len(self.frames)
        self.slider_text.slider.setValue(self.curr_idx)
        self.drawFrame(self.curr_idx)

        self.sliderValueChanged(0)

        # Connect widgets to functions
        self.slider_text.slider.valueChanged.connect(self.sliderValueChanged)
        self.savebutton.clicked.connect(self.savebuttonClick)

    def is_already_clicked(self,idx):
        if(len(self.mouse_list) > 0):
            if(self.curr_idx in np.asarray(self.mouse_list)[:,0]):
                return True
            else:
                return False
        else:
            return False


    def keyPressEvent(self,event):

        if event.key() == QtCore.Qt.Key_F:
            if(self.curr_idx < len(self.frames)-1):
                self.sliderValueChanged(self.curr_idx+1)
        if event.key() == QtCore.Qt.Key_D:
            if(self.curr_idx > 0):
                self.sliderValueChanged(self.curr_idx-1)

    def on_click(self, event):


        x = event.pos().x()
        y = event.pos().y()
        im_shape = self.curr_img.shape
        x_norm, y_norm = gaze.gazePix2Norm(x,y,im_shape[1], im_shape[0])

        if event.button() == QtCore.Qt.LeftButton:
        # Determine which image was clicked
            if(x > im_shape[1]): # Right frame
                x_norm -= 1.0
                x -= im_shape[1]
                self.mouse_list = vutls.update_mouse_list(self.curr_idx,
                                                          self.ref_frame,
                                                            self.mouse_list,
                                                            x_norm,
                                                            y_norm,
                                                            operation='overwrite',
                                                          mode='right')
                self.labels_list = vutls.update_label_list(self.curr_idx,
                                                           self.ref_frame,
                                                        self.labels,
                                                        self.labels_list,
                                                        x,
                                                        y,
                                                           operation='overwrite',
                                                           mode='right')
            else: # Ref frame
                self.mouse_list_ref = vutls.update_mouse_list(self.curr_idx,
                                                              self.ref_frame,
                                                                self.mouse_list_ref,
                                                                x_norm,
                                                                y_norm,
                                                                operation='add',
                                                              mode='ref')
                self.labels_list_ref = vutls.update_label_list(self.curr_idx,
                                                               self.ref_frame,
                                                                self.labels,
                                                                self.labels_list_ref,
                                                                x,
                                                                y,
                                                                operation='add',
                                                               mode='ref')
                print('self.labels_list_ref')
                print(self.labels_list_ref)
        else: # Right button click
            if(x > im_shape[1]): # Right frame
                x_norm -= 1.0
                x -= im_shape[1]
                self.mouse_list = vutls.update_mouse_list(self.curr_idx,
                                                          self.ref_frame,
                                                            self.mouse_list,
                                                            x_norm,
                                                            y_norm,
                                                            operation='remove',
                                                          mode='right')
                self.labels_list = vutls.update_label_list(self.curr_idx,
                                                           self.ref_frame,
                                                        self.labels,
                                                        self.labels_list,
                                                        x,
                                                        y,
                                                           operation='remove',
                                                           mode='right')

                self.mouse_list_ref = vutls.update_mouse_list(self.curr_idx,
                                                              self.ref_frame,
                                                                self.mouse_list_ref,
                                                                x_norm,
                                                                y_norm,
                                                                operation='remove',
                                                              mode='ref')
                print('removing on labels_list_ref')
                print('self.labels_list_ref')
                print(self.labels_list_ref)
                self.labels_list_ref = vutls.update_label_list(self.curr_idx,
                                                               self.ref_frame,
                                                                self.labels,
                                                                self.labels_list_ref,
                                                                x,
                                                                y,
                                                                operation='remove',
                                                               mode='ref')
                print('self.labels_list_ref')
                print(self.labels_list_ref)

        self.update_text()

        # Update frame
        self.drawFrame(self.curr_idx)

    def sliderValueChanged(self, int_value):

        self.curr_idx = int_value
        self.update_text()


        self.drawFrame(self.curr_idx)

    def savebuttonClick(self):

        vutls.save_all(self.save_dir,
                       self.dataset_dir,
                       self.ref_frame,
                       vutls.calc_coverage(self.pos_sps, self.labels_list_ref),
                       self.labels_list,
                       self.mouse_list,
                       self.labels_list_ref,
                       self.mouse_list_ref)


    def update_text(self):

        string_to_disp = ''
        string_to_disp += "Frame " + str(self.curr_idx+1) + "/" + \
                          str(len(self.frames))
        #print(string_to_disp)
        string_frame = "Frame " + \
                       str(self.curr_idx + 1) + \
                       "/" + str(len(self.frames))

        #print('Coverage [%]: ' + )
        coverage_perc = np.round_(100*vutls.calc_coverage(self.pos_sps,
                                                          self.labels_list_ref),
                                  decimals=2)
        string_coverage = 'Coverage: ' + \
                          str(coverage_perc) + \
                          "%"

        self.slider_text.set_text(string_frame + ' ' + string_coverage,
                                  self.is_already_clicked(self.curr_idx))

    def createMenuBar(self):
        # file menu actions
        exit_action = QtGui.QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        # create an instance of menu bar
        menubar = self.menuBar()
        # add file menu and file menu actions
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(exit_action)

    def createDockWidget(self):
        my_dock_widget = QtGui.QDockWidget()
        my_dock_widget.setObjectName('Control Panel')
        my_dock_widget.setAllowedAreas(QtCore.Qt.TopDockWidgetArea | QtCore.Qt.BottomDockWidgetArea)
        # create a widget to house user control widgets like sliders
        my_house_widget = QtGui.QWidget()
        # every widget should have a layout, right?
        my_house_layout = QtGui.QVBoxLayout()
        # add the slider initialized in __init__() to the layout
        my_house_layout.addWidget(self.slider_text)


        my_house_layout.addWidget(self.savebutton)
        # apply the 'house' layout to the 'house' widget
        my_house_widget.setLayout(my_house_layout)
        # set the house widget 'inside' the dock widget
        my_dock_widget.setWidget(my_house_widget)
        # now add the dock widget to the main window
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, my_dock_widget)

    def drawFrame(self,idx):

        img_ref = vutls.make_frame(idx,
                                   self.ref_frame,
                                   self.frames,
                                   self.labels,
                                   self.label_contours,
                                   self.gts,
                                   self.labels_list_ref,
                                   self.mouse_list_ref,
                                   mode='ref',
                                   highlight_idx=[idx])

        img_right = vutls.make_frame(idx,
                                     self.ref_frame,
                                     self.frames,
                                     self.labels,
                                     self.label_contours,
                                     self.gts,
                                     self.labels_list,
                                     self.mouse_list,
                                     mode='right')

        img = np.concatenate((img_ref, img_right), axis=0)

        self.window.setImage(img)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            pos = event.pos()
            #print(pos)
            #self.edit.setText('x: %d, y: %d' % (pos.x(), pos.y()))
        return QtGui.QMainWindow.eventFilter(self, source, event)

def main():
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('Mes jolies images')

    #Get frame file names
    root_dir = '/home/laurent.lejeune/medical-labeling'
    dataset_dir = 'Dataset03'
    save_dir  = os.path.join(root_dir, dataset_dir, 'results')

    # Used to compute coverage
    ref_frame = 0

    frame_dir = os.path.join(root_dir,
                             dataset_dir,
                             'input-frames')

    gt_dir = os.path.join(root_dir,
                          dataset_dir,
                          'ground_truth-frames')

    dir_clicks = os.path.join('.')

    sorted_frames = sorted(glob.glob(os.path.join(frame_dir,'*.png')))

    sorted_gt = sorted(glob.glob(os.path.join(gt_dir,'*.png')))

    gts = [utls.imread(sorted_gt[i]) for i in range(len(sorted_gt))]
    gts = [(g[...,0]>0)[...,np.newaxis] for g in gts]
    gts = np.concatenate(gts, axis=2)

    #Tweezer
    label_contour_path = 'precomp_descriptors/'
    label_contour_fname = 'sp_labels_tsp_contours'

    npzfile = np.load(os.path.join(root_dir,
                                   dataset_dir,
                                   label_contour_path,
                                   label_contour_fname)+'.npz',
                      fix_imports=True,
                      encoding='bytes')

    labelContourMask = npzfile['labelContourMask']

    labels = np.load(os.path.join(root_dir,
                                  dataset_dir,
                                  'input-frames',
                                  'sp_labels.npz'))['sp_labels']

    pos_sps = vutls.get_pos_sps(labels[...,ref_frame],
                          gts[...,ref_frame])

    my_window = Window(frame_dir,
                       dataset_dir,
                       save_dir,
                       sorted_frames,
                       sorted_gt,
                       ref_frame,
                       pos_sps,
                       dir_clicks,
                       labelContourMask,
                       labels)

    #Make event on click
    my_window.window.getImageItem().mousePressEvent = my_window.on_click

    #Mouse move coordinates
    #app.installEventFilter(my_window)

    my_window.show()
    app.exec_()

if __name__ == '__main__':
    main()
