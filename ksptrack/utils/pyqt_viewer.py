import tqdm
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from skimage.draw import (circle, set_color)
from skimage import color, io
from skimage.io import imsave
from skimage import (transform, segmentation)
import csv
import sys, getopt
import os.path
import natsort
import glob
import configargparse
import pandas as pd
from natsort import index_natsorted, order_by_index


def get_images(path, extension=('jpg', 'png')):
    """ Generates list of (sorted) images
    Returns List of paths to images
    """

    fnames = []

    if (isinstance(extension, str)):
        extension = [extension]

    for ext in extension:
        fnames += glob.glob(os.path.join(path, '*' + ext))

    fnames = sorted(fnames)

    return fnames


def imread(fname):

    img = io.imread(fname)

    if (img.dtype == np.uint16):
        img = ((img / (2**16)) * (2**8)).astype(np.uint8)

    if (len(img.shape) > 2):
        nchans = img.shape[2]
        if (nchans > 3):
            return img[:, :, 0:3]
        else:
            return img
    else:
        return color.gray2rgb(img)


def coord2Pixel(x, y, width, height, round_to_int=True):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = x * (width - 1)
    i = y * (height - 1)

    if (round_to_int):
        j = int(np.round(j, 0))
        i = int(np.round(i, 0))

    return i, j


def pix2Norm(j, i, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    x = j / (width - 1)
    y = i / (height - 1)

    return x, y


def draw2DPoint(mouse_clicks, frame_num, img, radius=2, color=(0, 255, 0)):
    height, width, _ = img.shape

    if(not len(mouse_clicks)):
        return img
    
    clicks = [c for c in mouse_clicks if(c['frame'] == frame_num)]

    for click in clicks:
        i, j = coord2Pixel(click['x'], click['y'], width, height)
        rr, cc = circle(i, j, radius, shape=(height, width))
        img[rr, cc, 0] = color[0]
        img[rr, cc, 1] = color[1]
        img[rr, cc, 2] = color[2]

    return img


class SliderWithText(QtGui.QWidget):
    def __init__(self, *var, **kw):
        QtGui.QWidget.__init__(self, *var, **kw)
        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.label_frame_num = QtGui.QLabel(self)
        self.label_keybinding_info = QtGui.QLabel(self)
        # QVBoxLayout the label above; could use QHBoxLayout for
        # side-by-side
        layout = QtGui.QVBoxLayout()
        layout.setSpacing(2)
        layout.addWidget(self.label_keybinding_info)
        layout.addWidget(self.label_frame_num)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def set_text_frame_num(self, string):

        self.label_frame_num.setText(string)

    def set_text_keybinding_info(self, string):
        print('set keybinding info!')
        self.label_keybinding_info.setText(string)


class Window(QtGui.QMainWindow):
    def __init__(self,
                 frames,
                 truths,
                 out_csv,
                 parent=None,
                 remove_duplicate_clicks=True):

        super(Window, self).__init__(parent)
        self.setWindowTitle('kikoooo')
        self.plot_item = pg.PlotItem()
        self.window = pg.ImageView(view=self.plot_item)
        self.slider_text = SliderWithText()
        self.remove_duplicate_clicks = remove_duplicate_clicks

        self.setCentralWidget(self.window)
        self.slider_text.slider.setMinimum(0)
        self.slider_text.slider.setMaximum(len(frames))
        self.slider_text.slider.setTickInterval(1)
        self.slider_text.slider.setSingleStep(1)

        self.label_frame_num = QtGui.QLabel(self)

        print('Caching frames...')
        pbar = tqdm.tqdm(total=len(frames))
        self.frames = []
        for i, f in enumerate(frames):
            f_ = imread(f)
            if (truths is not None):
                contour = segmentation.find_boundaries(imread(truths[i]),
                                                       mode='thick')
                idx_contour = np.where(contour)
                f_[idx_contour[0], idx_contour[1], :] = (255, 0, 0)
            self.frames.append(f_)
            pbar.update(1)
        pbar.close()

        self.mouse_clicks = []

        if(os.path.exists(out_csv)):
            print('found csv file {}'.format(out_csv))
            print('will load it')
            self.read_click_csv(out_csv)
        self.out_csv = out_csv

        path = os.path.split(self.out_csv)[0]
        if(not os.path.exists(path)):
            print('{} does not exist. creating.'.format(path))
            os.makedirs(path)

        print('done.')
        self.curr_img = self.frames[0]

        # add the menubar with the method createMenuBar()
        self.createMenuBar()
        # add the dock widget with the method createDockWidget()
        self.createDockWidget()

        # first set the default value to a
        self.curr_idx = 0
        self.max_idx = len(self.frames)
        self.slider_text.slider.setValue(self.curr_idx)
        self.drawFrame(self.curr_idx)

        self.updateFrame(0)
        string = self.make_frame_num_text()

        self.slider_text.set_text_frame_num(string)
        self.slider_text.slider.valueChanged.connect(self.updateFrame)

        #key bindings
        self.key_next = QtCore.Qt.Key_A
        self.key_prev = QtCore.Qt.Key_S
        self.key_quit = QtCore.Qt.Key_Q
        self.key_delete = QtCore.Qt.Key_D

        str_keybinds = 'Forward: {}, Backward: {}, Delete {}, Save and Quit: {}'.format(
            'A', 'S', 'D', 'Q')
        self.slider_text.set_text_keybinding_info(str_keybinds)

    def is_already_clicked(self, idx):
        if (len(self.mouse_clicks) > 0):
            if (self.curr_idx in np.asarray(self.mouse_clicks)[:, 0]):
                return True
            else:
                return False
        else:
            return False

    def make_frame_num_text(self):

        return 'Frame {} / {}'.format(self.curr_idx + 1, len(self.frames))

    def keyPressEvent(self, event):

        if event.key() == self.key_next:
            if (self.curr_idx < len(self.frames) - 1):
                self.updateFrame(self.curr_idx + 1)
        if event.key() == self.key_prev:
            if (self.curr_idx > 0):
                self.updateFrame(self.curr_idx - 1)
        if event.key() == self.key_quit:
            self.write_click_csv(self.frames)
            self.close()
        if event.key() == self.key_delete:
            self.mouse_clicks = [
                c for c in self.mouse_clicks if (c['frame'] != self.curr_idx)
            ]

        self.updateFrame(self.curr_idx)

    def on_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        im_shape = self.curr_img.shape

        x_norm, y_norm = pix2Norm(x, y, im_shape[1], im_shape[0])

        self.mouse_clicks.append({
            'frame': self.curr_idx,
            'x': x_norm,
            'y': y_norm
        })

        self.updateFrame(self.curr_idx)

    def read_click_csv(self, out_csv):
        in_ = pd.read_csv(out_csv, header=4, sep=';')
        self.mouse_clicks = [{
            'frame': r['frame'],
            'x': r['x'],
            'y': r['y']
        } for _, r in in_.iterrows()]

    def write_click_csv(self, n_frames):

        out = pd.DataFrame(self.mouse_clicks)
        out = out.reindex(index=order_by_index(
            out.index, index_natsorted(out['frame'], reverse=False)))

        out = out.assign(time=[0]*out.shape[0])
        out = out.assign(visible=[1]*out.shape[0])

        cols = ['frame', 'time', 'visible', 'x', 'y']
        out = out[cols]

        # reindex or change the order of columns
        str_ = out.to_csv(sep=';', index=False)
        h, w = self.frames[0].shape[:2]
        header = 'VideoWidth:{}\nVideoHeight:{}\nDisplayWidth:0\nDisplayHeight:0\n'.format(w, h)
        str_ = header + str_
        text_file = open(self.out_csv, 'w')
        n = text_file.write(str_)
        text_file.close()
        print('written {}'.format(self.out_csv))

    def updateFrame(self, int_value):

        self.curr_idx = int_value

        string_to_disp = self.make_frame_num_text()

        self.slider_text.set_text_frame_num(string_to_disp)
        self.slider_text.slider.setValue(self.curr_idx)
        self.drawFrame(self.curr_idx)

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
        my_dock_widget.setAllowedAreas(QtCore.Qt.TopDockWidgetArea
                                       | QtCore.Qt.BottomDockWidgetArea)
        # create a widget to house user control widgets like sliders
        my_house_widget = QtGui.QWidget()
        # every widget should have a layout, right?
        my_house_layout = QtGui.QVBoxLayout()
        # add the slider initialized in __init__() to the layout
        my_house_layout.addWidget(self.slider_text)
        # apply the 'house' layout to the 'house' widget
        my_house_widget.setLayout(my_house_layout)
        # set the house widget 'inside' the dock widget
        my_dock_widget.setWidget(my_house_widget)
        # now add the dock widget to the main window
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, my_dock_widget)

    def drawFrame(self, idx):
        img = self.frames[idx].copy()

        img = draw2DPoint(self.mouse_clicks, idx, img, radius=7)

        img = np.rot90(img)[::-1, :, :]
        self.window.setImage(img)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            pos = event.pos()
            print(pos)
            #self.edit.setText('x: %d, y: %d' % (pos.x(), pos.y()))
        return QtGui.QMainWindow.eventFilter(self, source, event)


def main(frame_dir, out_csv, truth_dir):
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('Image Viewer')

    sorted_frames = get_images(frame_dir)

    sorted_truths = None
    if (truth_dir is not None):
        sorted_truths = get_images(truth_dir)

    my_window = Window(sorted_frames,
                       sorted_truths,
                       out_csv,
                       remove_duplicate_clicks=False)

    #Make event on click
    my_window.window.getImageItem().mousePressEvent = my_window.on_click

    #Mouse move coordinates
    #app.installEventFilter(my_window)

    my_window.show()
    app.exec_()


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('-v', help='verbose', action='store_true')

    p.add('--frame-dir', required=True)
    p.add('--truth-dir', default=None)
    p.add('--out-csv', required=True)

    cfg = p.parse_args()

    main(cfg.frame_dir, cfg.out_csv, cfg.truth_dir)
