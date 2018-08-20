from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import cfg
import pandas as pd
import pickle as pk
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
import superPixels as spix
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import graphtracking as gtrack
import my_utils as utls
import dataset as ds
#import selective_search as ss
import shutil as sh
import learning_dataset
import logging
from skimage import filters
import superpixelmanager as spm
import networkx as nx

edges = []
edges.append(('a', 'b', 1))
edges.append(('a', 'e', 1))
edges.append(('a', 'g', 7))
edges.append(('e', 'b', 1))
edges.append(('b', 'c', 1))
edges.append(('b', 'f', 1))
edges.append(('c', 'd', 1))
edges.append(('d', 'z', 1))
edges.append(('g', 'z', 2))
edges.append(('f', 'd', 1))
edges.append(('f', 'z', 4))
edges.append(('b', 'f', 1))
edges.append(('e', 'f', 3))
edges.append(('c', 'd', 1))

#g_my = nx.DiGraph()
#g_my.add_weighted_edges_from(edges)
#g_my_ksp = gtrack.GraphTracking(None,
#                                tol=10e-8,
#                                mode='edge')
#g_my_ksp.g = g_my
#
#g_my_ksp.orig_weights = nx.get_edge_attributes(g_my, 'weight')
#g_my_ksp.source = 'a'
#g_my_ksp.sink = 'z'
#
#find_new_backward = g_my_ksp.disjointKSP(None, verbose=True)
