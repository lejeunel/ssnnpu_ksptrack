import numpy as np
import matplotlib.pyplot as plt

def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5*((x-x0)/std)**2)

    return y/np.sum(y)
    

def make_2d_gauss(shape, std, center):

    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1,1), (1, shape[1]))

    g = g_x*g_y

    return g/np.sum(g)
    
shape = (500, 700)

ci = 50
cj = 200

center = (ci, cj)

std = 50

g = make_2d_gauss(shape, std, center)

