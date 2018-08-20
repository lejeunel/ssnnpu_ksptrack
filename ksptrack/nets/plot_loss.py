import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from cycler import cycler
import os

#path = '/home/laurent.lejeune/medical-labeling/learning_exps/learning_Slitlamp_2017-12-11_09-51-30/my/fold_0/train/log.csv'
paths = [
    '/home/laurent.lejeune/medical-labeling/learning_exps/learning_Slitlamp_2017-12-11_09-51-30/my/fold_0/train/log.csv',
    '/home/laurent.lejeune/medical-labeling/learning_exps/learning_Slitlamp_2018-01-10_13-29-17/my/fold_0/train/log.csv']

df_losses = [pd.read_csv(p, sep=';') for p in paths]

loss_type = 'binary_crossentropy'

color = iter(cm.rainbow(np.linspace(0,1,len(paths))))

for l in df_losses:
    c = next(color)
    loss = l[loss_type]
    epochs = l['epoch']

    plt.plot(epochs, loss, c=c)
    plt.title(loss_type)
    plt.ylabel('validation error')
    plt.xlabel('epoch')

plt.legend(paths)
plt.grid()
plt.show()
