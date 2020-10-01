from skimage import data, segmentation
from skimage.future import graph
import matplotlib.pyplot as plt

img = data.coffee()
labels = segmentation.slic(img)
g =  graph.rag_mean_color(img, labels)
lc = graph.show_rag(labels, g, img)
cbar = plt.colorbar(lc)
plt.show()
