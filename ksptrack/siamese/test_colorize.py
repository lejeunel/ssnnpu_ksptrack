import numpy as np
import matplotlib.pyplot as plt

labels = np.load('/home/ubelix/lejeune/data/medical-labeling/Dataset00/precomp_desc/sp_labels.npz')
labels = labels['sp_labels'][..., 0]

shape = labels.shape
n_clusters = 10
predictions = np.random.randint(0, n_clusters-1, size=np.unique(labels).size)
cmap = plt.get_cmap('viridis')
mapping = np.array([(np.array(cmap(c / n_clusters)[:3]) * 255).astype(np.uint8)
                    for c in predictions.ravel()])
mapping = np.concatenate((np.unique(labels)[..., None], mapping), axis=1)

_, ind = np.unique(labels, return_inverse=True)
clusters_colorized = mapping[ind, 1:].reshape((shape[0], shape[1], 3))

plt.subplot(121)
plt.imshow(labels)
plt.subplot(122)
plt.imshow(clusters_colorized)
plt.show()
