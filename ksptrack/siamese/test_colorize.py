import numpy as np
import matplotlib.pyplot as plt


shape = (10, 10)
n_clusters = 10
predictions = np.random.randint(0, n_clusters-1, size=np.prod(shape)).reshape(shape)
labels = np.arange(np.prod(shape)).reshape(shape)
cmap = plt.get_cmap('viridis')
mapping = np.array([(np.array(cmap(c / n_clusters)[:3]) * 255).astype(np.uint8)
                    for c in predictions.ravel()])
mapping = np.concatenate((np.unique(labels)[..., None], mapping), axis=1)

clusters_colorized = np.zeros((shape[0]*shape[1], 3)).astype(np.uint8)

_, ind = np.unique(labels, return_inverse=True)
clusters_colorized = mapping[mapping[:,0]][:, 1:].reshape((shape[0], shape[1], 3))

plt.subplot(121)
plt.imshow(predictions)
plt.colorbar()
plt.subplot(122)
plt.imshow(clusters_colorized)
plt.show()
