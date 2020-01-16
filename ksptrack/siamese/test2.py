import numpy as np
from skimage import draw, filters, segmentation
import matplotlib.pyplot as plt
import networkx as nx
import tqdm
import scipy
import collections
import features
import copy


def _new_label_image(F, i, j, t):
    Fk = np.copy(F)
    Fk[Fk == i] = Fk[Fk == j] = t
    return Fk


def _merge_similarity_set(feature_extractor, Ak, S, i, j, t):
    # remove entries which have i or j
    S = list(filter(lambda x: not (i in x[1] or j in x[1]), S))

    # calculate similarity between region t and its adjacencies
    St = [(feature_extractor.similarity(t, x), (t, x)) for x in Ak[t] if t < x] +\
         [(feature_extractor.similarity(x, t), (x, t)) for x in Ak[t] if x < t]

    return sorted(S + St)


def _new_adjacency_dict(A, i, j, t):
    Ak = copy.deepcopy(A)
    Ak[t] = (Ak[i] | Ak[j]) - {i, j}
    del Ak[i], Ak[j]
    for (p, Q) in Ak.items():
        if i in Q or j in Q:
            Q -= {i, j}
            Q.add(t)

    return Ak


def _build_initial_similarity_set(A0, feature_extractor):
    S = list()

    for (i, J) in A0.items():
        S += [(feature_extractor.similarity(i, j), (i, j)) for j in J if i < j]

    return sorted(S)


def make_pw_distance(arr, neighborhood=8):
    assert ((neighborhood == 8)
            or (neighborhood == 4)), 'set neighborhood to 4 or 8'

    idx = np.arange(0, arr.size)
    idx = idx.reshape(arr.shape)

    dist_adj = {}

    # top
    dist = np.concatenate((np.zeros((1, arr.shape[1])),
                           np.abs((arr[1:, :] - arr[0:-1, :]))),
                          axis=0)
    dist_adj['top'] = dist

    # right
    dist = np.concatenate((np.abs((arr[:, 1:] - arr[:, 0:-1])),
                           np.zeros((arr.shape[0], 1))),
                          axis=1)
    dist_adj['right'] = dist

    if (neighborhood == 8):
        # top-right
        dist = np.concatenate((np.zeros((1, arr.shape[1]-1)),
                               np.abs((arr[1:, :-1] - arr[:-1, 1:]))),
                              axis=0)
        dist = np.concatenate((dist,
                               np.zeros((dist.shape[0], 1))),
                              axis=1)

        dist_adj['topright'] = dist

    return dist_adj

def _calc_adjacency_matrix(label_img, n_region):
    r = np.vstack([label_img[:, :-1].ravel(), label_img[:, 1:].ravel()])
    b = np.vstack([label_img[:-1, :].ravel(), label_img[1:, :].ravel()])
    t = np.hstack([r, b])
    A = scipy.sparse.coo_matrix((np.ones(t.shape[1]), (t[0], t[1])),
                                shape=(n_region, n_region),
                                dtype=bool).todense().getA()
    A = A | A.transpose()

    for i in range(n_region):
        A[i, i] = True

    dic = {i: {i} ^ set(np.flatnonzero(A[i])) for i in range(n_region)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix=A, dictionary=dic)


shape = 512
n_segments = 200
sigma = 2

truth = np.zeros((shape, shape))
rr, cc = draw.circle(shape // 2, shape // 2, shape // 4)
truth[rr, cc] = 1

truth_filtered = filters.gaussian(truth, sigma=sigma)
affinities = make_pw_distance(truth_filtered)
labels = segmentation.slic(truth_filtered, n_segments=200, compactness=20.)

feat_extractor = features.Features([v for v in affinities.values()],
                                   labels,
                                   truth_filtered.size)

n_region = np.unique(labels.ravel()).shape[0]
adj_mat, A0 = _calc_adjacency_matrix(labels, n_region)

S = _build_initial_similarity_set(A0, feat_extractor)
g = nx.DiGraph()
g.add_nodes_from([(l, {'mass': 1,
                       'stack': 0}) for l in np.unique(labels)])

A = [A0]  # stores adjacency relation for each step
F = [labels]  # stores label image for each step

# stores region label and its parent (empty if initial).
R = {i: () for i in range(n_region)}
# greedy hierarchical grouping loop
stack = 0
while len(S):
    stack += 1
    (s, (i, j)) = S.pop()
    t = feat_extractor.merge(i, j)
    n_elems_left = nx.get_node_attributes(g, 'n_elems')[i]
    n_elems_right = nx.get_node_attributes(g, 'n_elems')[j]
    new_n_elems = n_elems_left + n_elems_right
    g.add_node(t, n_elems=new_n_elems, stack=stack)
    g.add_edge(t, i, diff_n_elems=new_n_elems - n_elems_left)
    g.add_edge(t, j, diff_n_elems=new_n_elems - n_elems_right)

    # record merged region (larger region should come first)
    R[t] = (
        i,
        j) if feat_extractor.size[j] < feat_extractor.size[i] else (
            j, i)

    Ak = _new_adjacency_dict(A[-1], i, j, t)
    A.append(Ak)

    S = _merge_similarity_set(feat_extractor, Ak, S, i, j, t)

    new_label_image = _new_label_image(F[-1], i, j, t)
    F.append(new_label_image)



fig, ax = plt.subplots(2, 3)
ax = ax.flatten()

ax[0].imshow(truth)
ax[1].imshow(truth_filtered)
ax[2].imshow(pw_dist['top'][-1, :].reshape((shape - 1, shape)))
ax[3].imshow(pw_dist['right'][-1, :].reshape((shape, shape - 1)))
ax[4].imshow(pw_dist['topright'][-1, :].reshape((shape - 1, shape - 1)))
plt.show()
