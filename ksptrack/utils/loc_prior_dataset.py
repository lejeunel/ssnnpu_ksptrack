from ksptrack.utils.base_dataset import BaseDataset
from os.path import join as pjoin
import os
import pandas as pd
import numpy as np
import imgaug as ia
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage import segmentation
from scipy import ndimage as nd
import networkx as nx


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)


def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


def readCsv(csvName, seqStart=None, seqEnd=None):

    out = np.loadtxt(open(csvName, "rb"), delimiter=";",
                     skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)

    return pd.DataFrame(data=out,
                        columns=['frame', 'time', 'visible', 'x', 'y'])


def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


class LocPriorDataset(BaseDataset, data.Dataset):
    """
    Adds objectness prior using 2d locations
    """
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=iaa.Noop(),
                 resize_shape=None,
                 csv_fname='video1.csv',
                 sig_prior=0.04):
        super().__init__(root_path=root_path,
                         augmentations=augmentations,
                         normalization=normalization,
                         resize_shape=resize_shape)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, 'gaze-measurements', csv_fname)
        if (os.path.exists(locs2d_path)):
            self.locs2d = readCsv(locs2d_path)
        else:
            raise Exception('couldnt find 2d locs file {}'.format(locs2d_path))

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        orig_shape = self.imgs[0].shape[:2]
        new_shape = sample['image'].shape

        locs = self.locs2d[self.locs2d['frame'] == idx]
        locs = [
            coord2Pixel(l['x'], l['y'], orig_shape[1], orig_shape[0])
            for _, l in locs.iterrows()
        ]

        keypoints = ia.KeypointsOnImage(
            [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
            shape=(orig_shape[0], orig_shape[1]))

        keypoints = self.reshaper_seg.augment_keypoints(keypoints)

        if (len(locs) > 0):
            obj_prior = [
                make_2d_gauss((new_shape[0], new_shape[1]),
                              self.sig_prior * max(new_shape), (kp.y, kp.x))
                for kp in keypoints.keypoints
            ]
            obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
            # offset = np.ones_like(obj_prior) * 0.5
            obj_prior -= obj_prior.min()
            obj_prior /= obj_prior.max()
            # obj_prior *= 0.5
            # obj_prior += offset
        else:
            obj_prior = (np.ones((new_shape[0], new_shape[1])))[..., None]

        sample['prior'] = obj_prior
        sample['loc_keypoints'] = keypoints

        coords = np.array([(np.round(kp.y).astype(int),
                            np.round_(kp.x).astype(int))
                           for kp in keypoints.keypoints])
        if (coords.shape[0] > 0):
            coords[:, 0] = np.clip(coords[:, 0],
                                   a_min=0,
                                   a_max=keypoints.shape[0] - 1)
            coords[:, 1] = np.clip(coords[:, 1],
                                   a_min=0,
                                   a_max=keypoints.shape[1] - 1)

        sample['labels_clicked'] = [
            sample['labels'][i, j, 0] for i, j in coords
        ]

        return sample

    @staticmethod
    def collate_fn(data):

        out = super(LocPriorDataset, LocPriorDataset).collate_fn(data)

        obj_prior = [np.rollaxis(d['prior'], -1) for d in data]
        obj_prior = torch.stack(
            [torch.from_numpy(i).float() for i in obj_prior])

        out['prior'] = obj_prior
        out['loc_keypoints'] = [d['loc_keypoints'] for d in data]
        out['labels_clicked'] = [s['labels_clicked'] for s in data]

        return out


def _add_edge_filter(values, g):
    """Add an edge between first element in `values` and
    all other elements of `values` in the graph `g`.
    `values[0]` is expected to be the central value of
    the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    g : RAG
        The graph to add edges in.

    Returns
    -------
    0.0 : float
        Always returns 0.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        g.add_edge(current, value)
    return 0.0


if __name__ == "__main__":

    from ilastikrag import rag
    import vigra

    dset = LocPriorDataset(root_path=pjoin(
        '/home/ubelix/lejeune/data/medical-labeling/Dataset00'),
                           normalization='rescale',
                           resize_shape=512)
    frames = [14, 15]
    labels_comp = [
        2048, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
        21, 22, 2067, 24, 2068, 26, 27, 28, 2074, 2075, 2073, 32, 35, 38, 39,
        42, 43, 47, 48, 2095, 2096, 51, 52, 2101, 54, 55, 56, 57, 2104, 59, 60,
        61, 62, 63, 64, 2109, 66, 2107, 2116, 2118, 2119, 72, 75, 76, 77, 81,
        2131, 84, 85, 2132, 87, 89, 90, 91, 2053, 2054, 97, 100, 105, 107, 108,
        110, 111, 112, 116, 119, 120, 121, 122, 124, 125, 126, 132, 135, 137,
        142, 145, 147, 148, 152, 153, 154, 156, 157, 158, 161, 163, 164, 165,
        167, 171, 179, 185, 187, 188, 196, 199, 200, 202, 205, 219, 220, 227,
        228, 233, 234, 236, 238, 253, 260, 261, 268, 269, 270, 274, 291, 294,
        295, 296, 297, 298, 309, 315, 316, 320, 327, 335, 336, 337, 338, 339,
        345, 2105, 354, 355, 361, 365, 373, 374, 376, 377, 378, 383, 393, 402,
        408, 418, 420, 429, 432, 433, 434, 437, 444, 451, 463, 466, 469, 472,
        473, 479, 485, 486, 492, 493, 494, 495, 504, 505, 507, 509, 515, 516,
        517, 519, 520, 528, 530, 547, 548, 552, 553, 555, 556, 557, 561, 562,
        565, 588, 590, 591, 592, 593, 594, 598, 599, 600, 609, 614, 615, 619,
        621, 622, 623, 628, 647, 648, 649, 655, 656, 657, 661, 662, 682, 693,
        694, 695, 696, 699, 704, 720, 721, 722, 730, 741, 755, 763, 764, 765,
        788, 789, 795, 796, 821, 822, 832, 833, 834, 835, 852, 866, 887, 896,
        914, 926, 930, 931, 932, 957, 962, 965, 969, 981, 986, 998, 1001, 1004,
        1009, 1026, 1027, 1032, 1035, 1036, 1039, 1046, 1047, 1049, 1050, 1063,
        1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076,
        1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088,
        1090, 1092, 1096, 1099, 1100, 1103, 1104, 1105, 1112, 1113, 1114, 1115,
        1118, 1119, 1120, 1121, 1122, 1123, 1124, 1126, 1127, 1128, 1130, 1133,
        1136, 1138, 1140, 1141, 1142, 1145, 1146, 1152, 1153, 1154, 1155, 1160,
        1161, 1164, 1168, 1170, 1171, 1172, 1174, 1177, 1183, 1184, 1185, 1187,
        1188, 1197, 1198, 1200, 1205, 1207, 1213, 1216, 1217, 1219, 1220, 1221,
        1224, 1225, 1226, 1227, 1230, 1242, 1245, 1248, 1250, 1257, 1259, 1261,
        1262, 1263, 1264, 1268, 1283, 1288, 1289, 1293, 1295, 1300, 1302, 1323,
        1324, 1327, 1331, 1332, 1333, 1334, 1339, 1358, 1359, 1361, 1362, 1363,
        1372, 1373, 1379, 1383, 1395, 1396, 1397, 1398, 1399, 1400, 1407, 1416,
        1417, 1427, 1435, 1436, 1438, 1439, 1440, 1447, 1462, 1467, 1472, 1473,
        1481, 1483, 1492, 1495, 1496, 1497, 1500, 1501, 1504, 1514, 1516, 1523,
        1527, 1529, 1530, 1533, 1536, 1537, 1542, 1547, 1554, 1555, 1556, 1557,
        1559, 1568, 1569, 1571, 1573, 1579, 1580, 1581, 1583, 1584, 1592, 1594,
        1608, 1613, 1617, 1618, 1619, 1623, 1627, 1630, 1657, 1658, 1659, 1660,
        1661, 1663, 1665, 1666, 1667, 1675, 1682, 1683, 1688, 1689, 1690, 1695,
        1697, 1713, 1717, 1718, 1722, 1723, 1729, 1731, 1748, 1762, 1763, 1764,
        1765, 1767, 1771, 1781, 1789, 1796, 1797, 1809, 1822, 1823, 1824, 1832,
        1855, 1857, 1863, 1864, 1887, 1890, 1900, 1901, 1902, 1920, 1934, 1935,
        1954, 1964, 1965, 1981, 1994, 1995, 1999, 2000, 2003, 2026, 2032, 2036,
        2037, 2047
    ]
    all_labels = []
    rags = []
    max_node = 0
    for f in frames:
        labels = dset[f]['labels'] + max_node
        all_labels.append(labels.squeeze())
        max_node += labels.max() + 1

    # all_labels = np.concatenate(all_labels, axis=-1)
    # all_labels = vigra.Volume(all_labels, dtype=np.uint32)
    # full_rag = rag.Rag(all_labels).edge_ids.T.astype(np.int32)

    labels_on = [np.zeros_like(all_labels[0]) for l in all_labels]
    for i in range(len(all_labels)):
        for l in labels_comp:
            labels_on[i][all_labels[i] == l] = True

    plt.subplot(221)
    plt.imshow(labels_on[0])
    plt.subplot(222)
    plt.imshow(all_labels[0])
    plt.subplot(223)
    plt.imshow(labels_on[1])
    plt.subplot(224)
    plt.imshow(all_labels[1])
    plt.show()

# frames = [10, 11]
# label_stack = []
# max_node = 0
# for f in frames:
#     labels = dset[f]['labels']
#     label_stack.append(labels + max_node)
#     max_node += labels.max() + 1

# labels_stack = np.concatenate(label_stack, axis=-1)

# g = nx.Graph()

# # run the add-edge filter on the regions
# nd.generic_filter(labels_stack,
#                   function=_add_edge_filter,
#                   footprint=fp,
#                   mode='nearest',
#                   extra_arguments=(g, ))
