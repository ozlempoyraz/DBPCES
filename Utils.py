import os
import sys

from kneed import KneeLocator
from pandas import np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

from enum import Enum
class Algo(Enum):
    DBPCES = 1
    KMEANS = 2
    DBSCAN = 3

def readData(path):
    dirname = os.path.dirname(__file__)
    path1 = os.path.join(dirname, path)
    if os.path.exists(path1):
        X = np.loadtxt(path1)
        Y = X[:, -1]  # for last column
        X = X[:, :-1]  # for all but last column
        return X, Y
    else:
        print(path1 + ' not found.')
        print('\nusage: python DBPCES.py datasetFilePath window_size init_size reinit_period algo:dbpces/kmeans/dbscan\n')
        sys.exit(1)

def purity_score(y_true, y_pred):
    con_mat = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(con_mat, axis=0))) / float(np.sum(con_mat))

def plot_with_knee(y):
    x = range(1, len(y) + 1)
    kn = KneeLocator(x, y, curve='convex', direction='increasing')
    return kn.knee_y

def find_optimal_epsilon(buffer, knn):
    if buffer.shape[0] >= knn:
        neigh = NearestNeighbors(n_neighbors=knn)
        nbrs = neigh.fit(buffer)
        distances, indices = nbrs.kneighbors(buffer)
        distances = distances[:, knn - 1]
        distances = np.sort(distances, axis=0)
        return plot_with_knee(distances)

def findParameters_binary(data, start, end, max_sil2=(-1, -1, -1)):
    mid1 = int((end - start) / 4 + start)
    mid2 = int((end - start) * 3 / 4 + start)
    if mid1 <= 1 or mid1 >= mid2:
        print("Max sil: " + str(max_sil2[0]))
        print("Max sil eps2: " + str(max_sil2[1]))
        print("Max sil minPts2: " + str(max_sil2[2]))
        return max_sil2[1], max_sil2[2]

    eps1 = find_optimal_epsilon(data, mid1)
    eps2 = find_optimal_epsilon(data, mid2)
    dbscan_cluster1 = DBSCAN(eps=eps1, min_samples=mid1).fit(data)
    sil1 = -1
    sil2 = -1
    if len(np.unique(dbscan_cluster1.labels_)) > 1:
        sil1 = silhouette_score(data, dbscan_cluster1.labels_)
    dbscan_cluster2 = DBSCAN(eps=eps2, min_samples=mid2).fit(data)
    if len(np.unique(dbscan_cluster2.labels_)) > 1:
        sil2 = silhouette_score(data, dbscan_cluster2.labels_)

    if sil1 >= sil2:
        if sil1 > max_sil2[0]:
            return findParameters_binary(data, start, (end - start) / 2 + start, max_sil2=(sil1, eps1, mid1))
        elif sil1 == max_sil2[0]:
            epsparam = max_sil2[1]
            minPtsparam = max_sil2[2]
            if eps1 <= max_sil2[1]:
                epsparam = eps1
                minPtsparam = mid1
            return findParameters_binary(data, start, (end - start) / 2 + start, max_sil2=(sil1, epsparam, minPtsparam))
        else:
            return findParameters_binary(data, start, (end - start) / 2 + start, max_sil2)
    else:
        if sil2 > max_sil2[0]:
            return findParameters_binary(data, (end - start) / 2 + start, end, max_sil2=(sil2, eps2, mid2))
        elif sil2 == max_sil2[0]:
            epsparam = max_sil2[1]
            minPtsparam = max_sil2[2]
            if eps2 <= max_sil2[1]:
                epsparam = eps2
                minPtsparam = mid2
            return findParameters_binary(data, (end - start) / 2 + start, end, max_sil2=(sil2, epsparam, minPtsparam))
        else:
            return findParameters_binary(data, (end - start) / 2 + start, end, max_sil2)