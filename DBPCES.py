import errno
import sys
import os
import time
import warnings
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

from Utils import readData, findParameters_binary, Algo, purity_score

randomState1 = 1
warnings.filterwarnings('ignore')

if len(sys.argv) != 6:
    print('\nusage: python DBPCES.py datasetFilePath horizon init_size reinit_period algo:dbpces/kmeans/dbscan\n')
    sys.exit(1)

path = sys.argv[1]
streamName = path.split("/")[-1].split(".")[0]
X, labels = readData(path)

try:
    horizon = int(sys.argv[2])
except ValueError:
    # Handle the exception
    print('Horizon should be an integer')
    print('\nusage: python DBPCES.py datasetFilePath horizon init_size reinit_period algo:dbpces/kmeans/dbscan\n')
    sys.exit(1)
try:
    init_size = int(sys.argv[3])
except ValueError:
    # Handle the exception
    print('init_size should be an integer')
    print('\nusage: python DBPCES.py datasetFilePath horizon init_size reinit_period algo:dbpces/kmeans/dbscan\n')
    sys.exit(1)
try:
    reinit_period = int(sys.argv[4])
except ValueError:
    # Handle the exception
    print('reinit_period should be an integer')
    print('\nusage: python DBPCES.py datasetFilePath horizon init_size reinit_period algo:dbpces/kmeans/dbscan\n')
    sys.exit(1)


algo = Algo.DBPCES
folder = "results/dbpces/"

if sys.argv[5] == 'dbpces':
    algo = Algo.DBPCES
    folder = "results/dbpces/"
elif sys.argv[5] == 'kmeans':
    algo = Algo.KMEANS
    folder = "results/kmeans/"
elif sys.argv[5] == 'dbscan':
    algo = Algo.DBSCAN
    folder = "results/dbscan/"
else:
    print('\nusage:Invalid algorithm name. valid options:dbpces/kmeans/dbscan\n')
    print('\nusage: python DBPCES.py datasetFilePath horizon init_size reinit_period algo:dbpces/kmeans/dbscan\n')
    sys.exit(1)


filename = streamName + "-horizon" + str(horizon) + "-initSize" + str(init_size) + "-reinit" + str(reinit_period) + ".txt"
output = folder + filename

os.makedirs(os.path.dirname(output), exist_ok=True)

sys.stdout = open(output, "w")
cluster_count = len(np.unique(labels))

length = len(X)
half_init = int(init_size / 2)
reinit_cnt = int(np.math.ceil(length / reinit_period))

minPts = 5
print('data length : {}'.format(length))

print('reinit period: {}\ninit size: {}\nhorizon: {}\nminPts: {}'.format(reinit_period, init_size, horizon, minPts))
total_embedding = np.empty(shape=[0, 2])
total_klabels = []

embedding = None
dataPeriod = None
match_fnx = None
first_run = True

all_times = []
all_siluets = []
all_purity = []
all_aris = []
cluster_counts = []


for reinit in range(reinit_cnt):
    embed_count = 0

    reducer = umap.UMAP(random_state=randomState1)
    randomState1 = randomState1 + 1
    start = time.time()

    label = []

    initTimeStart = time.time()
    # if embedding is None:
    if first_run:
        # at the beginning of the list, real init
        # print('initing UMAP algorithm with {}-{}'.format(reinit * reinit_period, reinit * reinit_period + init_size))
        dataPeriod = X[reinit * reinit_period:reinit * reinit_period + init_size]

        if algo == Algo.DBPCES or algo == Algo.KMEANS:
            embedding = reducer.fit_transform(dataPeriod)

        # data = X[reinit * reinit_period:reinit * reinit_period + init_size]
        label = labels[reinit * reinit_period:reinit * reinit_period + init_size]
        embed_count = embed_count + init_size

        step_cnt = (reinit_period - init_size) / horizon
        start_index = reinit * reinit_period + init_size
    else:
        # re-init
        # print('reiniting UMAP algorithm with {}-{}'.format(reinit * reinit_period - half_init,
        #                                                    reinit * reinit_period + half_init))
        dataPeriod = X[reinit * reinit_period - half_init:reinit * reinit_period + half_init]
        label = labels[reinit * reinit_period - half_init:reinit * reinit_period + half_init]

        if algo == Algo.DBPCES or algo == Algo.KMEANS:
            embedding = reducer.fit_transform(dataPeriod)

        embed_count = embed_count + half_init
        step_cnt = np.math.ceil((reinit_period - half_init) / horizon)
        start_index = reinit * reinit_period + half_init
    end_index = (reinit + 1) * reinit_period
    initTimeEnd = time.time()
    print('***init elapsed time is : {}***'.format((initTimeEnd - initTimeStart)))

    init_embedding = embedding
    init_data = dataPeriod

    # each window of umap
    windowTimeStart = time.time()
    for i in range(int(step_cnt)):
        firstIndex = start_index + i * horizon
        if firstIndex > end_index:
            break
        lastIndex = start_index + (i + 1) * horizon
        if lastIndex > end_index:
            lastIndex = end_index
        embed_count = embed_count + lastIndex - firstIndex
        # data = np.concatenate(data, X[start_index + i * horizon:start_index + (i + 1) * horizon])
        label = np.concatenate((label, labels[firstIndex:lastIndex]), axis=0)
        if algo != Algo.DBSCAN:
            embedding = np.append(embedding, reducer.transform(X[firstIndex:lastIndex]), axis=0)
        dataPeriod = np.concatenate((dataPeriod, X[firstIndex:lastIndex]), axis=0)
        # print('X[{}:{}]'.format(start_index+i*horizon, start_index+(i+1)*horizon))
    windowTimeEnd = time.time()
    print('***windows elapsed time is : {}***'.format((windowTimeEnd - windowTimeStart)))
    timedbscanStart = time.time()

    if algo == Algo.DBPCES:
        eps1, minPts1 = findParameters_binary(init_embedding, 0, len(init_embedding))
        clustering = DBSCAN(eps=eps1, min_samples=minPts1).fit(embedding)
    elif algo == Algo.KMEANS:
        clustering = KMeans(n_clusters=cluster_count, random_state=0).fit(embedding)
    elif algo == Algo.DBSCAN:
        eps1, minPts1 = findParameters_binary(init_data, 0, len(init_data))
        clustering = DBSCAN(eps=eps1, min_samples=minPts1).fit(dataPeriod)

    timedbscanEnd = time.time()

    klabels = [x for x in clustering.labels_]

    print('***dbscan elapsed time is : {}***'.format((timedbscanEnd - timedbscanStart)))

    if not first_run:
        prev_tomatch = 0
        curr_tomatch = 0
        prev_tomatch = prev_labels[-half_init:]
        curr_tomatch = klabels[:half_init]

        cluster_cnt_prev = int(np.max(prev_tomatch) + 1)
        cluster_cnt_current = int(np.max(curr_tomatch) + 1)

        match = np.zeros(shape=[cluster_cnt_current, cluster_cnt_prev])
        match_fnx = np.zeros(shape=[cluster_cnt_current])
        for i in range(half_init):
            if int(curr_tomatch[i]) != -1:
                match[int(curr_tomatch[i]), int(prev_tomatch[i])] += 1
        for i in range(cluster_cnt_current):
            match_fnx[i] = np.argmax(match[i])

        for i in range(len(klabels)):
            if klabels[i] < len(match_fnx) and klabels[i] != -1:
                klabels[i] = match_fnx[klabels[i]]

    print("Actual cluster count:" + str(len(np.unique(labels))))
    print("Predicted cluster count:" + str(len(np.unique(klabels))))
    cluster_counts.append(len(np.unique(klabels)))

    prev_labels = klabels

    end = time.time()
    elapsedtime = end - start
    all_times.append(elapsedtime)
    if algo != Algo.DBSCAN:
        print('embedding shape is : {}'.format(embedding.shape))
    print('***Elapsed time is : {}***'.format(elapsedtime))

    ### plot results
    if first_run:
        if algo == Algo.DBSCAN:
            siluet = silhouette_score(dataPeriod, labels[:reinit_period])
        else:
            siluet = silhouette_score(embedding, labels[:reinit_period])

        purity = purity_score(labels[:reinit_period], klabels)
        ari = adjusted_rand_score(labels[:reinit_period], klabels)
        print('ari of dbscan labels of this part [{}:{}] is : {}'.format(0, reinit_period, ari))
        print('purity of dbscan labels of this part [{}:{}] is : {}'.format(0, reinit_period, purity))

    else:

        firstIndex = reinit * reinit_period - half_init
        lastIndex = (reinit + 1) * reinit_period

        if firstIndex > len(labels):
            break
        if lastIndex > len(labels):
            lastIndex = len(labels)

        siluet = 0
        if len(np.unique(klabels)) > 1:
            if algo == Algo.DBSCAN:
                siluet = silhouette_score(dataPeriod, klabels)
            else:
                siluet = silhouette_score(embedding, klabels)

        ari = adjusted_rand_score(labels[firstIndex:lastIndex],
                                  klabels)
        purity = purity_score(labels[firstIndex:lastIndex],
                                  klabels)
        print('ari of dbscan labels of this part [{}:{}] is : {}'.format(firstIndex, lastIndex, ari))
        print('purity of dbscan labels of this part [{}:{}] is : {}'.format(firstIndex, lastIndex, purity))

    all_aris.append(ari)
    all_siluets.append(siluet)
    all_purity.append(purity)
    print('silhouette score of clustering is : {}'.format(siluet))
    if algo != Algo.DBSCAN:
        total_embedding = np.append(total_embedding, embedding[-embed_count:, :], axis=0)
    total_klabels = total_klabels + klabels[-embed_count:]

    first_run = False

    print('----------')

print('---------- o ----------')

if algo != Algo.DBSCAN:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax1.scatter(total_embedding[:, 0], total_embedding[:, 1], c=labels, s=0.5, cmap='Spectral')
    ax1.set_title('Actual Labels', fontsize=24)
    ax2.scatter(total_embedding[:, 0], total_embedding[:, 1], c=total_klabels, s=0.5, cmap='Spectral')
    ax2.set_title('Predicted Labels', fontsize=24)
    filename2 = streamName + "-horizon" + str(horizon) + "-initSize" + str(init_size) + "-reinit" + str(reinit_period) + ".png"
    output2 = folder + filename2
    fig.savefig(output2)

print("Stream name: " + streamName)
print("Horizon: " + str(horizon))
print("Init size: " + str(init_size))
print("Overlap size" + str(half_init))
print("Reinit period: " + str(reinit_period))
print("Clustering Algorithm: " + str(algo))

if algo != Algo.DBSCAN:
    print('total embedding shape is : {}'.format(total_embedding.shape))

print('calculating adjusted rand index on total results')
ari = adjusted_rand_score(labels, total_klabels)
print('ari of total dbscan labels is : {}'.format(ari))

print('calculating purity on total results')
purity = purity_score(labels, total_klabels)
print('purity of total dbscan labels is : {}'.format(purity))

print('calculating cluster count on total results')
count = len(np.unique(total_klabels))
print('Cluster count on total results is : {}'.format(count))

print('calculating total execution time')
print('Total execution time is : {}'.format(sum(all_times)))


print('----------')
print(streamName + " Min " + "Max " + "Average")
print('measured time values: ' + str(min(all_times)) + " " + str(max(all_times)) + " " + str(mean(all_times)))
print('measured silhouette scores: ' + str(min(all_siluets)) + " " + str(max(all_siluets)) + " " + str(mean(all_siluets)))
print('measured aris: ' + str(min(all_aris)) + " " + str(max(all_aris)) + " " + str(mean(all_aris)))
print('measured purity: ' + str(min(all_purity)) + " " + str(max(all_purity)) + " " + str(mean(all_purity)))
print('predicted cluster count: ' + str(min(cluster_counts)) + " " + str(max(cluster_counts)) + " " + str(mean(cluster_counts)))
print('----------')
print('---------- o ----------')

sys.stdout.close()