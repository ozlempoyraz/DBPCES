# DBPCES
Density-Based and Parameterless Clustering of Embedded Data Streams
This repository consists of implementation of a data stream clustering algorithm: DBPCES. 
DBPCES algorithm embeds data streams into two dimensions using UMAP and clusters them using DBSCAN algorithm.
DBPCES does not require any data-dependent parameter for clustering. Parameters of DBSCAN are estimated using a heuristic. 
UMAP algorithm is adapted to concept drift in DBPCES. Additionally, DBSCAN parameters are estimated and updated periodically to adapt concept drift. 
Labels of each period is matched using the matching algorithm of DBPCES.
DBPCES requires 3 hyper-parameters: reinit_period, init_size, window_size.
**reinit_period:** The period of clustering in terms of the number of data points.
**init_size:** The number of data points used in the UMAP model generation and DBSCAN parameter estimation.
**window_size:** The number of data points embedded together using generated UMAP model.
