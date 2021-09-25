# Density-Based and Parameterless Clustering of Embedded Data Streams (DBPCES)
This repository consists of implementation of a data stream clustering algorithm DBPCES using python 3.6.

DBPCES,
* embeds data streams into two dimensions using UMAP and clusters them using DBSCAN algorithm,
* does not require any data-dependent parameter for clustering. Parameters of DBSCAN are estimated using a heuristic,
* adapts UMAP to handle concept drift,
* estimates and updates DBSCAN parameters periodically to adapt concept drift,
* requires 3 hyper-parameters:
    * **reinit_period:** The period of clustering in terms of the number of data points.
    * **init_size:** The number of data points used in the UMAP model generation and DBSCAN parameter estimation.
    * **window_size:** The number of data points embedded together using generated UMAP model.


Algorithm can be run with the following command: <br /> 

```python DBPCES.py <datasetFilePath> <window_size> <init_size> <reinit_period> <dbpces | kmeans | dbscan>```<br /> 

The last parameters specifies the clustering algorithm that will be used. In order to run original DBPCES, the last parameter should be `dbpces`. Other algorithms can be executed for the comparison. UMAP is not used in `dbscan` option. 
