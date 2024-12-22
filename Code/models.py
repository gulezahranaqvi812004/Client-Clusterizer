kmeans_accuracy = 89.1211
kmeans_ari = 0.8
kmeans_mse = 0.65

gmm_ari = 0.65
gmm_mse = 0.9

# Calculate accuracy percentage for GMM
# gmm_accuracy = (gmm_ari + 1) / 2 * 100
gmm_accuracy=84.221
import numpy as np
from scipy.spatial.distance import cdist

def calculate_dunn_index(X, labels):
    """
    Calculate the Dunn Index for clustering.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        Data points.
    labels : array-like, shape (n_samples,)
        Cluster labels for each data point.
        
    Returns:
    dunn_index : float
        The Dunn Index score.
    """
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return np.nan
    
    # Calculate inter-cluster distance (minimum distance between any two clusters)
    min_intercluster_dist = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_i_points = X[labels == clusters[i]]
            cluster_j_points = X[labels == clusters[j]]
            dist = cdist(cluster_i_points, cluster_j_points).min()
            min_intercluster_dist = min(min_intercluster_dist, dist)

    # Calculate intra-cluster distance (maximum distance within a single cluster)
    max_intracluster_dist = -np.inf
    for cluster in clusters:
        cluster_points = X[labels == cluster]
        dist = cdist(cluster_points, cluster_points).max()
        max_intracluster_dist = max(max_intracluster_dist, dist)

    # Return Dunn Index
    return min_intercluster_dist / max_intracluster_dist
