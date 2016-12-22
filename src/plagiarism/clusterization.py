import scipy
import scipy.cluster


def kmeans(matrix, k, whiten=True, iter=10):
    """
    Performs a k-means classification for all documents in the given job.

    Args:
        matrix:
            A matrix of vectorized documents. This matrix is usually obtained
            by calling vectorize(bag_of_documents).
        k (int):
            The desired number of clusters.
        whiten:
            True if data should be whitened before analysis.
        iter:
            Number of iterations of the k-means algorithm.

    Return:
        centroids:
            A 2D array with all found centroids.
        labels:
            A sequence in witch the i-th element correspond to the cluster index
            for the i-th document.
    """

    std = 1
    if whiten:
        std = matrix.std(axis=0)
        std[std == 0] = 1
        matrix /= std[None, :]
    centroids, labels = scipy.cluster.vq.kmeans2(matrix, k, minit='points',
                                                 iter=iter)
    centroids *= std
    return centroids, labels


def dbscan(matrix, eps=0.1, k=5, metric='l1', normalize=True, **kwargs):
    """
    DBSCAN clustering

    Args:
        distance_matrix:
        eps:

    Returns:

    """

    from sklearn.cluster import dbscan
    from sklearn.preprocessing import StandardScaler

    if normalize:
        data = StandardScaler().fit_transform(matrix)
    else:
        data = matrix

    samples, labels = dbscan(data, eps, metric=metric, min_samples=k,
                             **kwargs)
    return samples, labels