import scipy
import scipy.cluster


def kmeans(data, k, whiten=True):
    """
    Performs a k-means classification for all documents in the given job.

    Args:
        matrix:
            A list of vectorized documents or matrix.
        k (int):
            The desired number of clusters.

    Return:
        centroids:
            A 2D array with all found centroids.
        labels:
            A sequence in witch the i-th element correspond to the cluster index
            for the i-th document.
    """

    std = 1
    if whiten:
        std = data.std(axis=0)
        std[std == 0] = 1
        data /= std[None, :]
    centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='points')
    centroids *= std
    return centroids, labels