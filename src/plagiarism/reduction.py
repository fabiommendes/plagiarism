from sklearn.decomposition.pca import PCA
from matplotlib import pyplot as plt


def plot_reduced(matrix, show=False, **kwargs):
    data = reduce_pca(matrix, **kwargs)
    plt.title('Reduced components')
    plt.plot(*data.T, 'ok')
    plt.xlabel('x')
    plt.ylabel('y')

    if show:
        plt.show()

def reduce_pca(matrix, whiten=True):
    pca = PCA(2, whiten=whiten)
    r = pca.fit(matrix.T)
    return r.components_.T