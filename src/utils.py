import numpy as np
from sklearn.cross_decomposition import CCA


def svd_pca(data, k):
    """Reduce DATA using its K principal components."""
    data = data.astype("float64")
    data -= np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    return U[:,:k].dot(np.diag(S)[:k,:k])


def cross_decomp(X, Y, k):
    cc_ = CCA(n_components=k)
    cc_.fit(X, Y)
    X_, Y_ = cc_.transform(X, Y)
    return X_
