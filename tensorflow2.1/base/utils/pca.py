import numpy as np

from sklearn.decomposition import PCA
from .common import normalize

## note: need normalization after pca

def skpca(X_train, X_test, n_components=None, copy=True, whiten=True):
    '''
    X: numpy.ndarray, N x D
    '''
    pca = PCA(n_components=n_components, copy=copy, whiten=whiten)
    pca.fit(X_train)
    X_pw = pca.transform(X_test)
    return normalize(X_pw)

def whitenapply(X, m, P, dims=None):
    '''
    X: numpy.ndarray, D x N
    '''
    
    if not dims:
        dims = P.shape[0]

    X = np.dot(P[:dims, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X

def pcawhitenlearn(X):
    '''
    X: numpy.ndarray, D x N
    '''

    N = X.shape[1]
    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    
    return m, P

def whitenlearn(X, qidxs, pidxs):
    '''
    X: numpy.ndarray, D x N
    '''

    # Learning Lw with annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]
    P = np.linalg.inv(cholesky(S))
    df = np.dot(P, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P)

    return m, P

def cholesky(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha*np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                .format(os.path.basename(__file__), alpha))

class PCAW(object):
    def __init__(self, m, P, dims=None):
        '''
        m: mean matrix, numpy.ndarray, D x 1
        P: pca-whiten matrix, numpy.ndarray, D x D
        dims: principal componetns, int, d ( <= D)
        '''
        self.m = m
        self.P = P
        self.dims = dims

    def __call__(self, X, transpose=False):
        '''
        X: numpy.ndarray, D x N
        '''
        X = whitenapply(X, self.m, self.P, self.dims)
        if transpose:
            return X.T
        return X

    def __repr__(self):
        return "PCA-W [mean: {}, pcaw: {}, dims: {}]".format(
            self.m.shape, 
            self.P.shape,
            self.dims
        )