import numpy as np

def morton_to_flatten_indices(L, s, b_flatten=True):
    """ Permutes a morton-flattened vector to python-flattened vector,
    e.g.
        >> ind = morton_to_flatten_indices(L, s)
        >> X.flatten() == morton_flatten(X, L, s)[ind]
    """
    if L==0:
        return np.arange(s**2).reshape(s,s)
    else:
        blk = 4**(L-1)*s*s

        tmp = morton_to_flatten_indices(L-1,s, b_flatten=False)

        tmp1 = np.hstack((tmp, blk+tmp))
        tmp2 = np.hstack((2*blk+tmp, 3*blk+tmp))

        if b_flatten:
            return np.vstack((tmp1,tmp2)).flatten()
        else:
            return np.vstack((tmp1,tmp2))

def flatten_to_morton_indices(L, s):
    """ Permutes a python-flattened vector to morton-flattened vector,
    e.g.
        >> X = np.random.randn((2**L)*s, (2**L)*s)
        >> idx = flatten_to_morton_indices(L,s)
        >> X.flatten()[idx] == morton_flatten(X, L, s)
    """
    nx =  (2**L)*s
    X = np.arange(nx*nx).reshape(nx, nx)
    return morton_flatten(X, L, s)

def morton_flatten(x, L, s):
    """ Flatten via Z-ordering a (2^L)s by (2^L)s dimensional matrix. 
    """
    assert x.shape[0] == (2**L)*s
    assert x.shape[1] == (2**L)*s

    if L == 0:
        return x.flatten()
    else:
        blk = 2**(L-1)*s
        return np.hstack((morton_flatten(x[0:blk, 0:blk], L-1,s),
            morton_flatten(x[0:blk, blk:(2*blk)], L-1,s),
            morton_flatten(x[blk:(2*blk), 0:blk], L-1,s),
            morton_flatten(x[blk:(2*blk), blk:(2*blk)], L-1,s)))

def morton_reshape(x, L, s):
    """ Reassembles morton flattened vector into matrix.
    """
    assert x.shape[0] == (4**L)*s*s

    if L == 0:
        return x.reshape(s,s)
    else:
        blk = 4**(L-1)*s*s

        tmp1 = np.hstack((morton_reshape(x[0:blk], L-1, s),
                morton_reshape(x[blk:(2*blk)], L-1, s)))

        tmp2 = np.hstack((morton_reshape(x[(2*blk):(3*blk)], L-1, s),
                morton_reshape(x[(3*blk):(4*blk)], L-1, s)))

        return np.vstack((tmp1, tmp2))

if __name__=='__main__':

    L = 3
    s = 2
    sz = (2**L)*s
#     Y = np.random.randn(sz, sz)
#     print(morton_reshape(morton_flatten(Y,L,s),L,s)-Y)

    # given a morton flattened array
    Xin = np.random.randn(sz, sz)
    X_morton = morton_flatten(Xin, L, s)

    ind = morton_to_flatten_indices(L,s)[:]
    print(X_morton[ind].reshape(sz,sz)-Xin)



