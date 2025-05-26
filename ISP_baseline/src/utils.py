import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.ndimage import geometric_transform

def rotationindex(n):
    index = jnp.reshape(jnp.arange(0, n**2, 1), [n, n])
    return jnp.concatenate([jnp.roll(index, shift=[-i,-i], axis=[0,1]) for i in range(n)], 0)
    
def SparsePolarToCartesian(neta, nx):
    
    def CartesianToPolar(coords):
        """
        Transforms coordinates from Cartesian to polar coordinates with custom scaling.
    
        Parameters:
        - coords: A tuple or list containing the (i, j) coordinates to be transformed.
    
        Returns:
        - A tuple (rho, theta) representing the transformed coordinates.
        """
        i, j = coords[0], coords[1]
        # Calculate the radial distance with a scaling factor.
        rho = 2 * np.sqrt((i - neta / 2) ** 2 + (j - neta / 2) ** 2) * nx / neta
        # Calculate the angle in radians and adjust the scale to fit the specified range.
        theta = ((np.arctan2((neta / 2 - j), (i - neta / 2))) % (2 * np.pi)) * nx / np.pi / 2
        return theta, rho + neta // 2

    cart_mat = np.zeros((neta**2, nx, nx))

    for i in range(nx):
        for j in range(nx):
            # Create a dummy matrix with a single one at position (i, j) and zeros elsewhere.
            mat_dummy = np.zeros((nx, nx))
            mat_dummy[i, j] = 1
            # Pad the dummy matrix in polar coordinates to cover the target space in Cartesian coordinates.
            pad_dummy = np.pad(mat_dummy, ((0, 0), (neta // 2, neta // 2)), 'edge')
            # Apply the geometric transformation to map the dummy matrix to polar coordinates
            cart_mat[:, i, j] = geometric_transform(pad_dummy, CartesianToPolar, output_shape=[neta, neta], mode='grid-wrap').flatten()
    
    cart_mat = np.reshape(cart_mat, (neta**2, nx**2))
    # Removing small values
    cart_mat = np.where(np.abs(cart_mat) > 0.001, cart_mat, 0)
    
    return sparse.BCOO.fromdense(cart_mat)
    
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

