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
    
