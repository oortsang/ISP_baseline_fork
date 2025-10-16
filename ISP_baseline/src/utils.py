import numpy as np
import os

import jax
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.ndimage import geometric_transform

# from jax.experimental.sparse.bcoo import BCOO
from typing import Tuple
import psutil

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

### Additional helper functions ###
# Manage the cart_mat and r_index objects so we can just save them to disk
# for easier testing
def save_mats_to_fp(
    cart_mat: sparse.bcoo.BCOO,
    r_index: jnp.ndarray,
    mats_fp: str,
):
    """Save matrices to a specific file path; saves as dense matrices for easiest compatibility"""
    os.makedirs(os.path.split(mats_fp)[0], exist_ok=True)
    # # old version, which used dense representations
    # np.savez(mats_fp, cart_mat=np.array(cart_mat.todense()), r_index=np.array(r_index))
    cart_mat_data = cart_mat.data
    cart_mat_idcs = cart_mat.indices
    np.savez(
        mats_fp,
        cart_mat_data=cart_mat_data,
        cart_mat_idcs=cart_mat_idcs,
        cart_mat_shape=cart_mat.shape,
        r_index=np.array(r_index),
    )

def load_mats_from_fp(
    mats_fp: str,
) -> Tuple[sparse.bcoo.BCOO, jnp.ndarray]:
    """Load matrices from a specific file path; raises an error if the file does not exist"""
    mats_dd = np.load(mats_fp)
    cart_mat_data  = mats_dd["cart_mat_data"]
    cart_mat_idcs  = mats_dd["cart_mat_idcs"]
    cart_mat_shape = tuple(mats_dd["cart_mat_shape"])
    r_index_np     = mats_dd["r_index"]
    # print(f"Loaded matrices from the file!")

    cart_mat = sparse.bcoo.BCOO((cart_mat_data, cart_mat_idcs), shape=cart_mat_shape)
    r_index  = jnp.array(r_index_np)
    return cart_mat, r_index

def _get_mats_fp(
    neta: int,
    nx: int,
    mats_dir: str=None,
    mats_format: str=None,
) -> str:
    """Gets the matrices' filepath from basic information"""
    mats_format = mats_format if mats_format is not None else "mats_neta{0}_nx{1}.npz"
    mats_fp = os.path.join(mats_dir, mats_format.format(neta,nx)) if mats_dir is not None else None
    return mats_fp

def save_mats(
    neta: int,
    nx: int,
    mats_dir: str=None,
    mats_format: str=None,
):
    """Save matrices to disk, wrapper that handles file naming
    """
    mats_fp = _get_mats_fp(neta, nx, mats_dir, mats_format)
    save_mats_to_fp(cart_mat, r_index, mats_fp)

def load_or_create_mats(
    neta: int,
    nx: int,
    mats_dir: str=None,
    mats_format: str=None,
    save_if_created: bool=False,
) -> Tuple[sparse.bcoo.BCOO, jnp.ndarray]:
    """Loads the matrices if possible, otherwise creates new ones"""
    mats_fp = _get_mats_fp(neta, nx, mats_dir, mats_format)

    loaded_mats = False
    if mats_fp is not None or not os.path.exists(mats_fp):
        try:
            cart_mat, r_index = load_mats_from_fp(mats_fp)
            loaded_mats = True
        except:
            pass
    if not loaded_mats:
        cart_mat = SparsePolarToCartesian(neta, nx)
        r_index  = rotationindex(nx)

        # Save to disk if we just created new matrices
        if save_if_created:
            save_mats_to_fp(cart_mat, r_index, mats_fp)

    return cart_mat, r_index

### VRAM / RAM helper functions ###
def get_memory_info_jax(device=None, print_msg: bool=True):
    """Helper function that tells the RAM and VRAM usage"""
    msg_1  = f"RAM Used (MB): {psutil.virtual_memory().used >> 20}"
    device = (
        device if device is not None else
        jax.config.jax_default_device
    )
    if device is None or device.device_kind == "cpu":
        if not print_msg:
            print(msg_1)
        return msg_1
    stats = device.memory_stats()
    # First values are in bytes
    t = stats["bytes_limit"] # total (within preallocation)
    u = stats["bytes_in_use"]
    p = stats["peak_bytes_in_use"]
    f = t-u # free (within preallocation)
    msg_2 = (
        f"VRAM (MB): {f>>20} free of {t>>20} total (within preallocation); "
        f"usage is currently {u>>20} and peaked at {p>>20}"
    )
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    # msg_2 = f"VRAM (MB): {f>>20} free of {r>>20} reserved; {a>>20} allocated out of {t>>20} total"
    msg_full = msg_1 + "\n" + msg_2
    if print_msg:
        print(msg_full)
    return msg_full

def get_vram_total_mb_jax(device=None):
    """Helper function that gets total VRAM amount"""
    device = (
        device if device is not None else
        jax.config.jax_default_device
    )
    tot_mb = device.memory_stats["bytes_in_use"] >> 20
    return tot_mb

def vram_mb_to_frac_jax(block_mb: float, device=None) -> float:
    """Helper function that takes an amount of vram in megabytes
    and returns what fraction of total VRAM that would be
    """
    tot_mb = get_vram_total_mb_jax(device)
    return (block_mb / tot_mb) if tot_mb != 0 else 0

