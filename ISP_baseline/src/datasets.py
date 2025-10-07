# datasets.py
# Convert dictionary-type representations of the
# datasets from mfisnet-style to wide-band model style
# Loosely modeled off the example jupyter notebooks
# Added by Olivia Tsang

import numpy as np
import tensorflow as tf

from scipy.ndimage import gaussian_filter

def apply_blur_to_q(
    q: np.ndarray,
    blur_sigma: float=0.5,
) -> np.ndarray:
    """Applies a gaussian blur of the specified level to a given
    scattering potential q, given in cartesian coordinates as q_cart
    q_cart expected shape: (N_batch, N_x, N_x)
    """
    if blur_sigma == 0:
        return q

    N_samples = q.shape[0]
    blur_fn = lambda x: gaussian_filter(x, sigma=blur_sigma)
    q_blurred = np.stack(
        [
            blur_fn(q[i, :, :].T)
            for i in range(N_samples)
        ],
        axis=0,
    ).astype('float32')
    return q_blurred

def d_rs_to_scatter(d_rs: np.ndarray) -> np.ndarray:
    """Sends the d_rs object to the scattered wave array, scatter"""
    scatter = d_rs.transpose(0, 3, 2, 1)
    return scatter

def q_cart_to_eta(q_cart: np.ndarray, blur_sigma: float=0.5) -> np.ndarray:
    """Sends q_cart to eta, and for simplicity also apply blurring here"""
    q_blurred = apply_blur_to_q(q_cart, blur_sigma=blur_sigma)
    eta = q_blurred # is it this simple?
    return eta

def convert_mfisnet_data_dict(mfisnet_dd: dict, blur_sigma: float=0.5) -> dict:
    """Converts a mfisnet-style data dictionary for use with
    the wide-band equivariant network models
    """
    q_cart = mfisnet_dd["q_cart"]
    d_rs   = mfisnet_dd["d_rs"]

    wb_dd = {
        "eta": q_cart_to_eta(q_cart, blur_sigma=blur_sigma),
        "scatter": d_rs_to_scatter(d_rs),
        "neta": q_cart.shape[-1],
        "nx": d_rs.shape[-1],
    }
    return wb_dd

def setup_tf_dataset(
    eta: np.ndarray,
    scatter: np.ndarray,
    batch_size: int=16,
) -> tf.data.Dataset:
    """Sets up the tensorflow-type dataset
    Assumes data_dd is the type used for wide-band equivariant models
    This comes from the jupyter notebook
    """
    # Load the scattered wave data, with frequency in the last dimension
    # data_obj = (scatter, eta)
    data_obj = {
        "eta": eta,
        "scatter": scatter,
    }

    dataset = tf.data.Dataset.from_tensor_slices(data_obj)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataloader = dataset.as_numpy_iterator()

    return dataset, dataloader
