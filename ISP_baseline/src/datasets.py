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

def d_rs_to_scatter(
    d_rs: np.ndarray,
    as_real: bool=True,
    real_imag_axis: int=1,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """Sends the d_rs object to the scattered wave array, scatter
    If as_real is set to True, then the real/imag components are placed
    into different channels along a new axis, given by real_imag_axis
    """
    scatter = d_rs.transpose(0, 3, 2, 1)
    # Do I need to flip 2 and 3? Should it be like this:
    # scatter = d_rs.transpose(0, 2, 3, 1)
    dsr = downsample_ratio
    scatter = scatter[:, ::dsr, ::dsr, :]

    s_shape = scatter.shape
    scatter = scatter.reshape(
        s_shape[0],
        s_shape[1]*s_shape[2],
        s_shape[3],
    )
    if as_real:
        scatter = np.stack(
            [scatter.real, scatter.imag],
            axis=real_imag_axis,
        )
    return scatter

def q_cart_to_eta(
    q_cart: np.ndarray,
    blur_sigma: float=0.5,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """Sends q_cart to eta, and for simplicity also apply blurring here"""
    q_blurred = apply_blur_to_q(q_cart, blur_sigma=blur_sigma)
    ds_slice = np.s_[..., ::downsample_ratio, ::downsample_ratio]
    eta = q_blurred[ds_slice] # is it this simple?
    return eta

def convert_mfisnet_data_dict(
    mfisnet_dd: dict,
    blur_sigma: float=0.5,
    scatter_as_real: bool=True,
    downsample_ratio: int=1,
    real_imag_axis: int=1,
) -> dict:
    """Converts a mfisnet-style data dictionary for use with
    the wide-band equivariant network models
    """
    q_cart = mfisnet_dd["q_cart"]
    d_rs   = mfisnet_dd["d_rs"]
    eta = q_cart_to_eta(
        q_cart,
        blur_sigma=blur_sigma,
        downsample_ratio=downsample_ratio
    )
    scatter = d_rs_to_scatter(
        d_rs,
        as_real=scatter_as_real,
        real_imag_axis=real_imag_axis,
        downsample_ratio=downsample_ratio,
    )
    nx = d_rs.shape[-2] // downsample_ratio

    wb_dd = {
        "eta":     eta,
        "scatter": scatter,
        "neta":    eta.shape[-1],
        "nx":      nx,
    }
    return wb_dd

def setup_tf_dataset(
    eta: np.ndarray,
    scatter: np.ndarray,
    batch_size: int=16,
    repeats: bool=False,
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
    if repeats:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataloader = dataset.as_numpy_iterator()

    return dataset, dataloader

def get_io_mean_std(scatter: np.ndarray, eta: np.ndarray):
    """Get the input/output mean/stdev"""
    sc_axes = tuple(list(np.arange(scatter.ndim-1)))
    scatter_mean = np.mean(scatter, axis=sc_axes)
    scatter_std  = np.std(scatter, axis=sc_axes)
    eta_mean = np.mean(eta)
    eta_std  = np.std(eta)
    return scatter_mean, scatter_std, eta_mean, eta_std
