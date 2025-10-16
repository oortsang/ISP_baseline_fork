# (OOT, 2025-10-16)
# Helper functions to save make predictions and save
# them to disk


import numpy as np
import jax
import tensorflow as tf
import functools
from typing import Tuple

from ISP_baseline.src import models, trainers, utils
from ISP_baseline.models import Uncompressed

from swirl_dynamics import templates

# Metrics
from swirl_dynamics.lib import metrics
from pysteps.utils.spectral import rapsd
from ISP_baseline.src.more_metrics import (
    l2_error,
    mse_alt,
    psnr,
)

# I/O helper
from ISP_baseline.src.data_io import save_single_dir_slice

def get_loss_fns(loss_fns:list=["rrmse", "rel_l2"]):
    """Get the basic loss functions: rrmse, rel_l2 (maybe later also psnr)
    """
    all_loss_fns_dict = {
        "rrmse": functools.partial(
            # metrics.mean_squared_error,
            mse_alt,
            mse_axes=(-1,-2),
            # sum_axes=(-1, -2),
            relative=True,
            squared=False,
        ),
        "rel_l2": functools.partial(
            l2_error,
            l2_axes=(-1, -2),
            relative=True,
            squared=False,
        ),
        "psnr": functools.partial(
            psnr,
            psnr_axes=(-1, -2),
            squared=True,
            decibels=True,
        ),
    }
    loss_fn_dict = {
        key: val
        for (key, val) in all_loss_fns_dict.items()
        if key in loss_fns
    }
    return loss_fn_dict

def aggregate_loss_vals(loss_vals_dict: dict):
    """Aggregate the loss values for each sample in a dictionary of
    different loss types (given by keys)
    Returns mean/std values
    """
    agg_loss_dict = {
        **{f"{key}_mean": np.mean(val).item() for (key,val) in loss_vals_dict.items()},
        **{f"{key}_std":  np.std(val).item()  for (key,val) in loss_vals_dict.items()}
    }
    return agg_loss_dict


def eval_model(
    model_state,
    core_module,
    # dataset: tf.data.Dataset,
    dset_scatter: np.ndarray,
    dset_eta: np.ndarray,
    eval_batch_size: int,
    loss_fn_dict: dict,
    return_sample_losses: bool=True,
) -> Tuple[np.ndarray, dict]:
    """Evaluates a model
    Returns predictions and evaluated loss functions
    """
    inference_fn = trainers.DeterministicTrainer.build_inference_fn(
        model_state, core_module,
    )
    loss_vals_dict = {key: [] for key in loss_fn_dict.keys()}

    # Set up the dataset
    tmp_dataset = (
        # dataset
        tf.data.Dataset.from_tensor_slices({
            "scatter": dset_scatter,
            "eta": dset_eta,
        })
        .repeat(count=1)
        .batch(eval_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    dataloader = tmp_dataset.as_numpy_iterator()

    # Iterate through the dataset
    pred_eta_list = []
    for b, batch in enumerate(dataloader):
        batch_pred = inference_fn(batch["scatter"])
        batch_true = batch["eta"]
        # print(f"batch_pred shape: {batch_pred.shape}")
        # print(f"batch_true shape: {batch_true.shape}")
        # import pdb; pdb.set_trace()

        # Track the predictions and loss values
        pred_eta_list.append(batch_pred)
        for key, loss_fn in loss_fn_dict.items():
            loss_vals_dict[key].append(
                loss_fn(pred=batch_pred, true=batch_true)
            )

    # Flatten the predictions and loss values
    pred_eta = np.concatenate(
        pred_eta_list,
        axis=0,
    )
    loss_vals_dict = {
        key: np.concatenate(val, axis=0)
        for (key, val) in loss_vals_dict.items()
    }
    agg_loss_dict  = aggregate_loss_vals(loss_vals_dict)
    if return_sample_losses:
        # Keep the loss values per sample
        loss_vals_dict = {
            **loss_vals_dict,
            **agg_loss_vals,
        }
    else:
        # Just return the aggregate values (mean/std)
        loss_vals_dict = agg_loss_dict
    return pred_eta, loss_vals_dict

def save_preds_q_cart(
    q_cart: np.ndarray,
    x_vals: np.ndarray,
    dir_name: str,
    file_format: str="scattering_objs_{0}.h5",
    shard_size: int = 1000,
) -> list:
    """Save eta or q_cart to dict, splitting into shards
    """
    pred_dd = {
        "x_vals": x_vals,
        "sample_completion": np.ones(q_cart.shape[0], dtype=bool),
        "q_cart": q_cart,
    }
    shard_names = save_single_dir_slice(
        pred_dd,
        dir_name,
        file_format=file_format,
        shard_size=shard_size,
        sample_keys=["q_cart", "sample_completion"],
    )
    return shard_names
