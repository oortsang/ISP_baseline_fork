import functools
import os
import shutil
import sys
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt
from clu import metric_writers
import optax
import orbax.checkpoint as ocp

jax_device = jax.devices("gpu")[0]
jax.config.update("jax_default_device", jax_device)

import argparse
import h5py
import natsort
import tensorflow as tf
from scipy.ndimage import geometric_transform
from scipy.ndimage import gaussian_filter

from ISP_baseline.src import models, trainers, utils
from ISP_baseline.models import Uncompressed

from swirl_dynamics import templates
from swirl_dynamics.lib import metrics
from pysteps.utils.spectral import rapsd

from ISP_baseline.src.data_io import (
    load_hdf5_to_dict,
    load_cart_multifreq_dataset,
    load_single_dir_slice,
    load_multi_dir_slice,
    get_multifreq_dset_dirs,
    save_single_dir_slice,
)
from ISP_baseline.src.datasets import (
    convert_mfisnet_data_dict,
    setup_tf_dataset,
    get_io_mean_std,
)
from ISP_baseline.src.more_metrics import (
    l2_error
)
from ISP_baseline.src.predictions import (
    get_loss_fns,
    aggregate_loss_vals,
    eval_model,
    save_preds_q_cart,
)

tf.config.set_visible_devices([], device_type='GPU')


# Set up logging...
import logging
FMT = "%(asctime)s:MFISNets: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]

    # Expect a single frequency but let the argument accept a list to avoid
    # needing to re-write the data loading code
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    # parser.add_argument("--data_input_nus", type=str) # actually single frequency here

    # File/directory-related arguments
    parser.add_argument(
        "--ref_data_dir_base",
        type=str,
        help="For the reference dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument(
        "--work_dir", type=str,
    )
    # parser.add_argument(
    #     "--neta", default=96, type=int
    # )
    # parser.add_argument(
    #     "--nx", default=96, type=int,
    # )
    # parser.add_argument(
    #     "--downsample_ratio", default=1, type=int,
    # )

    parser.add_argument(
        "--blur_sigma", default=0.5, type=float,
    )

    ### Training/validation-related arguments ###
    parser.add_argument("--dset_names", type=str, nargs="+")
    parser.add_argument("--truncate_nums", type=int)
    parser.add_argument("--seed", type=int, default=35675)

    parser.add_argument("--log_batch_size", type=int, default=16, help="batch size while logging")
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seeds", type=int, nargs="+")
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")

    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise


    ### Logging options ###
    parser.add_argument("--debug", default=False, action="store_true")
    # parser.add_argument("--selection_field", default="val_rel_l2")
    # parser.add_argument("--selection_mode", default="min", choices=["min", "max"])

    parser.add_argument("--output_pred_save", choices=bool_choices)
    parser.add_argument(
        "--output_pred_dir",
        type=str,
        help="target location to save the outputs if output_pred_save is set to true",
    )
    parser.add_argument(
        "--output_pred_shard_size",
        type=int,
        default=1000,
        help="specify the shard size of the outputted predictions"
    )

    # Weights and Biases setup
    # parser.add_argument("--wandb_project", type=str, help="W&B project name")
    # parser.add_argument("--wandb_entity", type=str, help="The W&B entity")
    # parser.add_argument(
    #     "--wandb_mode", choices=["offline", "online", "disabled"], default="offline"
    # )

    # Misc. options
    a = parser.parse_args()
    bool_args = [
        "use_noise_seed",
        "output_pred_save",
    ]
    # Process the boolean arguments from strings
    for bool_arg in bool_args:
        str_val = getattr(a, bool_arg)
        setattr(a, bool_arg, str_val == "true")
    return a

def kv_shrinker(key, val):
    """Little helper function to see the shapes of entires in a dictionary"""
    if isinstance(val, np.ndarray):
        if val.size > 1:
            return f"{key}<shape>", val.shape
        else:
            return key, val.item()
    elif hasattr(val, "__len__") and len(val) > 1:
        return f"{key}<len>", len(val)
    else:
        return key, val

def main(
    args: argparse.Namespace,
    # Extra arguments for testing purposes
    return_model: bool = False,
) -> None:
    """
    1. Load datasets
    2. Prepare the datasets
    3. Prepare the logging function
    4. Evaluation run; optionally write to disk
    """
    # 1. Basic setup
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    ref_data_dir_base  = args.ref_data_dir_base
    str_nu_list = (
        args.data_input_nus
    )
    kbar_str_list = str_nu_list
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"ref_data_dir_base: {ref_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # 2. Set up logging functions...


    # 3. Load NN model


    blur_sigma = args.blur_sigma


    #########################################################
    # 4. Evaluate on all the datasets, then optionally write the outputs to disk
    # 4a. load the test set and predictions
    # Common setup
    base_output_dir = args.output_pred_dir if args.output_pred_save else None


    dset_list = args.dset_names
    # expt_info_list = [pred_train_meta_dd, pred_val_meta_dd, pred_test_meta_dd]
    last_eval_dict = {}
    key_max_num_chars = max(len(key) for key in cart_loss_fn_dd.keys())
    cart_dd_list = []

    for i, dset in enumerate(dset_list):
        #########################################################
        # 4b. Load the relevant dataset
        logging.info(f"Loading {dset}...")
        truncate_num = args.truncate_nums[i]
        eff_noise_seed = args.noise_seeds[i] if args.use_noise_seed else None

        # Prepare the file directory names
        ref_dset_dirs = get_multifreq_dset_dirs(
            dset,
            kbar_str_list,
            base_dir=ref_data_dir_base,
            dir_fmt="{0}_measurements_nu_{1}"
        )
        print(f"(dset={dset}) ref dirs: {ref_dset_dirs}")

        dset_mfisnet_dd = load_cart_multifreq_dataset(
            ref_dset_dirs,
            global_idx_start=0,
            global_idx_end=NTEST,
            noise_to_sig_ratio=args.noise_to_signal_ratio,
            noise_seed=eff_noise_seed_list_test,
            noise_seed_mode="sequential",
            noise_norm_mode="inf",
        )
        print(f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in dset_mfisnet_dd.items()])}")
        dset_wb_dd = convert_mfisnet_data_dict(
            dset_mfisnet_dd,
            scatter_as_real=True,
            real_imag_axis=1,
            blur_sigma=blur_sigma,
            downsample_ratio=downsample_ratio,
        )
    
        # Try downsampling since the sparsepolartocartesian step is so slow :((
        dset_eta     = dset_wb_dd["eta"]
        dset_scatter = dset_wb_dd["scatter"]
        print(f"dset_eta     shape: {dset_eta.shape}")
        print(f"dset_scatter shape: {dset_scatter.shape}")
    
        dset_batch_size = args.log_batch_size
        dset_dataset, dset_dloader = setup_tf_dataset(
            dset_eta,
            dset_scatter,
            batch_size=dset_batch_size,
        )

if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    print(f"Received the following arguments: {a}")
    logging.info(f"Received the following arguments: {a}")
    main(a, return_model=False)
