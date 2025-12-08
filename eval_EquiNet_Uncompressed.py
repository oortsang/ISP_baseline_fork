# Evaluation script
# (OOT, 2025-11-02) I'm not sure if noise seeds are handled correctly
# (caveat emptor)

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
        "--model_dir", type=str,
    )
    parser.add_argument(
        "--model_step", type=int,
    )
    parser.add_argument(
        "--neta", default=96, type=int
    )
    parser.add_argument(
        "--nx", default=96, type=int,
    )
    parser.add_argument(
        "--downsample_ratio", default=1, type=int,
    )

    parser.add_argument(
        "--blur_sigma", default=0.0, type=float,
    )
    parser.add_argument("--blur_test", choices=bool_choices, default="false")

    ### Training/validation-related arguments ###
    parser.add_argument("--dset_names", type=str, nargs="+")

    parser.add_argument("--truncate_nums", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=35675)

    parser.add_argument("--log_batch_size", type=int, default=16, help="batch size while logging")
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seeds", type=int, nargs="+")
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")
    parser.add_argument(
        "--noise_to_signal_ratio", default=0.0, type=float
    )  # train and test with noise

    # Architecture
    parser.add_argument("--n_cnn_layers_2d",   type=int, default=9)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=6)
    parser.add_argument("--kernel_size_2d", type=int, default=5)
    parser.add_argument("--grad_checkpoint", choices=bool_choices, default="false")

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
        "blur_test",
        "grad_checkpoint",
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
    nk = N_freqs
    logging.info(f"ref_data_dir_base: {ref_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # 2. Set up logging functions...

    # 3. Load NN model
    downsample_ratio = args.downsample_ratio
    neta = args.neta // downsample_ratio
    nx   = args.nx   // downsample_ratio

    blur_sigma = args.blur_sigma
    model_dir = os.path.join(os.path.abspath(''), args.model_dir)
    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        # f"{model_dir}/checkpoints", step=None,
        f"{model_dir}/checkpoints", step=args.model_step,
    )
    N_cnn_layers = args.n_cnn_layers_2d
    N_cnn_channels = args.n_cnn_channels_2d
    kernel_size = args.kernel_size_2d
    cart_mat, r_index = utils.load_or_create_mats(
        neta,
        nx,
        mats_dir=os.path.join("tmp", "cart_and_rot_mats"),
        mats_format="mats_neta{0}_nx{1}.npz",
        save_if_created=True,
    )
    core_module = Uncompressed.UncompressedModelFlexible(
        nx = nx,
        neta = neta,
        cart_mat = cart_mat,
        r_index = r_index,
        # New parameters
        nk=nk,
        N_cnn_layers=N_cnn_layers,
        N_cnn_channels=N_cnn_channels,
        kernel_size=kernel_size,
        grad_checkpoint=args.grad_checkpoint,
        # I/O normalization
        in_norm=False,
        out_norm=False,
    )


    #########################################################
    # 4. Evaluate on all the datasets, then optionally write the outputs to disk
    # 4a. load the test set and predictions
    # Common setup
    base_output_dir = args.output_pred_dir if args.output_pred_save else None

    dset_list = args.dset_names
    loss_fn_dict = get_loss_fns(["rrmse", "rel_l2", "psnr"])
    all_loss_strs = {loss_name: f"" for loss_name in loss_fn_dict.keys()}

    # expt_info_list = [pred_train_meta_dd, pred_val_meta_dd, pred_test_meta_dd]
    # last_eval_dict = {}
    # key_max_num_chars = max(len(key) for key in loss_fn_dict.keys())
    # dd_list = []
    prev_dd = None
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
        logging.info(f"(dset={dset}) ref dirs: {ref_dset_dirs}")

        if (i>0 and dset == dset_list[i-1]
            and truncate_num == args.truncate_nums[i-1]
            ):
            logging.info(f"Reusing data from the last dset")
            dset_mfisnet_dd = prev_dd
        else:
            dset_mfisnet_dd = load_cart_multifreq_dataset(
                ref_dset_dirs,
                global_idx_start=0,
                global_idx_end=truncate_num,
                noise_to_sig_ratio=args.noise_to_signal_ratio,
                noise_seed=eff_noise_seed,
                noise_seed_mode="sequential",
                noise_norm_mode="inf",
            )
        prev_dd = dset_mfisnet_dd
        x_vals = dset_mfisnet_dd["x_vals"]
        logging.info(
            f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in dset_mfisnet_dd.items()])}"
        )
        dset_wb_dd = convert_mfisnet_data_dict(
            dset_mfisnet_dd,
            scatter_as_real=True,
            real_imag_axis=1,
            blur_sigma=blur_sigma,
            downsample_ratio=downsample_ratio,
            flip_scobj_axes=False,
        )

        dset_eta     = dset_wb_dd["eta"]
        dset_scatter = dset_wb_dd["scatter"]
        logging.info(f"dset_eta     shape: {dset_eta.shape}")
        logging.info(f"dset_scatter shape: {dset_scatter.shape}")

        dset_batch_size = args.log_batch_size
        # dset_dataset, dset_dloader = setup_tf_dataset(
        #     dset_eta,
        #     dset_scatter,
        #     batch_size=dset_batch_size,
        # )

        # Evaluate and save to disk
        logging.info(f"{dset}_scatter shape: {dset_scatter.shape}")
        logging.info(f"{dset}_eta shape:     {dset_eta.shape}")
        t0 = time.perf_counter()
        return_sample_losses = True
        dset_preds, dset_loss_vals = eval_model(
            trained_state,
            core_module,
            # dataset,
            dset_scatter=dset_scatter,
            dset_eta=dset_eta,
            eval_batch_size=args.log_batch_size,
            loss_fn_dict=loss_fn_dict,
            return_sample_losses=return_sample_losses,
        )
        t1 = time.perf_counter()
        logging.info(f"dset {dset} was evaluated in {t1-t0:.3f} seconds")
        if return_sample_losses:
            logging.info(f"Sample losses: {dset_loss_vals['rel_l2']}")
        else:
            logging.info(f"dset {dset} losses: {dset_loss_vals}")
        for loss_name in loss_fn_dict.keys():
            loss_mean = dset_loss_vals[f"{loss_name}_mean"]
            loss_std  = dset_loss_vals[f"{loss_name}_std"]
            delim_str = " " if i > 0 else ""
            all_loss_strs[loss_name] += (
                f"{delim_str}{loss_mean:.6e}±{loss_std:.4e}"
            )
        if args.output_pred_save:
            logging.info(f"Saving predictions to disk")
            dset_output_pred_dir = os.path.join(
                args.output_pred_dir,
                f"{dset}_scattering_objs"
            )
            save_preds_q_cart(
                dset_preds,
                x_vals,
                dset_output_pred_dir,
                file_format="scattering_objs_{0}.h5",
                shard_size=args.output_pred_shard_size,
            )
        else:
            logging.info(f"Not saving predictions to disk")
    for loss_name in loss_fn_dict.keys():
        logging.info(f"Overall {loss_name}: {all_loss_strs[loss_name]}")
    vram_msg = utils.get_memory_info_jax(jax_device, print_msg=False)
    logging.info(f"After evaluation: {vram_msg}")
    logging.info("Bye!")
    print("Bye!")



if __name__ == "__main__":
    a = setup_args()

    # Logging settings from ChatGPT:
    logging.basicConfig(
        format=FMT, datefmt=TIMEFMT,
        level=logging.DEBUG if a.debug else logging.INFO,
        force=True,
    )
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger("jaxlib").setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    # Many JAX messages go through absl; set its verbosity too
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.WARNING)
    except Exception:
        pass

    # for name, logger in logging.root.manager.loggerDict.items():
    #     logging.getLogger(name).setLevel(logging.WARNING)

    # if a.debug:
    #     logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    # else:
    #     logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    print(f"Received the following arguments: {a}")
    logging.info(f"Received the following arguments: {a}")
    main(a, return_model=False)
