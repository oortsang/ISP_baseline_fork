import functools
import shutil
import os
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
from ISP_baseline.models import Uncompressed, Compressed

from swirl_dynamics import templates
from swirl_dynamics.lib import metrics
from pysteps.utils.spectral import rapsd

from ISP_baseline.src.data_io import (
    load_hdf5_to_dict,
    load_cart_multifreq_dataset,
    load_single_dir_slice,
    load_multi_dir_slice,
    get_multifreq_dset_dirs,
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
        "--quadtree_l", default=4, type=int
    )
    parser.add_argument(
        "--quadtree_s", default=12, type=int,
    )
    # parser.add_argument(
    #     "--quadtree_r", default=3, type=int,
    # )
    tmp_seed = int(float(str(time.perf_counter())[::-1]))
    rng = np.random.default_rng(tmp_seed)
    tag = "0x"+"".join([hex(x)[2:] for x in rng.integers(256, size=6)])
    date_str = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    parser.add_argument(
        "--work_dir", default=f"tmp/{date_str}_tmp_workdir_{tag}", type=str
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
        "--blur_sigma", default=0.5, type=float,
    )

    ### Training/validation-related arguments ###
    parser.add_argument("--truncate_num_train", type=int)
    parser.add_argument("--truncate_num_val", type=int)
    parser.add_argument("--truncate_num_test", type=int)
    parser.add_argument("--seed", type=int, default=35675)
    # parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    # parser.add_argument("--noise_seed_train",  type=int, default=10128329)
    # parser.add_argument("--noise_seed_val",    type=int, default=20293834)
    # parser.add_argument("--noise_seed_test",   type=int, default=30943792)
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seed_list_train",  nargs="*", type=int)
    parser.add_argument("--noise_seed_list_val",    nargs="*", type=int)
    parser.add_argument("--noise_seed_list_test",   nargs="*", type=int)
    parser.add_argument("--n_resnet_layers",   type=int, default=6)
    parser.add_argument("--n_resnet_channels",   type=int, default=3)
    parser.add_argument("--n_cnn_layers_2d",   type=int, default=3)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=6)
    parser.add_argument("--kernel_size_2d", type=int, default=5)
    parser.add_argument("--grad_checkpoint", choices=bool_choices, default="true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_batch_size", type=int, default=100, help="batch size while logging")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=1e-5)
    parser.add_argument("--io_norm", choices=bool_choices, default="false")
    parser.add_argument("--n_epochs_per_log", type=int, default=5) # currently unused
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

    # Processing for ease of use
    # if a.use_targets == "smoothed":
    #     a.use_smoothed_targets = True
    # elif a.use_targets == "original":
    #     a.use_smoothed_targets = False
    # a.freq_idx = a.freq_idx if a.freq_idx is not None else a.freq_lvl

    bool_args = [
        "use_noise_seed",
        "output_pred_save",
        "io_norm",
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
    Train the Compressed B-EquiNet model
    """
    np.random.seed(args.seed)
    print(f"Selecting work dir: {args.work_dir}")

    # Get noise seeds
    if args.use_noise_seed:
        eff_noise_seed_list_train = args.noise_seed_list_train
        eff_noise_seed_list_val   = args.noise_seed_list_val
        eff_noise_seed_list_test  = args.noise_seed_list_test
    else:
        eff_noise_seed_list_train = None
        eff_noise_seed_list_val   = None
        eff_noise_seed_list_test  = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_list_train} for the training set")
        logging.info(f"Using seed as {eff_noise_seed_list_val} for the val set")
        logging.info(f"Using seed as {eff_noise_seed_list_test} for the test set")
    else:
        logging.info(f"Not adding noise!")


    # Grab settings from arguments
    L = args.quadtree_l
    s = args.quadtree_s
    # r = args.quadtree_r
    downsample_ratio = args.downsample_ratio
    s = s // downsample_ratio
    neta = (2**L) * s
    nx = (2**L) * s
    blur_sigma = args.blur_sigma

    kbar_str_list = args.data_input_nus
    nk = len(kbar_str_list)
    NTRAIN = args.truncate_num_train
    NVAL   = args.truncate_num_val
    NTEST  = args.truncate_num_test
    vram_msg = utils.get_memory_info_jax(jax_device, print_msg=False)
    print(f"Before loading data: {vram_msg}")

    # Load the dataset
    train_dirs = get_multifreq_dset_dirs(
        "train",
        kbar_str_list,
        base_dir=args.ref_data_dir_base,
        dir_fmt="{0}_measurements_nu_{1}"
    )

    train_mfisnet_dd = load_cart_multifreq_dataset(
        train_dirs,
        global_idx_start=0,
        global_idx_end=NTRAIN,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        noise_seed=eff_noise_seed_list_train,
        noise_seed_mode="sequential",
        noise_norm_mode="inf",
    )
    print(f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in train_mfisnet_dd.items()])}")
    train_wb_dd = convert_mfisnet_data_dict(
        train_mfisnet_dd,
        blur_sigma=blur_sigma,
        scatter_as_real=True,
        real_imag_axis=2,
        downsample_ratio=downsample_ratio,
        flip_scobj_axes=True,
    )

    train_eta = train_wb_dd["eta"]
    train_scatter = train_wb_dd["scatter"]
    print(f"train_eta     shape: {train_eta.shape}")
    print(f"train_scatter shape: {train_scatter.shape}")
    (
    train_scatter_mean,
        train_scatter_std,
        train_eta_mean,
        train_eta_std
    ) = get_io_mean_std(train_scatter, train_eta)

    train_dataset, train_dloader = setup_tf_dataset(
        train_eta,
        train_scatter,
        batch_size=args.batch_size,
        repeats=True,
    )

    val_dirs = get_multifreq_dset_dirs(
        "val",
        kbar_str_list,
        base_dir=args.ref_data_dir_base,
        dir_fmt="{0}_measurements_nu_{1}"
    )
    val_mfisnet_dd = load_cart_multifreq_dataset(
        val_dirs,
        global_idx_start=0,
        global_idx_end=NVAL,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        noise_seed=eff_noise_seed_list_val,
        noise_seed_mode="sequential",
        noise_norm_mode="inf",
    )
    val_wb_dd = convert_mfisnet_data_dict(
        val_mfisnet_dd,
        blur_sigma=blur_sigma,
        scatter_as_real=True,
        real_imag_axis=2,
        downsample_ratio=downsample_ratio,
        flip_scobj_axes=True,
    )
    # Try downsampling since the sparsepolartocartesian step is so slow :((
    val_eta     = val_wb_dd["eta"]
    val_scatter = val_wb_dd["scatter"]

    val_dataset, val_dloader = setup_tf_dataset(
        val_eta,
        val_scatter,
        batch_size=args.log_batch_size,
        repeats=False,
    )
    _, val_dloader_looped = setup_tf_dataset(
        val_eta,
        val_scatter,
        batch_size=args.log_batch_size,
        repeats=True,
    )

    print(f"Setting up the model...")
    N_resnet_layers   = args.n_resnet_layers
    N_resnet_channels = args.n_resnet_channels
    N_cnn_layers      = args.n_cnn_layers_2d
    N_cnn_channels    = args.n_cnn_channels_2d
    kernel_size       = args.kernel_size_2d
    cart_mat, r_index = utils.load_or_create_mats(
        neta,
        nx,
        mats_dir=os.path.join("tmp", "cart_and_rot_mats"),
        mats_format="mats_neta{0}_nx{1}.npz",
        save_if_created=True,
    )
    print(f"cart_mat shape: {cart_mat.shape}")
    print(f"r_index  shape: {r_index.shape}")
    print(f"Created or loaded cart_mat")
    hyperparam_dict = {
        "L": L,
        "s": L,
        # "nx": nx,
        # "neta": neta,
        "nk": nk,
        "N_resnet_layers": N_resnet_layers,
        "N_resnet_channels": N_resnet_channels,
        "N_cnn_layers": N_cnn_layers,
        "N_cnn_channels": N_cnn_channels,
        "kernel_size": kernel_size,
        "lr_init": args.lr_init,
    }
    logging.info(f"Received hyperparameters: {hyperparam_dict}")
    # core_module = Compressed.CompressedModel(
    #     L=L,
    #     s=s,
    #     r=N_resnet_channels,
    #     cart_mat=cart_mat,
    #     r_index=r_index,
    #     NUM_RESNET=N_resnet_layers,
    #     NUM_CONV=N_cnn_layers,
    # )
    core_module = Compressed.CompressedModelFlexible(
        L=L,
        s=s,
        r=N_resnet_channels,
        cart_mat=cart_mat,
        r_index=r_index,
        # New parameters
        nk=nk,
        N_resnet_layers=N_resnet_layers,
        N_cnn_layers=N_cnn_layers,
        N_cnn_channels=N_cnn_channels,
        kernel_size=kernel_size,
        grad_checkpoint=args.grad_checkpoint,
        # NUM_RESNET=6,
        # NUM_CONV=9,
        # I/O normalization
        in_norm=args.io_norm,
        out_norm=args.io_norm,
        in_mean=jnp.array(train_scatter_mean),
        in_std=jnp.array(train_scatter_std),
        out_mean=jnp.array(train_eta_mean),
        out_std=jnp.array(train_eta_std),
    )

    print(f"nx: {nx}; neta: {neta}")
    print(f"Input shape: {train_scatter[0].shape}")
    Model = models.DeterministicModel(
        input_shape = train_scatter[0].shape,
        core_module = core_module
    )
    rng = jax.random.PRNGKey(888)
    params = Model.initialize(rng)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print('Number of trainable parameters:', param_count)

    print(f"Training...")
    num_train_steps = NTRAIN * args.n_epochs // 16  #@param
    workdir = os.path.join(os.path.abspath(''), args.work_dir)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    initial_lr = args.lr_init #@param
    peak_lr = 5e-3 #@pawram
    warmup_steps = num_train_steps // 20  #@param
    end_lr = 1e-8 #@param
    ckpt_interval = 2000  #@param
    max_ckpt_to_keep = 3  #@param

    trainer = trainers.DeterministicTrainer(
        model=Model,
        rng=jax.random.PRNGKey(42),
        optimizer=optax.adam(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=initial_lr,
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_train_steps,
                end_value=end_lr,
            ),
        ),
    )
    vram_msg = utils.get_memory_info_jax(jax_device, print_msg=False)
    print(f"Before loading data: {vram_msg}")

    # eval_dloader = train_dloader
    # val_dloader = train_dloader
    try:
        templates.run_train(
            train_dataloader=train_dloader,
            trainer=trainer,
            workdir=workdir,
            total_train_steps=num_train_steps,
            metric_writer=metric_writers.create_default_writer(
                workdir, asynchronous=False
            ),
            metric_aggregation_steps=10,
            eval_dataloader=val_dloader_looped,
            eval_every_steps = 100,
            num_batches_per_eval = 1,
            callbacks=(
                templates.TqdmProgressBar(
                    total_train_steps=num_train_steps,
                    train_monitors=("train_loss",),
                    eval_monitors=("eval_rel_l2_mean",),
                ),
                templates.TrainStateCheckpoint(
                    base_dir=workdir,
                    options=ocp.CheckpointManagerOptions(
                        save_interval_steps=ckpt_interval, max_to_keep=max_ckpt_to_keep
                    ),
                ),
            ),
        )

    except Exception as e:
        logging.error(f"Exception: {e}")
        vram_msg = utils.get_memory_info_jax(jax_device, print_msg=False)
        print(f"{vram_msg}")
        sys.exit(1)
    vram_msg = utils.get_memory_info_jax(jax_device, print_msg=False)
    print(f"After training: {vram_msg}")


    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{workdir}/checkpoints", step=None
    )

    inference_fn = trainers.DeterministicTrainer.build_inference_fn(
        trained_state, core_module
    )

    test_dirs = get_multifreq_dset_dirs(
        "test",
        kbar_str_list,
        base_dir=args.ref_data_dir_base,
        dir_fmt="{0}_measurements_nu_{1}"
    )

    test_mfisnet_dd = load_cart_multifreq_dataset(
        test_dirs,
        global_idx_start=0,
        global_idx_end=NTEST,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        noise_seed=eff_noise_seed_list_test,
        noise_seed_mode="sequential",
        noise_norm_mode="inf",
    )
    print(f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in test_mfisnet_dd.items()])}")
    test_wb_dd = convert_mfisnet_data_dict(
        test_mfisnet_dd,
        scatter_as_real=True,
        real_imag_axis=2,
        blur_sigma=blur_sigma,
        downsample_ratio=downsample_ratio,
        flip_scobj_axes=True,
    )

    # Try downsampling since the sparsepolartocartesian step is so slow :((
    test_eta     = test_wb_dd["eta"]
    test_scatter = test_wb_dd["scatter"]

    test_batch_size = args.log_batch_size
    test_dataset, test_dloader = setup_tf_dataset(
        test_eta,
        test_scatter,
        batch_size=test_batch_size,
    )

    x_vals = train_mfisnet_dd["x_vals"]
    loss_fn_dict = get_loss_fns(["rrmse", "rel_l2", "psnr"])
    dset_name_list = ["train", "val", "test"]
    # dataset_list = [train_dataset, val_dataset, test_dataset]
    dataset_list = [
        (train_scatter, train_eta),
        (val_scatter,   val_eta),
        (test_scatter,  test_eta),
    ]
    all_loss_strs = {loss_name: f"" for loss_name in loss_fn_dict.keys()}

    for i, dset in enumerate(dset_name_list):
        # dataset = dataset_list[i]
        dset_scatter, dset_eta = dataset_list[i]
        print(f"{dset}_scatter shape: {dset_scatter.shape}")
        print(f"{dset}_eta shape:     {dset_eta.shape}")
        dset_preds, dset_loss_vals = eval_model(
            trained_state,
            core_module,
            # dataset,
            dset_scatter=dset_scatter,
            dset_eta=dset_eta,
            eval_batch_size=args.log_batch_size,
            loss_fn_dict=loss_fn_dict,
            return_sample_losses=False
        )
        print(f"dset {dset} losses: {dset_loss_vals}")
        for loss_name in loss_fn_dict.keys():
            loss_mean = dset_loss_vals[f"{loss_name}_mean"]
            loss_std  = dset_loss_vals[f"{loss_name}_std"]
            delim_str = " " if i > 0 else ""
            all_loss_strs[loss_name] += f"{delim_str}{loss_mean:.6e}±{loss_std:.4e}"

        if args.output_pred_save:
            print(f"Saving predictions to disk")
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
            print(f"Not saving predictions to disk")

    # for loss_name in loss_fn_dict.keys():
    #     print(f"Overall {loss_name}: {all_loss_strs[loss_name]}")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    # # Somehow this double-prints the log entries
    # root  = logging.getLogger()
    # handler = logging.StreamHandler(sys.stderr)
    # if a.debug:
    #     handler.level = logging.DEBUG
    #     root.setLevel(logging.DEBUG)
    # else:
    #     handler.level = logging.WARNING
    #     root.setLevel(logging.WARNING)

    # formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    # handler.setFormatter(formatter)
    # root.addHandler(handler)

    print(f"Received the following arguments: {a}")
    logging.info(f"Received the following arguments: {a}")
    main(a, return_model=False)
