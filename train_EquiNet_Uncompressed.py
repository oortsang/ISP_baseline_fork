import functools
import os
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt
from clu import metric_writers
import optax
import orbax.checkpoint as ocp

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
)
from ISP_baseline.src.datasets import (
    convert_mfisnet_data_dict,
    setup_tf_dataset,
)
from ISP_baseline.src.more_metrics import (
    l2_error
)

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
    # parser.add_argument(
    #     "--quadtree_l", default=4, type=int
    # )
    # parser.add_argument(
    #     "--quadtree_s", default=12, type=int,
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
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seed_train",  type=int, default=10128329)
    parser.add_argument("--noise_seed_val",    type=int, default=20293834)
    parser.add_argument("--noise_seed_test",   type=int, default=30943792)
    parser.add_argument("--n_cnn_layers_2d",   type=int, default=3)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=6)
    parser.add_argument("--kernel_size_2d", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_batch_size", type=int, default=100, help="batch size while logging")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr_init_base", type=float, default=3e-4)
    parser.add_argument("--eta_min_base", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay_base", type=float, default=0.0)
    parser.add_argument("--n_epochs_per_log", type=int, default=5)
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
    Train the Uncompressed EquiNet model
    """

    # Set seeds for reproducibility
    np.random.seed(args.seed)

    # Grab settings from arguments
    # L = args.quadtree_l
    # s = args.quadtree_s
    # neta = (2**L) * s
    # nx = (2**L) * s
    downsample_ratio = args.downsample_ratio
    neta = args.neta // downsample_ratio
    nx   = args.nx   // downsample_ratio
    blur_sigma = args.blur_sigma

    kbar_str_list = args.data_input_nus
    nk = len(kbar_str_list)
    NTRAIN = args.truncate_num_train
    NVAL   = args.truncate_num_val
    NTEST  = args.truncate_num_test
    # NVAL   = args.truncate_num_val
    # NTEST  = args.truncate_num_test

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
    )
    print(f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in train_mfisnet_dd.items()])}")
    train_wb_dd = convert_mfisnet_data_dict(
        train_mfisnet_dd,
        blur_sigma=blur_sigma,
        downsample_ratio=downsample_ratio,
    )

    train_eta = train_wb_dd["eta"]
    train_scatter = train_wb_dd["scatter"]
    print(f"train_eta     shape: {train_eta.shape}")
    print(f"train_scatter shape: {train_scatter.shape}")
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
    )
    val_wb_dd = convert_mfisnet_data_dict(
        val_mfisnet_dd,
        blur_sigma=blur_sigma,
        downsample_ratio=downsample_ratio,
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
    print(f"Created or loaded cart_mat")
    hyperparam_dict ={
        "nx": nx,
        "neta": neta,
        "nk": nk,
        "N_cnn_layers": N_cnn_layers,
        "N_cnn_channels": N_cnn_channels,
        "kernel_size": kernel_size,
    }
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
    epochs = 100
    num_train_steps = NTRAIN * args.n_epochs // 16  #@param
    # workdir = os.path.abspath('') + "/tmp/Uncompressed10squaresDev"  #@param
    workdir = os.path.join(os.path.abspath(''), args.work_dir)
    initial_lr = 1e-5 #@param
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
    # print(f"trainer has: {trainer.__dir__()}")
    # import pdb; pdb.set_trace()
    # eval_dloader = train_dloader
    # val_dloader = train_dloader
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
                # eval_monitors=("eval_rrmse_mean",),
            ),
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=ckpt_interval, max_to_keep=max_ckpt_to_keep
                ),
            ),
        ),
    )
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
    )
    print(f"Loaded: {', '.join([f'{key}{val.shape}' for (key, val) in test_mfisnet_dd.items()])}")
    test_wb_dd = convert_mfisnet_data_dict(
        test_mfisnet_dd,
        blur_sigma=blur_sigma,
        downsample_ratio=downsample_ratio,
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

    val_errors_rrmse = []
    val_errors_rel_l2 = []
    val_errors_rapsd = []
    pred_eta = np.zeros(test_eta.shape)

    rrmse = functools.partial(
        metrics.mean_squared_error,
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )
    rel_l2 = functools.partial(
        l2_error,
        l2_axes=(-1, -2),
        relative=True,
        squared=False,
    )

    for b, batch in enumerate(val_dloader):
        pred = inference_fn(batch["scatter"])
        start_idx = b*test_batch_size
        end_idx   = min((b+1)*test_batch_size, pred_eta.shape[0])
        pred_eta[start_idx:end_idx, :, :] = pred
        true = batch["eta"]
        val_errors_rrmse.append(rrmse(pred=pred, true=true))
        val_errors_rel_l2.append(rel_l2(pred=pred, true=true))
        for i in range(true.shape[0]):
            val_errors_rapsd.append(np.abs(np.log(
                rapsd(pred[i],fft_method=np.fft)
                /rapsd(true[i],fft_method=np.fft)
            )))

    val_rel_l2_mean = np.mean(val_errors_rel_l2)
    val_rrmse_mean  = np.mean(val_errors_rrmse)
    print(f"Mean rel l2 error: {val_rel_l2_mean*100:.3f}%")
    print('Relative root-mean-square error = %.3f' % (val_rrmse_mean*100), '%')
    print('Mean energy log ratio = %.3f' % np.mean(val_errors_rapsd))

if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    logging.info(f"Received the following arguments: {a}")
    main(a, return_model=False)
