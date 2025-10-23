#!/bin/bash

dataset_dir=/home-nfs/oortsang/rlc-repo/dataset

echo "Please ensure the jaxisp-v3 environment is active before running this script!"
# export JAX_TRACEBACK_FILTERING=off
export JAX_TRACEBACK_FILTERING=remove_frames

python train_EquiNet_Uncompressed.py \
--ref_data_dir_base $dataset_dir \
--data_input_nus 1 2 3 4 5 6 7 8 9 10 \
--noise_to_signal_ratio 0.0 \
--neta 192 \
--nx 192 \
--downsample_ratio 1 \
--blur_sigma 0.5 \
--truncate_num_train 100 \
--truncate_num_val 100 \
--truncate_num_test 100 \
--n_cnn_layers_2d 3 \
--n_cnn_channels_2d 6 \
--kernel_size_2d 3 \
--lr_init 1e-5 \
--n_epochs 100 \
--batch_size 16 \
--log_batch_size 16 \
--output_pred_shard_size 1000 \
--output_pred_dir tmp/2025-10-07_output_pred_placeholder \
--debug
