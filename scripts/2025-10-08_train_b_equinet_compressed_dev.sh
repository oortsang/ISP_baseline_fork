#!/bin/bash

dataset_dir=/home-nfs/oortsang/rlc-repo/dataset

echo "Please ensure the jaxisp-v3 environment is active before running this script!"

# --data_input_nus 2 5 10 \
python train_B_EquiNet_Compressed.py \
--ref_data_dir_base $dataset_dir \
--data_input_nus 1 2 3 4 5 6 7 8 9 10 \
--quadtree_l 4 \
--quadtree_s 12 \
--neta 192 \
--nx 192 \
--downsample_ratio 1 \
--blur_sigma 0.5 \
--truncate_num_train 100 \
--truncate_num_val 100 \
--truncate_num_test 100 \
--n_resnet_layers 6 \
--n_resnet_channels 3 \
--n_cnn_layers_2d 3 \
--n_cnn_channels_2d 9 \
--kernel_size_2d 5 \
--n_epochs 100 \
--batch_size 16 \
--log_batch_size 20 \
--output_pred_shard_size 1000 \
--output_pred_dir tmp/2025-10-07_output_pred_placeholder \
--debug


# TODO: add optimization parameters like learning rate and weight decay
# also, noise levels...
