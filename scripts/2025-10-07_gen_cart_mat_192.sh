#!/bin/bash

#SBATCH --job-name=2025-10-07_gen_cart_mat_192
#SBATCH --partition=cpu
#SBATCH --mem=50G
#SBATCH --gpus=0
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/2025-10-07_gen_cart_mat_192.out
#SBATCH --error=logs/2025-10-07_gen_cart_mat_192.err
#SBATCH --exclude=
#SBATCH --mail-type=END
#SBATCH --mail-user=oortsang@uchicago.edu

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

mat_dir=tmp/cart_and_rot_mats
echo "Please make sure this is running in the jaxisp-v3 environment!"

python -c """print(f'hello!')
from ISP_baseline.src.utils import load_or_create_mats
n=192
print(f'About to create the matrices; nx=neta={n}...')

load_or_create_mats(
    n,
    n,
    mats_dir='${mat_dir}',
    mats_format='new_mats_neta{0}_nx{1}.npz',
    save_if_created=True,
)
print('Goodbye ^.^')
"""
date
