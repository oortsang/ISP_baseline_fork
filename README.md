# Baseline Models for Solving the Inverse Scattering Problem

## Overview
This repository currently contains four baseline deterministic models for solving the wideband inverse scattering problem. The models included are:

- **SwitchNet**
- **Uncompressed Equivariant Model**
- **Compressed Equivariant Model**
- **Wideband Butterfly Network**

## Installation
Project Environment can be installed by 
```
conda create -n jax_isp python=3.11 
conda activate jax_isp
pip install git+https://github.com/google-research/swirl-dynamics.git@main
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install jupyter matplotlib natsort 
```

## Demos
Demos for these models can be found in the `colabs` folder.

## Comments on the uncompressed and compressed rotationally equivariant models
-The two models are sensitive to the order of the source dimension (s) and the receiver dimension (r) in the far-field pattern data. If the models yield very poor results, try training with the perturbation data transposed.

-Using the warmup_cosine_decay_schedule scheduler to train the two models yields much better results (compared to the exponential_decay scheduler used in the TensorFlow codes).
