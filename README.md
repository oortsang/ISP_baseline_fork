# Baseline Models for Solving the Inverse Scattering Problem

## Overview
This repository contains four baseline deterministic models and the U-ViT diffusion model for solving the wideband inverse scattering problem. The included models are:

- **SwitchNet**
- **Wideband Butterfly Network**
- **Uncompressed Equivariant Model (EquiNet)**
- **Compressed Equivariant Model (B-EquiNet)**
- **U-ViT Diffusion Model**

SwitchNet is described in [SwitchNet: a neural network model for forward and inverse scattering problems](https://doi.org/10.1137/18M1222399).

Wideband Butterfly Network is detailed in [Wide-band butterfly network: stable and efficient inversion via multi-frequency neural networks](https://doi.org/10.1137/20M1383276).

The Uncompressed and Compressed Equivariant Models are explained in [Solving the wide-band inverse scattering problem via equivariant neural networks](https://doi.org/10.1016/j.cam.2024.116050).

The deterministic models were implemented by [Borong Zhang](https://borongzhang.com/) using code provided by the original authors and the [Swirl-Dynamics repository](https://github.com/google-research/swirl-dynamics), while the U-ViT Diffusion Model was implemented by [Martin Guerra](https://sites.google.com/wisc.edu/martinguerra/home) based on the [Swirl-Dynamics probabilistic diffusion project](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion).


## Installation
Project Environment can be installed by 
```
conda create -n isp_baseline python=3.11 
conda activate isp_baseline
pip install git+https://github.com/borongzhang/ISP_baseline.git@main
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Demos
Demos for these models can be found in the `examples` folder.

## Datasets
We have made the datasets and the data generation scripts publicly available in the [repository](https://github.com/borongzhang/back_projection_diffusion).

