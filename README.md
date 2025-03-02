# Baseline Models for Solving the Inverse Scattering Problem

## Overview
This repository currently contains four baseline deterministic models and the U-ViT diffusion model for solving the wideband inverse scattering problem. The models included are:

- **SwitchNet**
- **Wideband Butterfly Network**
- **Uncompressed Equivariant Model**
- **Compressed Equivariant Model**
- **U-ViT Diffusion Model**

SwitchNet is described in [SWITCHNET: A NEURAL NETWORK MODEL FOR FORWARD AND INVERSE SCATTERING PROBLEMS](https://doi.org/10.1137/18M1222399).
Wideband Butterfly Network is described in [Wide-band butterfly network: stable and efficient inversion via multi-frequency neural networks](https://doi.org/10.1137/20M1383276).
Uncompressed and Compressed Equivariant Models are described in [Solving the wide-band inverse scattering problem via equivariant neural networks](https://doi.org/10.1016/j.cam.2024.116050). 
The deterministic models are implemented by [Borong Zhang](https://borongzhang.com/).
U-ViT Diffusion Model is implemented by [Martin Guerra](https://sites.google.com/wisc.edu/martinguerra/home). 

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
