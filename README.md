# Baseline Models for Solving the Inverse Scattering Problem (Under Construction: will be ready by 02/27/2025) 

## TODO: examples, readme, classical methods, metrics

## Overview
This repository currently contains four baseline deterministic models and the U-ViT diffusion model for solving the wideband inverse scattering problem. The models included are:

- **SwitchNet**
- **Wideband Butterfly Network**
- **Uncompressed Equivariant Model**
- **Compressed Equivariant Model**
- **U-ViT Diffusion Model**

Uncompressed and Compressed Equivariant Models are describe in "..." The authors are [Borong Zhang](https://borongzhang.com/), [Qin Li](https://sites.google.com/view/qinlimadison/home), and [Leonardo Zepeda-Núñez](https://research.google/people/leonardozepedanez/?&type=google).

U-ViT Diffusion Model is implemented by [Martin Guerra](https://sites.google.com/wisc.edu/martinguerra/home). 

## TODO
Add FWI, least square, Metrics and their credits.


## Installation
Project Environment can be installed by 
```
conda create -n isp_baseline python=3.11 
conda activate isp_baseline
pip install git+https://github.com/borongzhang/ISP_baseline.git@main
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Demos
Demos for these models can be found in the `colabs` folder.

## Comments on the uncompressed and compressed rotationally equivariant models
-Using the warmup_cosine_decay_schedule scheduler to train the two models yields much better results (compared to the exponential_decay scheduler used in the TensorFlow codes).
