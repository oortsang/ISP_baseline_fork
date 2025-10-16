# Baseline Models for Solving the Inverse Scattering Problem

## Overview
This is a fork of [the ISP_baselines repository](https://github.com/borongzhang/ISP_baseline) developed by Borong Zhang and Martin Guerra, adapted for use with MFISNet-like datasets.
The repository is developed in JAX and includes four deterministic models as baselines, along with a U-ViT diffusion model for the multifrequency inverse scattering problem:

- **SwitchNet** [(link to the paper)](https://doi.org/10.1137/18M1222399)
- **Wideband Butterfly Network** [(link to the paper)](https://doi.org/10.1137/20M1383276)
- **Uncompressed Equivariant Model (EquiNet)** [(link to the paper)](https://doi.org/10.1016/j.cam.2024.116050)
- **Compressed Equivariant Model (B-EquiNet)** [(link to the paper)](https://doi.org/10.1016/j.cam.2024.116050)
- **U-ViT Diffusion Model** [(link to the paper)](https://doi.org/10.1016/j.cam.2024.116050)

From the original repo: "The deterministic models were implemented by [Borong Zhang](https://borongzhang.com/) using code provided by the original authors and the [Swirl-Dynamics repository](https://github.com/google-research/swirl-dynamics), while the U-ViT Diffusion Model was implemented by [Martin Guerra](https://sites.google.com/wisc.edu/martinguerra/home) based on the [Swirl-Dynamics probabilistic diffusion project](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion)."

### Adaptations
We add additional code to handle our dataset. Any other modifications are minimal. (I may also modify the environment information for packages/dependencies.)

New files:
- `ISP_baseline/src/data_io.py` handles loading and saving HDF5 files for the dataset.
- `ISP_baseline/src/datasets.py` converts the MFISNet code's dataset representation to the one used by the Wide-band Equivarianet Network models.
- `ISP_baseline/src/more_metrics.py` adds support for relative l2 error and PSNR.
- `ISP_baseline/src/noise.py` brings support for our additive noise model.
- `ISP_baseline/src/predictions.py` helps to evaluate models and save their predictions to disk.
- `train_EquiNet_Uncompressed.py` is based on `ISP_baseline/examples/EquiNet_Uncompressed_10hSquares.ipynb`.
- `train_B_EquiNet_Compressed.py` is based on `ISP_baseline/examples/B_EquiNet_Compressed_10hSquares.ipynb`.

Other modifications worth mentioning:
- `ISP_baseline/src/utils.py` now has an interface that saves the interpolation and rotation matrices to disk, to avoid the slow setup process each time. This is more noticeable on our larger 192x192 dataset compared to the original 80x80 dataset.
- `ISP_baseline/models/Uncompressed.py` has an alternate interface for the model that allows for an arbitrary number of input frequencies.