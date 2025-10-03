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