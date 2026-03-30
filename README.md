# Conditional Diffusion Model for Cosmological Parameters (Ωₘ, σ₈)

**Generates 256×256 HI intensity maps conditioned on cosmological parameters** using a DDPM + DDIM pipeline trained on CAMELS Latin Hypercube (LH) simulations.

## Features
- Label-conditioned U-Net (2 parameters: Ωₘ, σ₈)
- DDPM forward process + fast DDIM sampling (50 steps)
- EMA, gradient clipping, CosineAnnealingLR, early stopping
- WandB logging (optional)


## References
- DDPM: Ho et al. (2020) — https://arxiv.org/abs/2006.11239
- DDIM: Song et al. (2020) — https://arxiv.org/abs/2010.02502
- Data: CAMELS project[](https://www.camel-simulations.org/)

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/conditional-cosmological-diffusion.git
cd conditional-cosmological-diffusion

conda create -n cond-diffusion python=3.10
conda activate cond-diffusion
pip install -r requirements.txt
