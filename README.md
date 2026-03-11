# Narrow-Parameter Problem in NPE + Importance Sampling

Reproduction code for the experiments from our study of Dingo-IS (Dax et al. 2023).<br>
Compares forward KL, reverse KL, alpha-divergence, annealed training, and synthetic marginalization on 2D and 3D toy posteriors.

## Setup

```bash
git clone <repo>
cd narrow_posterior_IS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The code automatically uses CUDA if available.

## Run

```bash
# 2D experiments (less than 10min on T4 GPU)
python run_experiments.py 2d

# 3D experiments (slightly more)
python run_experiments.py 3d

# Options
python run_experiments.py 2d --out_dir figures --n_epochs 400 --n_is 20000 --seed 42
```

Figures are saved as PDF/PNG to `figures/` (or `--out_dir`).

## Structure

```
├── models/              # Real-NVP normalizing flow
├── posteriors/          # 2D and 3D toy posteriors
├── training/            # fKL, rKL, alpha-div, annealed objectives
├── inference/           # IS utilities + synthetic marginalization
├── plotting/            # all figure generation
├── report/              # The 5-page PDF report
├── run_experiments.py   # main script
└── everything.ipynb     # detailed and commented notebook
```
