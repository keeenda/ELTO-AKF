# Structured Noise Adaptation for Sequential Bayesian Filtering with Embedded Latent Transfer Operators

**Naichang Ke, Pongpisit Thanasutives, Yoshinobu Kawahara**

*[Venue — to be updated]* | [OpenReview](https://openreview.net/forum?id=smFAyzvh5r&noteId=5vV5D4qJt7)

> **Abstract** — *To be updated.*

---

## Overview

This repository contains the official implementation of **ELTO-AKF**, a framework for sequential Bayesian filtering that combines:

- **ELTO** (Embedded Latent Transfer Operator): a kernel-based spectral learning method that extracts a latent state representation from observations via CCA-based stochastic realization.
- **OAKF** (Online Adaptive Kalman Filter): a Kalman filter operating in the RKHS-embedded latent space, with structured noise covariance adaptation using a Block-wise Scalar (BWS) parameterization optimized by CMA-ES.

## Repository Structure

```
├── model/
│   ├── ELTO_Kernel.py      # Spectral learning of latent transfer operators
│   ├── OELTO_KF.py         # Online Adaptive Kalman Filter (OAKF)
│   ├── oELTO_eval.py       # High-level wrapper (OELTO)
│   ├── deep_kernel.py      # Kernel functions (RBF, Matérn, Laplacian, etc.)
│   └── utils.py            # Parameter decorators for CMA-ES
├── experiments/
│   ├── lorenz96_elto_eval.py   # Lorenz-96 scalability experiment (self-contained)
│   ├── split_lv.py             # Lotka-Volterra filtering experiment
│   ├── equations_parametric.py # Burgers equation experiment
│   └── baseline_runners.py     # SPKFnUI neural baseline
└── ssm_discovery/
    └── utils.py                # Data loading utilities
```

## Requirements

```bash
pip install torch numpy scipy scikit-learn cma matplotlib pandas torchvision
```

## Experiments

All scripts are run from the **repository root**.

### Lorenz-96 (no data download required)

Simulates Lorenz-96 dynamics internally and evaluates ELTO-AKF across state dimensions.

```bash
python experiments/lorenz96_elto_eval.py
python experiments/lorenz96_elto_eval.py --data-seed 7 --model-seed 123
python experiments/lorenz96_elto_eval.py --logdir ./logs
```

### Lotka-Volterra

Requires 4 `.npy` files placed in a local directory:

```
lv_100.npy
lv_100_noise1.npy
lv_100_noise10.npy
lv_100_noise100.npy
```

```bash
python experiments/split_lv.py --data-dir /path/to/lv/data
```

### Burgers Equation

Requires `burgers.mat` and `burgers_noisy50.mat` in `ssm_discovery/datasets/`.

```bash
python experiments/equations_parametric.py
```

## Citation

```bibtex
@inproceedings{ke2025eltoakf,
  title     = {Structured Noise Adaptation for Sequential {Bayesian} Filtering
               with Embedded Latent Transfer Operators},
  author    = {Ke, Naichang and Thanasutives, Pongpisit and Kawahara, Yoshinobu},
  booktitle = {[Venue — to be updated]},
  year      = {2025},
}
```
