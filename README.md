# SymmetryDiscovery

Software accompanying the paper **"Symmetry Discovery with Deep Learning"** by Krish Desai, Benjamin Nachman, and Jesse Thaler, published in *Physical Review D* (Vol. 105, Issue 9, Article 096031, May 2022).

> **Paper:** [https://doi.org/10.1103/PhysRevD.105.096031](https://doi.org/10.1103/PhysRevD.105.096031)
> **Repository:** [https://github.com/krishdesai7/SymmetryDiscovery_v0](https://github.com/krishdesai7/SymmetryDiscovery_v0)

---

## Overview

This repository implements a deep learning framework for discovering symmetries hidden in high-energy physics datasets. The core idea is to train a Generative Adversarial Network (GAN) whose generator learns a parameterized transformation (a composition of rotation matrices) such that the transformed data is indistinguishable from the original data. If such a transformation can be found, it reveals a symmetry of the underlying probability distribution.

The framework is applied to:
- **Synthetic data** – 2D multivariate Gaussian samples (MSE experiments)
- **LHCO anomaly detection dataset** – simulated dijet events (px, py for two jets)
- **CMS Open Data** – real collider jets from the CMS detector

---

## Repository Structure

```
SymmetryDiscovery_v0/
├── LHCOZ2.py              # 4D symmetry discovery on LHCO dijet data
├── MSE7.py                # 2D symmetry discovery on synthetic Gaussian data
├── LHCOZ2.ipynb           # Interactive version of LHCOZ2.py
├── MSE.ipynb              # Interactive version of MSE7.py
├── LHCO-KL.ipynb          # LHCO experiments with KL-divergence loss
├── Augmentation.ipynb     # Data augmentation experiments
├── CMS.ipynb              # Symmetry discovery on CMS Open Data jets
├── TestingLHCO6.ipynb     # Comprehensive validation notebook
├── Z7MSE.ipynb            # Analytical 2D MSE analysis
├── LHCOZ2.txt             # Output log from LHCOZ2.py
├── MSE7.txt               # Output log from MSE7.py
└── CITATION.cff           # Citation metadata
```

---

## Method

### Generator: Learnable Rotation Layer

A custom Keras layer (`MyLayer`) learns a set of rotation angles and assembles them into a transformation matrix applied to the input data.

- **2D case (MSE7.py):** 2 parameters (cos θ, sin θ) defining a 2×2 rotation.
- **4D case (LHCOZ2.py):** 6 rotation angles, each rotating a different pair of the four momentum coordinates (px₁, py₁, px₂, py₂). The six matrices are composed as:

  ```
  S = R₁ · R₂ · R₃ · R₄ · R₅ · R₆
  ```

### Loss Function

Training minimizes a combined loss:

```
L = CrossEntropy(D(g(x)), 1) + α · MSE(g(g(x)), x)
```

- **First term:** Distribution-matching loss — the transformed data should fool the discriminator into believing it is real data.
- **Second term:** Involution regularization — the transformation should be its own inverse, i.e., g(g(x)) ≈ x.
- **α** controls the regularization strength (default `α = 0.1`).

### Discriminator

A small fully-connected neural network with two hidden layers of 25 ReLU units and a sigmoid output, trained to distinguish real from transformed data.

---

## Requirements

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn scipy
```

For the CMS notebook, install the [EnergyFlow](https://energyflow.network/) package:

```bash
pip install energyflow
```

### External Data

The LHCO notebooks and scripts require the LHCO 2020 anomaly detection dataset:

```
events_anomalydetection_DelphesPythia8_v2_qcd_features.h5
```

This file is **not included** in the repository due to its size. It can be downloaded from the [LHCO 2020 dataset page](https://lhco2020.github.io/homepage/).

---

## Usage

### Python Scripts

Run the standalone scripts directly from the command line:

```bash
# 4D symmetry discovery on LHCO dijet data
python LHCOZ2.py

# 2D symmetry discovery on synthetic Gaussian data
python MSE7.py
```

Results (learned rotation angles) are printed to the terminal and appended to the corresponding `.txt` output files.

### Jupyter Notebooks

Launch Jupyter and open any notebook for an interactive session:

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `LHCOZ2.ipynb` | 4D rotation discovery on Z→2 jets (LHCO) |
| `MSE.ipynb` | 2D MSE-based rotation learning |
| `LHCO-KL.ipynb` | LHCO experiments using KL-divergence loss |
| `Augmentation.ipynb` | Data augmentation experiments |
| `CMS.ipynb` | Symmetry discovery on CMS Open Data jets |
| `TestingLHCO6.ipynb` | Comprehensive validation and testing |
| `Z7MSE.ipynb` | Analytical 2D MSE analysis |

---

## Citation

If you use this software, please cite:

```bibtex
@article{Desai:2022qex,
  author  = {Desai, Krish and Nachman, Benjamin and Thaler, Jesse},
  title   = {Symmetry Discovery with Deep Learning},
  journal = {Phys. Rev. D},
  volume  = {105},
  number  = {9},
  pages   = {096031},
  year    = {2022},
  doi     = {10.1103/PhysRevD.105.096031}
}
```

---

## License

Please see the repository for license information. If you use this code, cite the associated paper above.
