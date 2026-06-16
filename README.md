# SILAGE — Logistic Regression Experiments

This repository reproduces the synthetic logistic-regression experiments for the
SILAGE paper. The main deliverables are **Figures 1 and 2**, the two synthetic
optimization-trajectory grids:

- **Figure 1** — the $m \ge n$ regime ($n = 50$ groups of $m = 250$ samples each);
  in the code we call this setting `m_gt_n`.
- **Figure 2** — the $n > m$ regime ($n = 250$ groups of $m = 50$ samples each);
  in the code we call this setting `n_gt_m`.

Here $n$ is the number of groups/blocks/silos and $m$ the number of samples per
group. See [Notation & naming conventions](#notation--naming-conventions) for the
full code-to-paper symbol mapping.

Each figure is a $2\times2$ grid over the four heterogeneity regimes and compares
**SILAGE** against the **ZeroSARAH**, **SILVER**, and **D-ZeroSARAH** baselines.

The figures are produced by first running grid searches over the batch-size
grids described in the paper and then plotting the tuned choices. This
repository does **not** re-run that grid search on the main path. Instead it
follows the recorded reproduction workflow: generate the synthetic data and
constants, launch the methods at the recorded tuned parameters, and draw the
trajectory grids. The saved run logs needed to redraw Figures 1 and 2 are
bundled under [logistic_regression/logs/](logistic_regression/logs/).

## Reference

This is the public code release accompanying the SILAGE paper:

> I. Sokolov, L. Condat, P. Richtárik. *SILAGE: Memory-Efficient,
> Full-Gradient-Free Nonconvex Optimization for Nested Finite Sums.*

A preprint link will be added here once the paper is on arXiv: **arXiv: _coming
soon_**. Section, equation, assumption, table, and figure numbers throughout this
README refer to that paper.

If you find the methods or code useful for your research, please consider citing:

```bibtex
@article{sokolov2026silage,
  title   = {{SILAGE}: Memory-Efficient, Full-Gradient-Free Nonconvex Optimization for Nested Finite Sums},
  author  = {Sokolov, Igor and Condat, Laurent and Richt{\'a}rik, Peter},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Notation & naming conventions

The code uses short ASCII names for the paper's symbols. The objective is a binary
logistic regression with a smooth nonconvex regularizer,

$$f(x)=\frac{1}{n}\sum_{i=1}^{n} f_i(x),\quad f_i(x)=\frac{1}{m}\sum_{j=1}^{m} f_{i,j}(x),\quad f_{i,j}(x)=\log\left(1+\exp(-y_{i,j}\,a_{i,j}^{\top} x)\right)+\lambda\sum_{\ell=1}^{d}\frac{x_\ell^{2}}{1+x_\ell^{2}},$$

with $\lambda=200$, where $n$ is the number of groups/blocks/silos, $m$ the number
of samples per group, and $N=nm$ the total sample count.

**Size settings** — the two dataset shape regimes (one figure each):

| code | paper regime | $(n,m)$ | algorithm | figure / output file |
|---|---|---|---|---|
| `m_gt_n` | $m \ge n$ | $(50, 250)$ | SILAGE Algorithm 1 | Figure 1 — `trajectory_grid_m_gt_n.pdf` |
| `n_gt_m` | $n > m$ | $(250, 50)$ | SILAGE Algorithm 2 | Figure 2 — `trajectory_grid_n_gt_m.pdf` |

**Heterogeneity regimes** — the four panels in each figure. `d1` $\equiv \delta_1$
(across-group similarity, Assumption 3) and `d2` $\equiv \delta_2$ (within-group
similarity, Assumption 4); `small`/`large` denote relative magnitude:

| code | paper label | meaning |
|---|---|---|
| `d1_small_d2_small` | $(\delta_1\text{ small}, \delta_2\text{ small})$ | globally & locally homogeneous data |
| `d1_small_d2_large` | $(\delta_1\text{ small}, \delta_2\text{ large})$ | redundant blocks |
| `d1_large_d2_small` | $(\delta_1\text{ large}, \delta_2\text{ small})$ | homogeneous silos |
| `d1_large_d2_large` | $(\delta_1\text{ large}, \delta_2\text{ large})$ | heterogeneous silos |

**Computed constants** — estimated by preprocessing and printed by the notebook's
check cell (the $\delta_1,\delta_2,L,L_{\max}$ values reproduce Table 4 of the
paper). They calibrate each method's theoretically admissible stepsize (Remark 1)
and do not change algorithm behavior:

| code | symbol | meaning |
|---|---|---|
| `lambda_reg` | $\lambda$ | nonconvex regularization weight ($=200$) |
| `delta1_emp` | $\delta_1$ | across-group similarity (Assumption 3, eq. (2)) |
| `delta2_emp` | $\delta_2$ | within-group similarity (Assumption 4, eq. (3)) |
| `delta_flat_emp` | $\delta_{\mathrm{flat}}$ | flattened similarity (eq. (4)); $\delta_{\mathrm{flat}}\le\sqrt{\delta_1^{2}+\delta_2^{2}}$ |
| `L_ij_max_emp` | $L_{\max}=\max_{i,j}L_{i,j}$ | worst-case per-component smoothness (eq. (6)) |
| `L_i_max_emp` | $\max_i L_i$ | max per-group smoothness (eq. (5)) |
| `L_global_emp` | $L$ | smoothness of the averaged objective $f$ (Assumption 2) |

The `_emp` keys are empirical (probe-set) estimates; the saved `_wc` keys are
their worst-case diagnostic counterparts.

## Repository layout

```
.
├── environment.yml              # Linux/CUDA environment (full experiment rerun)
├── environment_plotting.yml     # macOS/CPU environment (figure reproduction from logs)
└── logistic_regression/
    ├── reproduction.ipynb       # Main entry point (step-by-step reproduction)
    ├── data_preprocessing.py    # Synthetic data generation + comp-param estimation
    ├── run_preprocessing_synthetic_local.py   # Launcher: regenerate the 2 x 4 grid
    ├── run_synthetic_experiments_local.py      # Launcher: run methods at tuned params
    ├── SILAGE_m_gt_n.py, SILAGE_n_gt_m.py,
    │   ZeroSARAH.py, D_ZeroSARAH.py, SILVER.py # Method entry points
    ├── src/                     # Algorithms, oracles, synthetic generator, plotting
    └── logs/                    # Saved .npy/.npz run logs for Figures 1 and 2
```

## Two ways to reproduce

There are two supported workflows. Pick the one that matches your goal.

### A. Redraw Figures 1 and 2 from the bundled logs (macOS/CPU, fast)

Use this if you only want to regenerate the figures from the run logs already
included in this repository. It does not require a GPU and does not re-run any
experiment.

```bash
conda env create -f environment_plotting.yml
conda activate silage-plot
python -m ipykernel install --user --name silage-plot --display-name "Python (silage-plot)"
```

The `silage-plot` environment registers a Jupyter kernel (via the `ipykernel`
command above) but does not include a notebook server. Open
`logistic_regression/reproduction.ipynb` in your existing Jupyter (Lab/Notebook)
or in VS Code, and select the **Python (silage-plot)** kernel.

In the notebook, run **Section 1** (environment check) and **Section 4** (draw
figures). The outputs are written to:

- `logistic_regression/plots/synthetic_trajectory_grids/trajectory_grid_m_gt_n.pdf`
- `logistic_regression/plots/synthetic_trajectory_grids/trajectory_grid_n_gt_m.pdf`

> **Workflow A is faithful from the logs alone.** The trajectory curves and the
> panel titles (the qualitative $(\delta_1,\delta_2)$ regime label plus the
> $n,m$ suptitle) come entirely from the bundled `logs/`; preprocessing is **not**
> required to draw the figures. The measured similarity constants
> $\delta_1,\delta_2,L,L_{\max}$ (Table 4 of the paper) are produced separately by
> the preprocessing step (`data_synthetic_dirichlet_logreg/`) and reported by the
> check cell in Section 2 — see workflow B.

### B. Full experiment rerun (Linux/CUDA)

Use this to regenerate everything from scratch on a Linux machine with an
NVIDIA GPU. This is the environment used for the paper runs (PyTorch pinned to
CUDA 11.8).

```bash
conda env create -f environment.yml
conda activate silage
```

Run the steps from `logistic_regression/`:

```bash
cd logistic_regression

# 1. Generate the 2 x 4 synthetic grid and the comp-param constants.
python run_preprocessing_synthetic_local.py

# 2. Launch the tuned method runs used in Figures 1 and 2.
python run_synthetic_experiments_local.py --algorithm silage_m_gt_n --launch_mode tmux
python run_synthetic_experiments_local.py --algorithm silage_n_gt_m --launch_mode tmux
python run_synthetic_experiments_local.py --algorithm zerosarah     --launch_mode tmux
python run_synthetic_experiments_local.py --algorithm d_zerosarah   --launch_mode tmux
python run_synthetic_experiments_local.py --algorithm silver        --launch_mode tmux

# 3. Draw the figures (Section 4 of reproduction.ipynb).
```

Notes:

- The launchers default to `cuda:0`. To force CPU or a single device, pass
  `--device cpu` or `--device cuda:0`. CPU execution works but is much slower
  than the CUDA workstation used for the paper.
- `tmux` launches run jobs in detached sessions. For a sequential debug run,
  use `--launch_mode direct --job_index <idx>`.
- The synthetic regime parameters (including `lambda_reg = 200.0`) are defined
  in [logistic_regression/src/synthetic_logreg.py](logistic_regression/src/synthetic_logreg.py),
  which is the single source of truth for data generation.

## Reproducibility notes

- The bundled logs under `logistic_regression/logs/` are sufficient to redraw
  Figures 1 and 2 (workflow A).
- Generated artifacts (synthetic data, launcher configs, per-run stdout logs,
  rendered plots) are not tracked; they are recreated by the steps above.
