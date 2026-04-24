# Bayesian sensitivity analysis toolkit

A Python-based toolkit for **global Bayesian sensitivity analysis** using the **Fisher Divergence (FD)**.

---

## Features

- **Distributions**
  - Gaussian (univariate and multivariate)
  - Log-normal
  - Gamma, Inverse-Gamma
  - Beta
  - Cauchy, Half-Cauchy
  - Uniform
  - Chi-squared
  - Inverse-Wishart
  - Composite product of independent marginals
  - Extensible to custom distributions

- **FD computations**
  - FD between posterior samples and candidate posterior
  - FD between prior samples and candidate prior

- **Optimization tools for sensitivity analysis**
  - Parametric corner-point search over prior and loss parameters

- **Plotting utilities**
  - Prior vs posterior samples
  - FD sensitivity surfaces across parameter grids
  - Comparisons across methods and models

- **Configuration-driven**
  [Hydra](https://hydra.cc/) for flexible experiment configuration via YAML files.

---

## Main components

Analysis is structured around three building blocks that are composed in each experiment script.

### 1. Bayesian model — `src/bayesian_model/`

The abstract base [`BayesianModel`](src/bayesian_model/base.py) defines the interface: it holds a prior and a likelihood, exposes score functions, and handles posterior/prior sampling. Concrete subclasses implement model-specific closed-form posteriors:

- [`SimpleGaussianModel`](src/bayesian_model/gaussian.py) — univariate Gaussian likelihood with Gaussian or Log-normal prior.
- [`MultivariateGaussianModel`](src/bayesian_model/gaussian.py) — multivariate Gaussian likelihood with Gaussian prior on the mean.

### 2. Fisher Divergence — `src/discrepancies/`

FD is computed separately for the prior and the posterior.

- [`PriorFDBase`](src/discrepancies/prior_fisher.py) — FD between prior samples and a candidate prior; uses the exponential family score decomposition.
- [`PosteriorFDBase`](src/discrepancies/posterior_fisher.py) — FD for the posterior, combining the reference prior score with the candidate prior's natural statistics evaluated on posterior samples.

The low-level [`FisherDivergenceBase`](src/discrepancies/fisher.py) computes the mean squared score difference and is used internally by both classes above.

### 3. Optimizer — `src/optimization/`

Given a discrepancy object, the optimizer searches for the worst-case prior (or loss learning rate) over a user-specified parameter box.

- [`OptimizationCornerPointsUnivariateGaussianConjugate`](src/optimization/corner_points_fisher.py) — exact closed-form corner-point search for univariate conjugate Gaussian models.
- [`OptimizationCornerPointsCompositePrior`](src/optimization/corner_points_fisher.py) — global search (differential evolution / dual annealing) for composite independent-marginal priors; supports Gaussian, FGM, and Frank copula perturbations.

---

## Contributions

We are happy with any help in adding distributions and models to the project!

---

## License

---

## Config structure

Experiments are configured via YAML files loaded by Hydra. Below is an example for a univariate Gaussian model:

```yaml
data:
  base_prior:
    _target_: src.distributions.gaussian.Gaussian
    mu: 2
    sigma: 4
  true_dgp:
    _target_: src.distributions.gaussian.Gaussian
    mu: 3
    sigma: 2
  loss:
    _target_: src.losses.gaussian_log_likelihood.GaussianLogLikelihood
    mu: 3
    sigma: 2
  loss_lr: 1.0
  observations_num: 100
  posterior_samples_num: 1000
  prior_samples_num: 1000

model:
  _target_: src.bayesian_model.gaussian.SimpleGaussianModel

fd:
  optimize:
    prior:
      Gaussian:
        parameters_box_range:
          ranges:
            mu: [-10, 10]
            sigma: [2, 5]
          nums:
            mu: 21
            sigma: 4
    loss:
      GaussianLogLikelihood:
        parameters_box_range:
          ranges:
            lr: [0.5, 2.5]
          nums:
            lr: 9

flags:
  plots:
    output_dir: outputs/paper/plots/fisher/univariate
```

---

## Tests

The `tests/` directory is organized into four folders, one per experimental setting.
Each folder contains scripts named with the `_fisher` suffix for the FD-based experiments.

### `tests/paper/`
Toy Gaussian experiments and finite-sample complexity comparisons.
- `run_toy_fisher.py` — sensitivity analysis on univariate/multivariate Gaussian models; generates FD sensitivity curves and comparison plots against competing methods.
- `run_comparison_fisher.py` — side-by-side comparison of FD-based sensitivity against other divergences on toy models.

### `tests/ising/`
Experiments on the Ising model with different loss functions (pseudolikelihood, FD-Bayes).
- `run_ising_fisher.py` — computes and plots FD sensitivity over the inverse temperature parameter across loss types.

### `tests/posteriordb/`
Real-data experiments using models from the PosteriorDB benchmark.
- `run_ark_fisher.py` — FD sensitivity analysis for the ARK (autoregressive kernel) model.
- `run_ark_kilpisjarvi.py` — FD sensitivity analysis for the Kilpisjarvi dataset model.

### `tests/sbi/`
Experiments on the Turin channel model fitted via simulation-based inference (SBI).
- `run_turin_fisher.py` — FD sensitivity analysis for the Turin SBI model.
- `run_turin_fisher_fgm.py` — Turin model with FGM copula prior; plots pairwise copula grids.
- `run_turin_fisher_frank.py` — Turin model with Frank copula prior.
