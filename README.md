# Bayesian sensitivity analysis toolkit

A Python-based toolkit for **Bayesian sensitivity analysis** using the **Kernel Stein Discrepancy (KSD)**.
It provides a modular framework for:
- Computing prior and posterior KSD
- Performing parametric and nonparametric sensitivity analysis to the loss learning rates and priors
- Visualizing sensitivity results

---

## Features

- **Distributions Support**
  - Gaussian (both univariate and multivariate)
  - Inverse-Wishart
  - Log-normal
  - can be extended to custom distributions

- **KSD computations**
  - KSD between posterior samples and candidate posterior
  - KSD between prior samples and candidate prior

- **Kernels**
  - Inverse multiquadric

- **Optimization tools for sensitivity analysis**
  - Parametric corner-point search
  - Nonparametric basis expansions (e.g., RBFs, polynomials) with SDP relaxation for non-convex QCQP problems

- **Plotting utilities**
  - Prior vs posterior samples
  - Log prior approximations for different optimization settings
  - Comparisons between SDP and direct KSD minimization

- **Configuration-driven**
   [Hydra](https://hydra.cc/) for flexible experiment configuration via YAML files.

---

## Contributions

We are happy with any help in adding distributions, kernels and models to the project!

---

## License


---

## Playground

This is a living guide for users to understand which objects they can pick in configs. And how to run the package.

### Distributions
- **Gaussian** (univariate): `src.distributions.gaussian.Gaussian`
- **MultivariateGaussian**: `src.distributions.gaussian.MultivariateGaussian`
- **LogNormal**: `src.distributions.log_normal.LogNormal`
- **InverseWishart**: `src.distributions.inverse_wishart.InverseWishart`

You can run 
```bash
python -m playground.zoo
```
to explore the zoo of all supported distributions/kernels etc.

### Config structure example
```yaml
data:
  base_prior:
    _target_: src.distributions.gaussian.Gaussian
    # ... base prior parameters
  true_dgp:
    _target_: src.distributions.gaussian.Gaussian
    # ... true dgp parameters
  loss:
    _target_: src.losses.gaussian_log_likelihood.GaussianLogLikelihood
    # ... loss parameters
  loss_lr: 
  observations_num: 
  posterior_samples_num: 
  prior_samples_num: 
  
model:
  _target_: src.bayesian_model.univariate_gaussian.UnivariateGaussianModel

ksd:
  kernel:
    _target_: src.kernels.rbf.RBFKernel
    # any kernel-specific params
  optimize:
    prior:
      Gaussian:
        parameters_box_range:
          ranges: 
          nums:

      nonparametric:
        basis_funcs_type: 
        basis_funcs_kwargs: 
        radius_lower_bound: 
        scale_samples: 
    loss:
      GaussianLogLikelihood:
        parameters_box_range:
          ranges: 
          nums:
```

### Kernels
- **IMQ / InverseMultiquadric**: `src.kernels.inverse_multiquadric.InverseUnivariateMultiquadricKernel` or 
`src.kernels.inverse_multiquadric.InverseMultivariateMultiquadricKernel`

**Instantiation**: the scripts do `instantiate(cfg.ksd.kernel, reference_data=samples)`.
Your kernel should accept `reference_data` in the constructor.

### Basis Functions (for non-parametric optimization)
- Basis is provided internally by `src.optimization.nonparametric.OptimizationNonparametricBase` via its `basis_function` field.
- To add a custom basis, ensure your optimizer config points to it or your optimizer builds it.

### Losses / Objectives
- Parametric: uses corner-point search defined in `cfg.ksd.optimize.prior.<Dist>` or `cfg.ksd.optimize.loss.<Dist>` sections.
- Nonparametric: SDP relaxation controlled by `cfg.ksd.optimize.prior.nonparametric`.

### File Inputs (for the CLI scripts)
- Posterior samples: `.npy` saved numpy array with shape `(n, d)`.
- Prior samples (nonparametric only): `.npy` array, shape `(n, d)`.

### Example run
```bash
# Parametric (corner points)
python playground/run_parametric_cli.py \
  --config-path configs/paper/ksd_calculation/toy \
  --config-name univariate_gaussian \
  playground.posterior_path=data/posterior.npy \
  playground.output_prefix=param_run

# Nonparametric (SDP)
python playground/run_nonparametric_cli.py \
  --config-path configs/paper/ksd_calculation/toy \
  --config-name univariate_gaussian \
  playground.posterior_path=data/posterior.npy \
  playground.prior_path=data/prior.npy \
  playground.radius=3.0 \
  playground.save_psi=true \
  playground.output_prefix=np_run
```

### Outputs
- JSON summary and CSV table saved under `cfg.flags.plots.output_dir`.
- For nonparametric, optional `*_psi_opt.npy` with the optimized coefficients.

---
