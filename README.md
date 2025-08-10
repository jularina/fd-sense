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
