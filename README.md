# Bayesian sensitivity analysis toolkit

**stein-sense** is a Python toolkit for **Bayesian sensitivity analysis** using the **Kernel Stein Discrepancy (KSD)**.  
It provides a modular framework for:
- Computing prior and posterior KSD
- Performing parametric and nonparametric sensitivity analysis
- Optimizing perturbations to priors subject to KSD constraints
- Visualizing sensitivity results

---

## 📚 Features

- **Bayesian model definitions**  
  Support for Gaussian, inverse-Wishart, log-normal priors, and extensible to custom distributions.

- **Kernel Stein Discrepancy computations**  
  - Posterior KSD
  - Prior KSD
  - Flexible kernel selection (e.g., inverse multiquadric)

- **Optimization tools**
  - Parametric corner-point search
  - Nonparametric basis expansions (e.g., RBFs, polynomials)
  - Semidefinite programming (SDP) relaxation for non-convex QCQP problems
  - KSD-minimizing solutions

- **Plotting utilities**  
  Visualizations for:
  - Prior vs posterior samples
  - Log prior approximations for different optimization settings
  - Comparisons between SDP and direct KSD minimization

- **Configuration-driven**  
  Uses [Hydra](https://hydra.cc/) for flexible experiment configuration via YAML files.

---


