import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cmx
from typing import Dict, Tuple, List, Optional, Sequence
from utils.distributions import DISTRIBUTION_MAP as _DIST_MAP_EXTPROJ


def _deep_get(cfg, path, default=None):
    """Safe nested getter that works with dicts or OmegaConf/objects."""
    if cfg is None:
        return default
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })

def _new_fig_ax(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    return fig, ax

def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)

def _palette(plot_cfg, n: int) -> List[str]:
    base = list(plot_cfg.plot.color_palette.colors)
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def _latex_name(plot_cfg, key: str, default: str) -> str:
    try:
        return getattr(plot_cfg.plot.param_latex_names, key)
    except Exception:
        return default

def _gaussian_pdf(x, mu, sigma):
    x = np.asarray(x, float)
    s2 = sigma * sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (x - mu) ** 2 / s2)

def _lognormal_pdf(x, mu_log, sigma_log):
    x = np.asarray(x, float)
    pdf = np.zeros_like(x)
    pos = x > 0
    z = (np.log(x[pos]) - mu_log) / sigma_log
    pdf[pos] = (1.0 / (x[pos] * sigma_log * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)
    return pdf

def _get_base_gaussian(cfg, name: str) -> Tuple[float, float]:
    dist = cfg.data.base_prior.distributions[name]
    return float(dist.mu), float(dist.sigma)

def _extract_worst(rows: List[Dict]) -> Dict:
    return max(rows, key=lambda r: float(r["value"]))

def _extract_params(d: Dict, key: str) -> Dict:
    """Return dict of params for given key (beta1/beta3) if present."""
    if not isinstance(d, dict):
        return {}
    return d.get(key, {}).get("params", {})

def plot_ar_time_series(
    y: np.ndarray,
    plot_cfg,
    output_dir: str,
    filename: str = "ar_pred_ribbon.pdf",
):
    """
    y: (T,)
    y_rep: (n_draws, T) or None; if None, only observed series is plotted
    """
    _apply_plot_rc(plot_cfg)
    fig, ax = _new_fig_ax(plot_cfg)

    T = len(y)
    t = np.arange(1, T + 1)

    # lines
    ax.plot(t, y, label=_latex_name(plot_cfg, "observed", "observed"), linewidth=1.5)
    ax.set_xlabel(_latex_name(plot_cfg, "time_t", r"time $t$"))
    ax.set_ylabel(_latex_name(plot_cfg, "y_t", r"$y_t$"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, output_dir, filename, plot_cfg)


def _get_component_cfg(cfg, name: str):
    """Find component config by name from cfg.ksd.optimize.prior.Composite.components."""
    comps = _deep_get(cfg, "ksd.optimize.prior.Composite.components", [])
    for c in comps:
        if (isinstance(c, dict) and c.get("name") == name) or getattr(c, "name", None) == name:
            return c
    raise KeyError(f"Component '{name}' not found in cfg.")

def _parse_family_from_target(target: str) -> str:
    # e.g. "src.distributions.gaussian.Gaussian" -> "Gaussian"
    return target.split(".")[-1]

def _get_base_prior_spec(cfg, name: str) -> Tuple[str, Dict[str, float]]:
    """Read family+params from cfg.data.base_prior.distributions[name]."""
    d = _deep_get(cfg, f"data.base_prior.distributions.{name}", None)
    if d is None:
        raise KeyError(f"Base prior for '{name}' not found in cfg.data.base_prior.distributions")
    target = d.get("_target_", "")
    family = _parse_family_from_target(target) if isinstance(target, str) else d.get("family", "")
    # copy params, excluding _target_
    params = {k: float(v) for k, v in d.items() if k != "_target_"}
    return family, params

def _make_pdf(family: str, params: Dict[str, float]):
    """Prefer your DISTRIBUTION_MAP, otherwise fallback for Gaussian/Gamma."""
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)
    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f


def _auto_x_range_from_ref(family: str, ref_params: Dict[str, float]) -> Tuple[float, float]:
    """
    Choose a reasonable x-range for plotting PDFs, based on a reference prior.

    Supported families (case-insensitive):
      - "Gaussian" / "Normal":      (mu - 4*sigma, mu + 4*sigma)
      - "HalfCauchy":               (0, 10*gamma)              # x >= 0
      - "LogNormal":                (exp(mu_log - 5*sigma_log), exp(mu_log + 5*sigma_log)), clipped at > 0
      - "Gamma": (shape α, scale θ) (0, mean + 6*sd) where mean=αθ, sd=√α * θ

    Falls back to (-5, 5) if family is unknown.
    """
    low = family.strip().lower()

    # Gaussian / Normal
    if low in ("gaussian", "normal"):
        mu = float(ref_params["mu"])
        s  = float(ref_params["sigma"])
        return (mu - 4.0 * s, mu + 4.0 * s)

    if low in ("halfcauchy"):
        gamma = float(ref_params.get("gamma", ref_params.get("scale", 1.0)))
        return (0.0, 10.0 * gamma)

    # LogNormal with parameters mu_log, sigma_log (accept "sigma-log" typo too)
    if low == "lognormal":
        mu_log   = float(ref_params["mu_log"])
        sigma_log = float(ref_params.get("sigma_log", ref_params.get("sigma-log")))
        lo = np.exp(mu_log - 5.0 * sigma_log)
        hi = np.exp(mu_log + 5.0 * sigma_log)
        # Ensure strictly positive lower bound
        return (max(1e-8, lo), hi)

    # Gamma with shape alpha and scale theta; support x > 0
    if low == "gamma":
        alpha = float(ref_params["alpha"])
        theta = float(ref_params["theta"])
        mean = alpha * theta
        sd   = np.sqrt(alpha) * theta
        return (0.0, max(1e-8, mean + 6.0 * sd))

    # Fallback
    return (-5.0, 5.0)

def _linspace_pad(xmin, xmax, n=1200, pad=0.05):
    if not np.isfinite([xmin, xmax]).all():
        xmin, xmax = -5.0, 5.0
    if xmin == xmax:
        xmin -= 1.0; xmax += 1.0
    span = xmax - xmin
    return np.linspace(xmin - pad*span, xmax + pad*span, n)

def _sample_param_sets(ranges: Dict[str, List[float]], n: int, rng: np.random.Generator):
    keys = list(ranges.keys())
    lows = np.array([ranges[k][0] for k in keys], dtype=float)
    highs = np.array([ranges[k][1] for k in keys], dtype=float)
    out = []
    for _ in range(n):
        u = rng.random(len(keys))
        vals = lows + (highs - lows) * u
        out.append({k: float(v) for k, v in zip(keys, vals)})
    return out

def _canonicalize_params(params: Dict[str, float], tol: float = 1e-10) -> Tuple[Tuple[str, float], ...]:
    # Sort keys for stable ordering; round to tolerance to avoid tiny float diffs
    return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))

def _group_beta_most_sensitive(rows_all: Dict, beta_names: List[str]) -> List[Tuple[List[str], str, Dict[str, float]]]:
    """
    Returns list of groups: [([beta names], family, params_dict_for_group), ...]
    Group by (family, params) with rounding.
    """
    bykey = {}
    for b in beta_names:
        fam = rows_all[b]["family"]
        p = rows_all[b]["params"]
        key = (fam, _canonicalize_params(p))
        if key not in bykey:
            bykey[key] = ([], fam, p)
        bykey[key][0].append(b)
    # stable order by first beta index
    idx = {"beta1":1,"beta2":2,"beta3":3,"beta4":4,"beta5":5}
    groups = list(bykey.values())
    groups.sort(key=lambda g: min(idx.get(n, 99) for n in g[0]))
    return groups

def plot_three_panel_priors(
    rows_all: Dict,               # {'alpha': {...}, 'beta1': {...}, ..., 'sigma': {...}} with 'family' & 'params'
    cfg,                          # config with families & parameter ranges
    plot_cfg,                     # plotting config
    output_dir: str,
    prefix: str = "ark_param",
    sample_n_alpha: int = 30,
    sample_n_sigma: int = 30,
    sample_n_beta: int = 30,      # per-beta cloud count
    seed: int = 123,
    filename: str = None,
    x_alpha=None,                 # optional manual x-ranges
    x_sigma=None,
    x_beta=None,
):
    """
    Three-panel figure:
      row 1: [alpha | sigma]
      row 2: [betas consolidated]

    - Π_ref (dashed black) from cfg midpoint ranges
    - Candidate clouds from cfg ranges (faint)
    - Most sensitive: from rows_all['...']['params'].
      For betas, identical most-sensitive params are GROUPED and drawn in red but with
      distinct line styles; the legend shows grouped beta names.
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_three_panel_priors.pdf"

    rng = np.random.default_rng(seed)
    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.18
    palette_full = list(plot_cfg.plot.color_palette.colors)

    def _cloud_colors(n: int, skip_first: int = 2) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    # Figure & grid
    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    plt.subplots_adjust(right=0.82, wspace=0.)
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[25, 25, 50],
        wspace=0.25
    )
    ax_alpha = fig.add_subplot(gs[0, 0])  # left (alpha)
    ax_sigma = fig.add_subplot(gs[0, 1])  # middle (sigma)
    ax_betas = fig.add_subplot(gs[0, 2])  # right (betas, double width)

    # ---------------- ALPHA ----------------
    famA_ref, refA_params = _get_base_prior_spec(cfg, "alpha")
    comp_a = _get_component_cfg(cfg, "alpha")
    ranges_a = comp_a["parameters_box_range"]["ranges"]
    famA_ms = rows_all["alpha"]["family"]
    msA_params = rows_all["alpha"]["params"]

    xA_rng = _auto_x_range_from_ref(famA_ref, refA_params) if x_alpha is None else x_alpha
    xA = _linspace_pad(*xA_rng)
    pdf_ref_a = _make_pdf(famA_ref, refA_params)
    pdf_ms_a  = _make_pdf(famA_ms,  msA_params)

    ax_alpha.plot(xA, pdf_ref_a(xA), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    cand_colors = _cloud_colors(sample_n_alpha, skip_first=2)
    for (p, c) in zip(_sample_param_sets(ranges_a, sample_n_alpha, rng), cand_colors):
        pdf_c = _make_pdf(comp_a["family"], p)
        ax_alpha.plot(xA, pdf_c(xA), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax_alpha.plot(xA, pdf_ms_a(xA), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_alpha.set_ylabel(_latex_name(plot_cfg, "nonparametric_prior", "prior"))
    ax_alpha.set_title(
        _latex_name(plot_cfg, "alpha", r"$\alpha$")
    )
    ax_alpha.spines["top"].set_visible(False)
    ax_alpha.spines["right"].set_visible(False)

    # ---------------- SIGMA ----------------
    famS_ref, refS_params = _get_base_prior_spec(cfg, "sigma")  # HalfCauchy(gamma=2.5) from data.base_prior
    comp_s = _get_component_cfg(cfg, "sigma")                   # candidates from Gamma ranges (per cfg)
    ranges_s = comp_s["parameters_box_range"]["ranges"]
    famS_ms = rows_all["sigma"]["family"]                       # e.g., "Gamma"
    msS_params = rows_all["sigma"]["params"]

    xS_rng = _auto_x_range_from_ref(famS_ref, refS_params) if x_sigma is None else x_sigma
    xS = _linspace_pad(*xS_rng)
    pdf_ref_s = _make_pdf(famS_ref, refS_params)
    pdf_ms_s  = _make_pdf(famS_ms,  msS_params)

    ax_sigma.plot(xS, pdf_ref_s(xS), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    cand_colors = _cloud_colors(sample_n_sigma, skip_first=2)
    for (p, c) in zip(_sample_param_sets(ranges_s, sample_n_sigma, rng), cand_colors):
        pdf_c = _make_pdf(comp_s["family"], p)
        ax_sigma.plot(xS, pdf_c(xS), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax_sigma.plot(xS, pdf_ms_s(xS), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_sigma.set_title(
        _latex_name(plot_cfg, "sigma", r"$\sigma$")
    )
    ax_sigma.spines["top"].set_visible(False); ax_sigma.spines["right"].set_visible(False)

    # ---------------- BETAS (grouped most-sensitive) ----------------
    beta_names = ["beta1", "beta2", "beta3", "beta4", "beta5"]
    beta_group_styles = ["-", ":", "--", "-."]

    # Shared Π_ref from base_prior (assume same family/params across β’s; use beta1 as source)
    famB_ref, refB_params = _get_base_prior_spec(cfg, "beta1")
    xB_rng = _auto_x_range_from_ref(famB_ref, refB_params) if x_beta is None else x_beta
    xB = _linspace_pad(*xB_rng)
    pdf_ref_b = _make_pdf(famB_ref, refB_params)
    ax_betas.plot(xB, pdf_ref_b(xB), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")

    # candidate clouds from ranges of each beta
    for bname in beta_names:
        comp_b = _get_component_cfg(cfg, bname)
        ranges_b = comp_b["parameters_box_range"]["ranges"]
        fam_b_cand = comp_b["family"]
        cand_colors = _cloud_colors(sample_n_beta, skip_first=2)
        for (p, c) in zip(_sample_param_sets(ranges_b, sample_n_beta, rng), cand_colors):
            pdf_c = _make_pdf(fam_b_cand, p)
            ax_betas.plot(xB, pdf_c(xB), linewidth=0.75, alpha=alpha_cloud, color=c)

    ax_betas.set_title(r"$\beta_{1 \cdots 5}$", pad=-9)
    # ax_betas.set_ylabel(_latex_name(plot_cfg, "nonparametric_prior", "prior"))
    ax_betas.spines["top"].set_visible(False); ax_betas.spines["right"].set_visible(False)

    # group equal most-sensitive betas and draw each group with distinct red linestyle
    groups = _group_beta_most_sensitive(rows_all, beta_names)
    legend_handles = []
    legend_labels  = []

    # Π_ref (only once)
    legend_handles.append(plt.Line2D([], [], color=col_ref, linestyle="--", linewidth=1.2))
    legend_labels.append(r"$\Pi_{\mathrm{ref}}$")

    legend_handles.append(plt.Line2D([], [], color=col_red, linestyle="-", linewidth=1.2))
    legend_labels.append(
        r"$\Pi$ (sup$_{\gamma \in C_\gamma} L_m(\gamma)$)"
    )

    for gi, (names, fam_g, params_g) in enumerate(groups):
        style = beta_group_styles[min(gi, len(beta_group_styles)-1)]
        pdf_ms = _make_pdf(fam_g, params_g)
        ax_betas.plot(xB, pdf_ms(xB), color=col_red, linestyle=style, linewidth=1.8)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        frameon=False,
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, -0.23)
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    _save_fig(fig, output_dir, filename, plot_cfg)


def _prepare_inputs(y, posterior_samples_init, K):
    y = np.asarray(y).squeeze()
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (got {y.shape}).")
    if posterior_samples_init.ndim != 2:
        raise ValueError("posterior_samples_init must be 2D (S, K+2).")
    S, P = posterior_samples_init.shape
    if P != (K + 2):
        raise ValueError(f"Expected {K+2} columns, got {P}.")
    if len(y) < K + 1:
        raise ValueError(f"y length must be ≥ {K+1}, got {len(y)}.")
    return y, posterior_samples_init


def _compute_in_sample_means(y, alpha, betas):
    K, N = len(betas), len(y)
    mu = np.full(N, np.nan)
    for t in range(K, N):
        mu[t] = alpha + np.dot(betas, y[t - np.arange(1, K + 1)])
    return mu


def _scatter_matrix(params, labels, cfg):
    M = params.shape[1]
    scale = _deep_get(cfg, "ar.fig4.figsize_scale", 2.6)
    bins = _deep_get(cfg, "ar.fig4.hist_bins", 40)
    s = _deep_get(cfg, "ar.fig4.point_size", 4)
    a = _deep_get(cfg, "ar.fig4.scatter_alpha", 0.6)

    fig, axes = plt.subplots(M, M, figsize=(scale*M, scale*M))
    cols = _palette(cfg, 1)
    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            if i == j:
                ax.hist(params[:, j], bins=bins)
                ax.set_title(f"(med={np.median(params[:, j]):.3f})")
            elif i > j:
                ax.scatter(params[:, j], params[:, i], s=s, alpha=a, c=cols[0])
            else:
                ax.axis("off")
            if i == M-1: ax.set_xlabel(labels[j])
            if j == 0: ax.set_ylabel(labels[i])
    if _deep_get(cfg, "plot.figure.tight_layout", True): fig.tight_layout()
    return fig


def plot_ar_results(y, posterior_samples_init, K=5, plot_cfg=None, save=False, output_dir=None, prefix="ar"):
    if plot_cfg: _apply_plot_rc(plot_cfg)
    show_int = _deep_get(plot_cfg, "ar.show_intervals", True)
    interval = _deep_get(plot_cfg, "ar.interval", 0.9)
    legend_loc = _deep_get(plot_cfg, "plot.legend.loc", "best")
    tight = _deep_get(plot_cfg, "plot.figure.tight_layout", True)

    y, posterior_samples_init = _prepare_inputs(y, posterior_samples_init, K)
    N, S = len(y), posterior_samples_init.shape[0]
    a_s, b_s, s_s = posterior_samples_init[:,0], posterior_samples_init[:,1:1+K], posterior_samples_init[:,-1]
    a_med, b_med, s_med = np.median(a_s), np.median(b_s,0), np.median(s_s)
    mu_med = _compute_in_sample_means(y, a_med, b_med)

    low = high = None
    if show_int and S>1:
        mu_all = np.array([_compute_in_sample_means(y,a,b) for a,b in zip(a_s,b_s)])
        q = (1-interval)/2
        low, high = np.nanquantile(mu_all,[q,1-q],0)

    figs = {}
    cols = _palette(plot_cfg,3)
    n_alpha = _latex_name(plot_cfg,"alpha","$\\alpha$")
    n_sigma = _latex_name(plot_cfg,"sigma","$\\sigma$")

    # fig1
    f1,ax1=_new_fig_ax(plot_cfg)
    t=np.arange(N)
    ax1.plot(t,y,label=_deep_get(plot_cfg,"ar.fig1.label_y","Observed y"),c=cols[0])
    ax1.plot(t,mu_med,label=_deep_get(plot_cfg,"ar.fig1.label_mu","Posterior mean"),c=cols[1])
    if low is not None: ax1.fill_between(t,low,high,alpha=_deep_get(plot_cfg,"ar.fig1.band_alpha",0.25),label=_deep_get(plot_cfg,"ar.fig1.band_label",f"{int(interval*100)}% band"))
    ax1.set(xlabel=_deep_get(plot_cfg,"ar.fig1.x_label","t"),ylabel=_deep_get(plot_cfg,"ar.fig1.y_label","y"))
    ax1.legend(loc=legend_loc)
    if tight:f1.tight_layout(); figs["fig1"]=f1

    # fig2
    res=y[~np.isnan(mu_med)]-mu_med[~np.isnan(mu_med)]
    f2,ax2=_new_fig_ax(plot_cfg)
    ax2.plot(np.arange(len(res)),res,c=cols[2])
    ax2.axhline(0,ls="--")
    ax2.set(xlabel=_deep_get(plot_cfg,"ar.fig2.x_label",f"Index t={K}..{N-1}"),ylabel=_deep_get(plot_cfg,"ar.fig2.y_label","Residual"))
    if tight:f2.tight_layout(); figs["fig2"]=f2

    # fig3
    names=[n_alpha]+[_latex_name(plot_cfg,f"beta_{i}",f"$\\beta_{{{i}}}$") for i in range(1,K+1)]+[n_sigma]
    samps=[a_s]+[b_s[:,i] for i in range(K)]+[s_s]
    bins=_deep_get(plot_cfg,"ar.fig3.bins",50); ncols=_deep_get(plot_cfg,"ar.fig3.ncols",4); nrows=math.ceil(len(samps)/ncols)
    f3,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,2.5*nrows))
    axes=np.ravel(axes)
    for ax,n,s in zip(axes,names,samps): ax.hist(s,bins=bins); ax.set_title(f"{n} (med={np.median(s):.3f})")
    for ax in axes[len(samps):]: ax.axis("off")
    if tight:f3.tight_layout(); figs["fig3"]=f3

    # fig4
    params=np.column_stack([a_s,b_s,s_s])
    labels=[n_alpha]+[_latex_name(plot_cfg,f"beta_{i}",f"$\\beta_{{{i}}}$") for i in range(1,K+1)]+[n_sigma]
    f4=_scatter_matrix(params,labels,plot_cfg); figs["fig4"]=f4

    for f in figs.values():
        for ax in f.get_axes():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if save:
        out=output_dir or _deep_get(plot_cfg,"plot.output.dir","plots")
        _save_fig(f1,out,_deep_get(plot_cfg,"ar.output.filenames.fig1",f"{prefix}_series.pdf"),plot_cfg)
        _save_fig(f2,out,_deep_get(plot_cfg,"ar.output.filenames.fig2",f"{prefix}_residuals.pdf"),plot_cfg)
        _save_fig(f3,out,_deep_get(plot_cfg,"ar.output.filenames.fig3",f"{prefix}_marginals.pdf"),plot_cfg)
        _save_fig(f4,out,_deep_get(plot_cfg,"ar.output.filenames.fig4",f"{prefix}_pairs.pdf"),plot_cfg)

    return figs


def plot_complexity_bar(
    cfg: dict,
    plot_cfg,
    output_dir: str,
    m: int,
    D: int,
    P: int,
    H_total: int | None = None,
    prefix: str = "ark_param",
    filename: str | None = None,
    nuts_exponent: float = 5/4,
    use_log10: bool = True,
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_complexity_bar.pdf"

    # --- Infer H_total if not provided
    comps = cfg.get("components", [])
    if H_total is None:
        H_total = 0
        for c in comps:
            ranges = c.get("parameters_box_range", {}).get("ranges", {})
            H_total += len(ranges)

    Gprod = 1
    for c in comps:
        nums = c.get("parameters_box_range", {}).get("nums", {})
        gk = 1
        for v in nums.values():
            gk *= int(v)
        Gprod *= gk

    # --- Our method terms
    corners = 2 ** H_total
    cost_ours_data = (m ** 2) * D
    cost_ours_corners = corners * (P ** 2)
    cost_ours_total = cost_ours_data + cost_ours_corners

    # --- MCMC grid term
    nuts_factor = D ** nuts_exponent
    cost_mcmc = Gprod * m * nuts_factor

    # --- Assemble bars
    labels = [
        r"$\mathcal{O}(m^2 D)$ (dominating terms, ours)",
        r"$\mathcal{O}(2^H P^2)$ (ours)",
        "MCMC-based grid-search",
    ]
    values = np.array([cost_ours_data, cost_ours_corners, cost_mcmc], dtype=float)
    heights = np.log10(values) if use_log10 else values
    ylab = r"$\log_{10}(\text{cost})$" if use_log10 else "cost"

    # --- Get palette colors
    try:
        palette = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(palette) < len(labels):
        reps = int(np.ceil(len(labels) / len(palette)))
        palette = (palette * reps)[:len(labels)]

    # --- Plot
    style = "lollipop"  # options: "lollipop", "float_rect", "bar"

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (9, 5),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))

    if style == "bar":
        ax.bar(x, heights, color=palette[:len(labels)], width=0.55)
    elif style == "lollipop":
        # stems
        for xi, yi, c in zip(x, heights, palette):
            ax.vlines(xi, 0, yi, colors=c, linewidth=1.5, alpha=0.9)
        # markers at the top (small rectangles)
        rect_w = 0.28
        rect_h = 0.25 if use_log10 else 0.02 * np.max(heights)
        for xi, yi, c in zip(x, heights, palette):
            ax.add_patch(plt.Rectangle((xi - rect_w / 2, yi - rect_h / 2),
                                       rect_w, rect_h,
                                       facecolor=c, edgecolor=c, alpha=0.9, linewidth=1.5))
    elif style == "float_rect":
        # only floating rectangles (no stems)
        rect_w = 0.35
        rect_h = 0.28 if use_log10 else 0.02 * np.max(heights)
        for xi, yi, c in zip(x, heights, palette):
            ax.add_patch(plt.Rectangle((xi - rect_w / 2, yi - rect_h / 2),
                                       rect_w, rect_h,
                                       facecolor=c, edgecolor=c, alpha=0.2, linewidth=1.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel(r"$\log_{10}(\text{cost})$" if use_log10 else "cost")

    # light styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35)
    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")







