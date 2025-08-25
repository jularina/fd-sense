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
    print(f"Saved: {path}")

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

def _midpoint_params(ranges: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: 0.5 * (float(v[0]) + float(v[1])) for k, v in ranges.items()}

def _make_pdf(family: str, params: Dict[str, float]):
    """Prefer your DISTRIBUTION_MAP, otherwise fallback for Gaussian/Gamma."""
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)
    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f

def _auto_x_range_from_ref(family: str, ref_params: Dict[str, float]):
    low = family.lower()
    if low in ("gaussian", "normal"):
        mu, s = ref_params["mu"], ref_params["sigma"]
        return (mu - 4 * s, mu + 4 * s)
    if low == "gamma":
        a, th = ref_params["alpha"], ref_params["theta"]
        mean = a * th
        sd = np.sqrt(a) * th
        return (0.0, max(1e-8, mean + 6 * sd))
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


def plot_three_panel_priors(
    rows_all: Dict,
    cfg,
    plot_cfg,
    output_dir: str,
    prefix: str = "ark_param",
    sample_n_alpha: int = 30,
    sample_n_sigma: int = 30,
    sample_n_beta: int = 30,      # per-beta cloud count
    seed: int = 123,
    filename: str = None,
    x_alpha=None,                 # optional manual x-ranges: tuples
    x_sigma=None,
    x_beta=None,
):
    """
    Creates a single figure with 3 subplots:
      (row 1) [alpha | sigma]
      (row 2) [betas consolidated]

    - Dashed black: Π_ref (midpoint of cfg ranges)
    - Faint cloud: candidate priors sampled uniformly from cfg ranges
    - Solid red: most sensitive (from rows_all['...']['params'])
    - Bottom panel: shows Π_ref once, cloud across ranges for each β, and five red curves β₁..β₅.

    No titles. Legends on each subplot include Π_ref and most-sensitive lines.
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Colors
    palette_full = list(plot_cfg.plot.color_palette.colors)
    col_ref = "black"
    col_red = "red"

    def _cloud_colors(n: int, skip_first: int = 0) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        if not base:
            base = ["#aaaaaa"]
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    alpha_cloud = 0.18

    # Figure & grid
    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_alpha = fig.add_subplot(gs[0, 0])  # top-left
    ax_sigma = fig.add_subplot(gs[0, 1])  # top-right
    ax_betas = fig.add_subplot(gs[1, :])  # bottom full width

    # ---------------- ALPHA PANEL ----------------
    comp_a = _get_component_cfg(cfg, "alpha")
    fam_a = comp_a["family"]
    ranges_a = comp_a["parameters_box_range"]["ranges"]
    ref_a = _midpoint_params(ranges_a)
    ms_a = rows_all["alpha"]["params"]

    xA = _linspace_pad(*(_auto_x_range_from_ref(fam_a, ref_a) if x_alpha is None else x_alpha))
    pdf_ref_a = _make_pdf(fam_a, ref_a)
    pdf_ms_a = _make_pdf(fam_a, ms_a)

    # reference
    ax_alpha.plot(xA, pdf_ref_a(xA), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    # candidates
    for p, c in zip(_sample_param_sets(ranges_a, sample_n_alpha, rng), _cloud_colors(sample_n_alpha, 2)):
        pdf_c = _make_pdf(fam_a, p)
        ax_alpha.plot(xA, pdf_c(xA), linewidth=0.9, alpha=alpha_cloud, color=c)
    # most sensitive
    ax_alpha.plot(xA, pdf_ms_a(xA), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_alpha.legend(frameon=False)
    ax_alpha.spines["top"].set_visible(False); ax_alpha.spines["right"].set_visible(False)

    # ---------------- SIGMA PANEL (Gamma) ----------------
    comp_s = _get_component_cfg(cfg, "sigma")
    fam_s = comp_s["family"]
    ranges_s = comp_s["parameters_box_range"]["ranges"]
    ref_s = _midpoint_params(ranges_s)
    ms_s = rows_all["sigma"]["params"]

    xS = _linspace_pad(*(_auto_x_range_from_ref(fam_s, ref_s) if x_sigma is None else x_sigma))
    pdf_ref_s = _make_pdf(fam_s, ref_s)
    pdf_ms_s = _make_pdf(fam_s, ms_s)

    ax_sigma.plot(xS, pdf_ref_s(xS), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    for p, c in zip(_sample_param_sets(ranges_s, sample_n_sigma, rng), _cloud_colors(sample_n_sigma, 2)):
        pdf_c = _make_pdf(fam_s, p)
        ax_sigma.plot(xS, pdf_c(xS), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax_sigma.plot(xS, pdf_ms_s(xS), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_sigma.legend(frameon=False)
    ax_sigma.spines["top"].set_visible(False); ax_sigma.spines["right"].set_visible(False)

    # ---------------- BETAS PANEL (β1..β5 consolidated) ----------------
    # assume shared ref across betas (use beta1’s ranges for Π_ref + x-axis)
    comp_b1 = _get_component_cfg(cfg, "beta1")
    fam_b = comp_b1["family"]
    ranges_b = comp_b1["parameters_box_range"]["ranges"]
    ref_b = _midpoint_params(ranges_b)
    xB = _linspace_pad(*(_auto_x_range_from_ref(fam_b, ref_b) if x_beta is None else x_beta))
    pdf_ref_b = _make_pdf(fam_b, ref_b)

    # draw shared reference once
    ax_betas.plot(xB, pdf_ref_b(xB), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")

    # cloud of candidates for each beta
    beta_names = ["beta1", "beta2", "beta3", "beta4", "beta5"]
    for bname in beta_names:
        comp_b = _get_component_cfg(cfg, bname)
        fam_i = comp_b["family"]
        ranges_i = comp_b["parameters_box_range"]["ranges"]
        for p, c in zip(_sample_param_sets(ranges_i, sample_n_beta, rng), _cloud_colors(sample_n_beta, 2)):
            pdf_c = _make_pdf(fam_i, p)
            ax_betas.plot(xB, pdf_c(xB), linewidth=0.75, alpha=alpha_cloud, color=c)

    # most sensitive (five red lines)
    for i, bname in enumerate(beta_names, start=1):
        fam_i = _get_component_cfg(cfg, bname)["family"]
        ms_i = rows_all[bname]["params"]
        pdf_ms_i = _make_pdf(fam_i, ms_i)
        ax_betas.plot(xB, pdf_ms_i(xB), color=col_red, linewidth=1.6, label=fr"$\beta_{i}$")

    ax_betas.legend(frameon=False, ncol=3)
    ax_betas.spines["top"].set_visible(False); ax_betas.spines["right"].set_visible(False)

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






