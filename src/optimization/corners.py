from typing import Dict, List, Tuple
import numpy as np
import math

Record = Dict[str, object]


# --------- helpers ---------

def _endpoints(r: Tuple[float, float]) -> Tuple[float, float]:
    a, b = float(r[0]), float(r[1])
    return (a, b)

def _make_rec(params: Dict[str, float], eta: np.ndarray) -> Record:
    return {"params": params, "eta": np.asarray(eta, dtype=float).ravel()}

def _cartesian_corners(param_endpoints: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
    """
    Build all 2^k endpoint combinations for k parameters, respecting the input order.
    """
    keys = list(param_endpoints.keys())
    endpoints = [param_endpoints[k] for k in keys]
    recs = []
    # binary loop over corners
    for mask in range(1 << len(keys)):
        params = {}
        for i, k in enumerate(keys):
            lo, hi = endpoints[i]
            params[k] = hi if (mask >> i) & 1 else lo
        recs.append(params)
    return recs


# --------- Gamma(shape α, scale θ) ---------
# pdf(x) ∝ x^(α-1) exp(-x/θ) / (θ^α Γ(α)), x>0
# Natural parameters (minimal): η = [α-1,  -1/θ]
def gamma_corners(ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    a_min, a_max = _endpoints(ranges["alpha"])
    t_min, t_max = _endpoints(ranges["theta"])
    corners = [
        {"alpha": a, "theta": t}
        for a in (a_min, a_max)
        for t in (t_min, t_max)
    ]
    recs = []
    for p in corners:
        eta = np.array([p["alpha"] - 1.0, -1.0 / p["theta"]], dtype=float)
        recs.append(_make_rec(p, eta))
    return recs


# --------- Gaussian/Normal (mean μ, std σ) ---------
# pdf(x) ∝ exp( - (x-μ)^2 / (2σ^2) )
# Natural parameters (minimal): η = [ μ/σ^2,  -1/(2σ^2) ]
def normal_corners(ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    mu_min, mu_max = _endpoints(ranges["mu"])
    s_min, s_max = _endpoints(ranges["sigma"])
    corners = [
        {"mu": mu, "sigma": s}
        for mu in (mu_min, mu_max)
        for s in (s_min, s_max)
    ]
    recs = []
    for p in corners:
        s2 = p["sigma"] ** 2
        eta = np.array([p["mu"] / s2, -0.5 / s2], dtype=float)
        recs.append(_make_rec(p, eta))
    return recs


# --------- Beta(α, β) on (0,1) ---------
# pdf(x) ∝ x^(α-1) (1-x)^(β-1)
# Natural parameters (minimal): η = [ α-1,  β-1 ]
def beta_corners(ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    a_min, a_max = _endpoints(ranges["alpha"])
    b_min, b_max = _endpoints(ranges["beta"])
    corners = [
        {"alpha": a, "beta": b}
        for a in (a_min, a_max)
        for b in (b_min, b_max)
    ]
    recs = []
    for p in corners:
        eta = np.array([p["alpha"] - 1.0, p["beta"] - 1.0], dtype=float)
        recs.append(_make_rec(p, eta))
    return recs


# --------- Inverse-Gamma(shape α, scale β) ---------
# pdf(x) ∝ β^α / Γ(α) * x^{-α-1} * exp(-β/x), x>0
# Sufficient stats: [log x, 1/x]
# Natural parameters (minimal): η = [ -(α+1),  -β ]
def inverse_gamma_corners(ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    a_min, a_max = _endpoints(ranges["alpha"])
    b_min, b_max = _endpoints(ranges["beta"])
    corners = [
        {"alpha": a, "beta": b}
        for a in (a_min, a_max)
        for b in (b_min, b_max)
    ]
    recs = []
    for p in corners:
        eta = np.array([-(p["alpha"] + 1.0), -p["beta"]], dtype=float)
        recs.append(_make_rec(p, eta))
    return recs


# --------- Exponential ---------
# Two common parameterizations:
#   - rate λ > 0: pdf(x) = λ exp(-λ x),  η = [ -λ ]
#   - scale θ = 1/λ > 0: pdf(x) = (1/θ) exp(-x/θ),  η = [ -1/θ ]
# Provide either "rate" or "theta" in ranges.
def exponential_corners(ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    if "rate" in ranges:
        r_min, r_max = _endpoints(ranges["rate"])
        params = [{"rate": r_min}, {"rate": r_max}]
        recs = []
        for p in params:
            eta = np.array([-p["rate"]], dtype=float)
            recs.append(_make_rec(p, eta))
        return recs
    elif "theta" in ranges:
        t_min, t_max = _endpoints(ranges["theta"])
        params = [{"theta": t_min}, {"theta": t_max}]
        recs = []
        for p in params:
            eta = np.array([-1.0 / p["theta"]], dtype=float)
            recs.append(_make_rec(p, eta))
        return recs
    else:
        raise KeyError("exponential_corners: expected 'rate' or 'theta' in ranges")


def lognormal_corners(ranges):
    """
    LogNormal parameterized by (mu, sigma) where log X ~ N(mu, sigma^2).
    Natural parameters (minimal): eta = [ mu/sigma^2,  -1/(2 sigma^2) ].
    ranges: {"mu": (mu_min, mu_max), "sigma": (s_min, s_max)}
    Returns: list of {"params": {...}, "eta": np.ndarray}
    """
    mu_min, mu_max = float(ranges["mu"][0]), float(ranges["mu"][1])
    s_min,  s_max  = float(ranges["sigma"][0]), float(ranges["sigma"][1])

    recs = []
    for mu in (mu_min, mu_max):
        for s in (s_min, s_max):
            s2 = s * s
            eta = np.array([mu / s2, -0.5 / s2], dtype=float)
            recs.append({"params": {"mu": mu, "sigma": s}, "eta": eta})
    return recs


# --------- Generic dispatcher ---------
def get_corners(family: str, ranges: Dict[str, Tuple[float, float]]) -> List[Record]:
    """
    family: one of {"gamma","gaussian","normal","beta","inverse-gamma","inverse_gamma","exponential"}
    ranges: dict of parameter-name -> (min, max)
    returns: list of {"params": ..., "eta": np.ndarray}
    """
    fam = family.strip().lower()
    if fam == "gamma":
        return gamma_corners(ranges)
    if fam in ("gaussian"):
        return normal_corners(ranges)
    if fam == "beta":
        return beta_corners(ranges)
    if fam in ("invgamma"):
        return inverse_gamma_corners(ranges)
    if fam == "exponential":
        return exponential_corners(ranges)
    if fam in ("lognormal"):
        return lognormal_corners(ranges)
    raise ValueError(f"Unsupported family '{family}'.")