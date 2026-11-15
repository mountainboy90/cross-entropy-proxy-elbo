"""Multi-chain convergence diagnostics (Section 9.3): rank-normalised split R-hat and bulk ESS
(Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021), implemented in numpy so ArviZ is optional.
"""
import numpy as np
from scipy.stats import norm, rankdata


def _split(chains):
    m, n = chains.shape
    half = n // 2
    return np.vstack([chains[:, :half], chains[:, n - half:]])


def _rank_normalize(chains):
    r = rankdata(chains, method="average").reshape(chains.shape)
    return norm.ppf((r - 0.375) / (chains.size + 0.25))


def _rhat_basic(chains):
    m, n = chains.shape
    W = chains.var(axis=1, ddof=1).mean()
    B = n * chains.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else np.nan


def rhat(chains):
    """chains: (n_chains, n_draws). Max of bulk and folded rank-normalised split R-hat."""
    s = _split(chains)
    bulk = _rhat_basic(_rank_normalize(s))
    folded = _rhat_basic(_rank_normalize(np.abs(s - np.median(s))))
    return max(bulk, folded)


def _autocov(x):
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    return ac / n


def ess(chains):
    """Bulk ESS on rank-normalised split chains with Geyer's initial monotone sequence."""
    s = _rank_normalize(_split(chains))
    m, n = s.shape
    acov = np.array([_autocov(c) for c in s])
    chain_var = acov[:, 0] * n / (n - 1)
    W = chain_var.mean()
    B_over_n = s.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B_over_n
    rho = 1 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum consecutive pairs while positive, enforce monotone
    t, tau = 0, 0.0
    prev = np.inf
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        tau += pair
        prev = pair
        t += 2
    tau = -1 + 2 * tau
    return m * n / max(tau, 1e-12)


def multichain_summary(chain_list, burn, names):
    """chain_list: list of (m, p) arrays. Returns per-coefficient R-hat and ESS."""
    rows = []
    for j, name in enumerate(names):
        c = np.array([ch[burn:, j] for ch in chain_list])
        rows.append(dict(feature=name, rhat=rhat(c), ess_bulk=ess(c)))
    return rows
