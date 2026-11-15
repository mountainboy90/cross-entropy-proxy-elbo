"""The two surrogate samplers of Section 3 (Algorithms 1 and 2) and the hyperprior (Eq. 15).

Both target  p(beta | Y) ∝ Bernoulli-logistic likelihood × N(0, s1 s2 I) prior, where
s1 ~ InvGamma(a, b), s2 ~ InvGamma(c, d) are redrawn from their fixed hyperprior every
iteration. They differ only in the label (Y1 for MH-MCMC, y for MALA) and in the proposal.
"""
import time

import numpy as np

from .utils import calc_ce, sigmoid


def build_hyperprior(n, p, h_lr_y1, h_m_y, ce_floor=0.0):
    """Eq. (15): a = c = (n-p)/2, b = a·H(Y1, p^LR), d = a·H(y, p^M), with an optional floor."""
    a = c = 0.5 * (n - p)
    b = a * max(h_lr_y1, ce_floor)
    d = a * max(h_m_y, ce_floor)
    return dict(a=a, b=b, c=c, d=d)


def _loglik(Y, z):
    # sum_i [Y z - log(1 + e^z)], numerically stable
    return float(np.sum(Y * z - np.logaddexp(0.0, z)))


def _log_normal(x, mean, sd):
    return float(np.sum(-0.5 * np.log(2 * np.pi * sd * sd) - (x - mean) ** 2 / (2 * sd * sd)))


def _draw_scales(rng, hp, m):
    s1 = 1.0 / rng.gamma(hp["a"], 1.0 / hp["b"], size=m)
    s2 = 1.0 / rng.gamma(hp["c"], 1.0 / hp["d"], size=m)
    return s1, s2


def mh_mcmc(Y, X, hp, m, seed, ridge=1e-8, beta0=None, verbose=True):
    """Algorithm 1: Bayesian LR by random-walk Metropolis-Hastings (fit to Y = Y1)."""
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    s1, s2 = _draw_scales(rng, hp, m)
    L = np.linalg.cholesky(np.linalg.inv(X.T @ X + ridge * np.eye(p))).T   # upper factor

    beta = np.zeros((m, p))
    loglik = np.zeros(m)
    cur = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    beta[0] = cur
    ll_cur = _loglik(Y, X @ cur)
    loglik[0] = ll_cur
    acc = 0
    for t in range(1, m):
        tau = np.sqrt(s1[t] * s2[t])
        prop = cur + (rng.standard_normal(p) * tau) @ L
        ll_prop = _loglik(Y, X @ prop)
        rho = (ll_prop + _log_normal(prop, 0, tau)) - (ll_cur + _log_normal(cur, 0, tau))
        if np.log(rng.uniform()) < rho:
            cur, ll_cur = prop, ll_prop
            acc += 1
        beta[t] = cur
        loglik[t] = ll_cur
    rate = acc / (m - 1)
    if verbose:
        print(f"    MH-MCMC: m={m}, acceptance {100 * rate:.1f}%")
    return dict(beta=beta, loglik=loglik, tau2=s1 * s2, accept=rate,
                runtime=time.perf_counter() - t0)


def mala(Y, X, hp, eta, m, seed, p_M=None, beta0=None, verbose=True, Y_grad=None, drift="paper"):
    """Algorithm 2: cross-entropy-directed MALA (fit to Y = y).

    Proposal  beta' = beta - eta·∇H(y, p^s(beta)) + N(0, 2·eta·I),  ∇H = X^T(p^s - y)/n,
    accepted under the exact log-posterior  -n·H(y, p^s) + log π(beta)  with the
    asymmetric proposal-density correction.  Also records the gradient and the surrogate
    closeness delta(beta_t) = H(y, p^s(beta_t)) - H(y, p^M) at every retained state.

    Y_grad (optional) is a separate label for the drift only: the proposal then follows
    X^T(p^s - Y_grad)/n while the likelihood and accept/reject still use Y. The proposal
    densities use the same drift, so the chain still targets the Y posterior exactly.

    drift="paper" (default) is Algorithm 2 as printed: mean beta - eta·g(beta), noise var 2·eta.
    drift="full" is textbook MALA on the full log-posterior: mean beta - (eta/2)(n·g(beta) + beta/tau²),
    noise var eta, where tau² is that iteration's prior variance (used for both directions).
    The recorded `grad` is g(beta) = X^T(p^s - Y_grad)/n in both cases.
    """
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Yg = Y if Y_grad is None else np.asarray(Y_grad, float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    s1, s2 = _draw_scales(rng, hp, m)
    if drift not in ("paper", "full"):
        raise ValueError(drift)
    noise_sd = np.sqrt(2 * eta) if drift == "paper" else np.sqrt(eta)

    def step_mean(b, g, tau):
        if drift == "paper":
            return b - eta * g
        return b - 0.5 * eta * (n * g + b / tau ** 2)

    h_M = calc_ce(Y, p_M) if p_M is not None else 0.0

    def state(b):
        z = X @ b
        ps = sigmoid(z)
        return _loglik(Y, z), X.T @ (ps - Yg) / n

    beta = np.zeros((m, p))
    grad = np.zeros((m, p))
    loglik = np.zeros(m)
    cur = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    ll_cur, g_cur = state(cur)
    beta[0], grad[0], loglik[0] = cur, g_cur, ll_cur
    acc = 0
    for t in range(1, m):
        tau = np.sqrt(s1[t] * s2[t])
        mean_fwd = step_mean(cur, g_cur, tau)
        prop = mean_fwd + noise_sd * rng.standard_normal(p)
        ll_prop, g_prop = state(prop)
        rho_tgt = (ll_prop + _log_normal(prop, 0, tau)) - (ll_cur + _log_normal(cur, 0, tau))
        rho_prop = (_log_normal(cur, step_mean(prop, g_prop, tau), noise_sd)
                    - _log_normal(prop, mean_fwd, noise_sd))
        if np.log(rng.uniform()) < rho_tgt + rho_prop:
            cur, ll_cur, g_cur = prop, ll_prop, g_prop
            acc += 1
        beta[t], grad[t], loglik[t] = cur, g_cur, ll_cur
    rate = acc / (m - 1)
    delta = -loglik / n - h_M          # H(y, p^s) = -loglik/n
    if verbose:
        print(f"    MALA ({drift} drift): m={m}, eta={eta:g}, acceptance {100 * rate:.1f}%")
    return dict(beta=beta, grad=grad, loglik=loglik, delta=delta, tau2=s1 * s2, accept=rate,
                runtime=time.perf_counter() - t0)


def summarize_chain(beta, burn):
    post = beta[burn:]
    return dict(mean=post.mean(0), sd=post.std(0, ddof=1),
                lo=np.quantile(post, 0.025, axis=0), hi=np.quantile(post, 0.975, axis=0))
