"""Drugowitsch (2013) mean-field VB logistic regression with the Jaakkola-Jordan bound,
plus the Monte Carlo cross-entropy proxy ELBO evaluated from the same q (Section 5, App. B).

Model: y_n | w ~ Bern(sigmoid(x_n^T w)), w | alpha ~ N(0, alpha^-1 I), alpha ~ Gamma(a0, b0).
"""
import numpy as np
from scipy.special import digamma, gammaln

from .utils import calc_ce, sigmoid


def _lam(xi):
    xi = np.asarray(xi, float)
    out = np.full_like(xi, 0.125)
    nz = np.abs(xi) > 1e-8
    out[nz] = np.tanh(xi[nz] / 2) / (4 * xi[nz])
    return out


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def run_drugowitsch_vb(X, y, p_M, iters=60, mc_draws=30, a0=1e-2, b0=1e-4, seed=0,
                       paper_kl=False):
    """paper_kl=True uses the KL(q(w)||p(w|alpha)) exactly as printed in Appendix B, with
    +p·log E[alpha]. That sign is wrong (the correct term is -p·E[log alpha]); the option
    exists only to recreate the printed Figure 1."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    h_M = calc_ce(y, p_M)

    xi = np.ones(n)
    a_N = a0 + p / 2
    # start q(alpha) from mu = 0, Sigma = 0 (b_N = b0), as in the paper's run:
    # E[alpha] begins at (a0 + p/2)/b0 and shrinks toward its fixed point
    b_N = b0 + 0.0
    E_alpha = a_N / b_N
    t_vec = X.T @ (y - 0.5)
    # common random numbers: the same M x p standard normals are mapped through each
    # iteration's q(w), so the proxy trace is not dominated by draw-to-draw MC noise
    Z = rng.standard_normal((mc_draws, p))
    rows = []
    for it in range(1, iters + 1):
        # q(w)
        prec = E_alpha * np.eye(p) + 2 * (X.T * _lam(xi)) @ X
        Sigma = np.linalg.inv(prec)
        Sigma = 0.5 * (Sigma + Sigma.T)
        mu = Sigma @ t_vec
        # q(alpha)
        b_N = b0 + 0.5 * (mu @ mu + np.trace(Sigma))
        E_alpha = a_N / b_N
        E_log_alpha = digamma(a_N) - np.log(b_N)
        # xi
        m_n = X @ mu
        s_n = m_n ** 2 + np.einsum("ij,jk,ik->i", X, Sigma, X)
        xi = np.sqrt(s_n)

        # KL terms (shared by both ELBOs)
        _, logdet = np.linalg.slogdet(Sigma)
        log_alpha_term = p * np.log(E_alpha) if paper_kl else -p * E_log_alpha
        kl_w = 0.5 * (E_alpha * (mu @ mu + np.trace(Sigma)) - p - logdet + log_alpha_term)
        kl_a = ((a_N - a0) * digamma(a_N) - gammaln(a_N) + gammaln(a0)
                + a0 * (np.log(b_N) - np.log(b0)) + a_N * (b0 - b_N) / b_N)

        # standard (JJ-bounded) data term
        lam = _lam(xi)
        data_std = np.sum(_log_sigmoid(xi) + (y - 0.5) * m_n - xi / 2 - lam * (s_n - xi ** 2))
        elbo_std = data_std - kl_w - kl_a

        # proposed proxy: -n E_q[delta] by Monte Carlo with the exact cross-entropy
        W = mu + Z @ np.linalg.cholesky(Sigma).T
        H = np.array([calc_ce(y, sigmoid(X @ w)) for w in W])
        E_delta = H.mean() - h_M
        elbo_prop = -n * E_delta - kl_w - kl_a

        rows.append(dict(iteration=it, elbo_standard=elbo_std, elbo_proposed=elbo_prop,
                         data_standard=data_std, data_proposed=-n * E_delta,
                         E_delta=E_delta, kl_w=kl_w, kl_alpha=kl_a, E_alpha=E_alpha))
    return dict(trace=rows, mu=mu, sd=np.sqrt(np.diag(Sigma)), Sigma=Sigma,
                E_alpha=E_alpha, a_N=a_N, b_N=b_N, h_M=h_M)
