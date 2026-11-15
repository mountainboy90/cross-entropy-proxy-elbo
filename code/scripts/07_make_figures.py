"""Stage 7: every figure in the paper, written to results/<dataset>/figures/.

  fig_base_learners_vs_ensemble_<ds>      Fig. 2      fig_surrogate_performance_<ds>   Fig. 3
  fig_posterior_density_comparison_<ds>   Fig. 4/A2   fig_burnin_traces_<ds>           Fig. 5/A3
  fig_mala_step_size_sensitivity_<ds>     Fig. 6/A6   fig_mcmc_scaling_vs_y_<ds>       Fig. 7/A7
  fig_shap_summary_bar_<ds>               Fig. 8/A4   fig_shap_dependence_all_features_<ds> Fig. 9/A5
  fig_drugowitsch_elbo_<ds>               Fig. 1/A1   fig_trace_all_coefs_{bayeslr,mala}_<ds> App. C
  fig_gradient_trace_mala_<ds>            App. C      fig_delta_trace_mala_<ds>        App. C
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from _common import parse
from proxyelbo import plotting as P
from proxyelbo.data import load_prepared
from proxyelbo.utils import load_json

args, cfg = parse(__doc__)
ds, out, tab, fig_dir = args.dataset, cfg["OUT"], cfg["TAB"], cfg["FIG"]
P.setup()
plt = P.plt
d = load_prepared(out)
feats = d["features"]
p = len(feats)
burn = cfg["BURN_IN"]
ncol = 6
print(f"[{cfg['NAME']}] figures")


def path(name):
    return fig_dir / f"{name}_{ds}.png"


# ---------------- ensemble vs base learners ----------------
perf = pd.read_csv(tab / "table_ensemble_performance.csv")
te = perf[perf.split == "Test"].reset_index(drop=True)
labels = ["RF", "XGBoost", "MLP", "Soft vote", "Stack"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (col, title) in zip(axes, (("auc", "Test AUC"), ("acc", "Test accuracy"),
                                   ("logloss", "Test log-loss (lower is better)"))):
    colors = [P.C_NEUTRAL] * 4 + [P.C_MALA]
    bars = ax.bar(labels, te[col], color=colors, width=0.62, edgecolor="white", linewidth=2)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, color=P.C_TEXT)
    if col in ("auc", "acc"):
        ax.axhline(0.5, ls="--", lw=1, color=P.C_TEXT)
    ax.set_title(title)
    ax.set_ylim(0, max(te[col]) * 1.18)
    ax.grid(axis="x", visible=False)
P.save(fig, path("fig_base_learners_vs_ensemble"))

# ---------------- surrogate performance ----------------
sp = pd.read_csv(tab / "table_surrogate_performance.csv")
sp = sp[(sp.split == "Test") & (sp.judged_against == "y")].reset_index(drop=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (col, title) in zip(axes, (("auc", "Test AUC vs y"), ("acc", "Test accuracy vs y"),
                                   ("logloss", "Test log-loss vs y (lower is better)"))):
    bars = ax.bar(["Standard LR", "Bayesian LR\n(MH-MCMC)", "MALA\n(CE-directed)"], sp[col],
                  color=[P.SURR_COLORS[s] for s in sp.surrogate], width=0.6,
                  edgecolor="white", linewidth=2)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, color=P.C_TEXT)
    ax.set_title(title)
    ax.set_ylim(0, max(sp[col]) * 1.18)
    ax.grid(axis="x", visible=False)
P.save(fig, path("fig_surrogate_performance"))

# ---------------- posterior densities ----------------
ch = np.load(out / "chains.npz")
mh, ml = ch["mh_beta"], ch["mala_beta"]
coef = pd.read_csv(tab / "table_coefficients.csv")
fig, axes = P.grid(p, ncol, w=2.3, h=1.6)
for j, ax in enumerate(axes):
    a, b = mh[burn::5, j], ml[burn::5, j]
    lo, hi = np.percentile(np.r_[a, b], [0.2, 99.8])
    pad = 0.15 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 200)
    for draws, c, lab in ((a, P.C_MH, "Bayesian LR (MH-MCMC, Y1)"), (b, P.C_MALA, "MALA (y)")):
        k = gaussian_kde(draws)(xs)
        ax.fill_between(xs, k, color=c, alpha=0.18, lw=0)
        ax.plot(xs, k, color=c, lw=1.4, label=lab)
    if ds == "nf1":
        v = coef.standard_lr[j]
        if xs[0] <= v <= xs[-1]:
            ax.axvline(v, color=P.C_LR, lw=1.2, ls="--", label="Standard LR (MLE)")
    ax.axvline(0, color=P.C_TEXT, lw=0.6)
    ax.set_title(feats[j], fontsize=8)
    ax.set_yticks([])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.97))
P.save(fig, path("fig_posterior_density_comparison"))

# ---------------- burn-in traces ----------------
j0 = feats.index(cfg["BURNIN_COEF"])
it = np.arange(burn)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(it, ch["mh_loglik"][:burn], color=P.C_MH, lw=0.8, label=r"Bayesian LR (MH-MCMC), vs $\hat{y}$")
axes[0].plot(it, ch["mala_loglik"][:burn], color=P.C_MALA, lw=0.8, label="MALA, vs y")
axes[0].set_title("Log-likelihood, first 5,000 iterations")
axes[1].plot(it, mh[:burn, j0], color=P.C_MH, lw=0.8)
axes[1].plot(it, ml[:burn, j0], color=P.C_MALA, lw=0.8)
axes[1].set_title(f"Coefficient: {cfg['BURNIN_COEF']}")
for ax in axes:
    ax.set_xlabel("Iteration")
axes[0].legend(loc="lower right")
P.save(fig, path("fig_burnin_traces"))

# ---------------- step-size sensitivity ----------------
f = tab / "table_mala_step_size_sensitivity.csv"
if f.exists():
    st = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(st.eta, st.acceptance, "o-", color=P.C_MALA, ms=5)
    axes[0].axhline(50, ls="--", lw=1, color=P.C_TEXT)
    axes[0].set_ylabel("Acceptance rate (%)")
    axes[1].plot(st.eta, st.mean_post_sd, "o-", color=P.C_MALA, ms=5)
    axes[1].set_ylabel("Mean posterior coef. s.d.")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Step size η (log scale)")
        ax.axvline(cfg["MALA_ETA"], color=P.C_NEUTRAL, lw=1, ls=":")
    P.save(fig, path("fig_mala_step_size_sensitivity"))

# ---------------- chain-length scaling ----------------
f = tab / "table_mcmc_length_scaling.csv"
if f.exists():
    sc = pd.read_csv(f)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    axes = axes.ravel()
    for ax, (k, title) in zip(axes, (("auc", "AUC vs y"), ("acc", "Accuracy vs y"),
                                     ("recall", "Recall vs y"), ("logloss", "Log-loss vs y"))):
        for meth, c in (("MH-MCMC", P.C_MH), ("MALA", P.C_MALA)):
            s = sc[sc.method == meth]
            ax.plot(s.m, s[f"{k}_test_vs_y"], "o-", color=c, ms=5, label=f"{meth} test")
            ax.plot(s.m, s[f"{k}_train_vs_y"], "o--", color=c, ms=4, alpha=0.7, label=f"{meth} train")
        ax.set_title(title)
    for meth, c in (("MH-MCMC", P.C_MH), ("MALA", P.C_MALA)):
        s = sc[sc.method == meth]
        axes[4].plot(s.m, s.coef_rmse_vs_max, "o-", color=c, ms=5)
        axes[5].plot(s.m, s.runtime_s, "o-", color=c, ms=5)
    axes[4].set_title(f"Coef. RMSE vs m={max(sc.m):,} estimate")
    axes[5].set_title("Runtime (s)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Posterior draws m")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    P.save(fig, path("fig_mcmc_scaling_vs_y"))

# ---------------- SHAP ----------------
f = tab / "table_shap_importance.csv"
if f.exists():
    imp = pd.read_csv(f).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * p + 0.8))
    ax.barh(imp.feature, imp.mean_abs_shap, color=P.C_MH, height=0.7, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (impact on ensemble P(y=1))")
    ax.grid(axis="y", visible=False)
    P.save(fig, path("fig_shap_summary_bar"))

    sv = np.load(out / "shap_values.npy")
    raw = pd.read_csv(out / "shap_eval_raw.csv")
    fig, axes = P.grid(p, ncol, w=2.3, h=1.7)
    for j, ax in enumerate(axes):
        x = raw[feats[j]].to_numpy(float)
        jit = 0.04 * np.random.default_rng(j).standard_normal(len(x)) if np.isin(x, [0, 1]).all() else 0
        ax.scatter(x + jit, sv[:, j], s=8, color=P.C_MH, alpha=0.6, lw=0)
        ax.axhline(0, color=P.C_TEXT, lw=0.6)
        ax.set_title(feats[j], fontsize=8)
    fig.supxlabel("Raw covariate value")
    fig.supylabel("SHAP value")
    fig.tight_layout()
    P.save(fig, path("fig_shap_dependence_all_features"))

# ---------------- Drugowitsch ELBO ----------------
# default: correct KL; *_paperkl: Appendix B's printed KL, which recreates the printed Fig. 1
for sfx, name in (("", "fig_drugowitsch_elbo"), ("_paperkl", "fig_drugowitsch_elbo_paperkl")):
    f = tab / f"table_drugowitsch_elbo_trace{sfx}.csv"
    if not f.exists():
        continue
    tr = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for col, c, lab in (("elbo_standard", P.C_MH, "Standard ELBO (Jaakkola–Jordan)"),
                        ("elbo_proposed", P.C_MALA, "Proposed proxy ELBO (MC cross-entropy)")):
        axes[0].plot(tr.iteration, tr[col], color=c, label=lab)
        v = tr[col]
        axes[1].plot(tr.iteration, (v - v.min()) / (v.max() - v.min()), color=c, label=lab)
    axes[0].set_title("Raw scale" + (" (paper's printed KL)" if sfx else ""))
    axes[1].set_title("Min–max normalised")
    for ax in axes:
        ax.set_xlabel("Coordinate-ascent iteration")
    axes[0].set_ylabel("ELBO")
    axes[1].legend(loc="lower right")
    P.save(fig, path(name))

# ---------------- full trace plots (thinned) ----------------
thin = cfg["THIN"]
idx = np.arange(0, mh.shape[0], thin)
for arr, c, name in ((mh, P.C_MH, "fig_trace_all_coefs_bayeslr"), (ml, P.C_MALA, "fig_trace_all_coefs_mala"),
                     (ch["mala_grad"], P.C_MALA, "fig_gradient_trace_mala")):
    fig, axes = P.grid(p, ncol, w=2.3, h=1.4)
    for j, ax in enumerate(axes):
        ax.plot(idx, arr[idx, j], color=c, lw=0.35)
        ax.axvline(burn, color=P.C_TEXT, lw=0.6, ls=":")
        if name.startswith("fig_gradient"):
            ax.axhline(0, color=P.C_TEXT, lw=0.6)
        ax.set_title(feats[j], fontsize=8)
        ax.tick_params(labelsize=6)
    fig.supxlabel("Iteration (every 10th shown)")
    fig.tight_layout()
    P.save(fig, path(name))

delta = ch["mala_delta"]
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(idx, delta[idx], color=P.C_MALA, lw=0.5)
mu = delta[burn:].mean()
ax.axhline(mu, color=P.C_TEXT, lw=1, ls="--", label=f"post-burn-in mean {mu:.3f}")
ax.axhline(0, color=P.C_NEUTRAL, lw=0.6)
ax.axvline(burn, color=P.C_TEXT, lw=0.6, ls=":")
ax.set_xlabel("Iteration (every 10th shown)")
ax.set_ylabel("δ(β) = H(y,pˢ) − H(y,pᴹ)")
ax.legend(loc="upper right")
lo, hi = np.percentile(delta[burn:], [0.5, 99.5])
ax.set_ylim(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo))
P.save(fig, path("fig_delta_trace_mala"))
