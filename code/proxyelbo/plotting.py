"""Shared matplotlib style for the paper figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (validated palette, first three slots are all-pairs safe)
C_MH = "#2a78d6"      # Bayesian LR (MH-MCMC)  - blue
C_MALA = "#eb6834"    # MALA (CE-directed)     - orange
C_LR = "#1baf7a"      # Standard LR            - aqua
C_NEUTRAL = "#8a8985"
C_TEXT = "#52514e"

SURR_COLORS = {"Standard LR": C_LR, "Bayesian LR (MH-MCMC)": C_MH, "MALA (CE-directed)": C_MALA}


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT, "ytick.color": C_TEXT,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def grid(n, ncol, w=2.4, h=1.7, sharex=False):
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), sharex=sharex)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path.name)
