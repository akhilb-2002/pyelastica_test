"""
Per-letter final-shape comparison: neural Koopman vs EDMD (RBF).
One figure per letter, single-column width, serif (Times) fonts.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

UP_OURS = Path("dklqr_final")
UP_EDMD = Path("edmd_lqr_runs")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Fonts. 'Times New Roman' first so the real face is used wherever
# it exists; Liberation Serif is metric-compatible and is the
# fallback here.
# ---------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,   # embed as TrueType, not Type 3 (IEEE requirement)
    "ps.fonttype": 42,
})

PAIRS = [
    ("I", "lqr_run_I.npz", "edmd_lqr_run_I.npz"),
    ("C", "lqr_run_C.npz", "edmd_lqr_run_C.npz"),
    ("L", "lqr_run_L.npz", "edmd_lqr_run_L.npz"),
    ("U", "lqr_run_U.npz", "edmd_lqr_U.npz"),
]

C_GOAL = "0.35"
C_OURS = "#0072B2"
C_EDMD = "#D55E00"


def plot_letter(letter, ours_file, edmd_file):
    do = np.load(UP_OURS / ours_file, allow_pickle=True)
    de = np.load(UP_EDMD / edmd_file, allow_pickle=True)

    goal = np.asarray(do["goal_pos"], dtype=float)
    ours = np.asarray(do["final_pos"], dtype=float)
    edmd = np.asarray(de["final_pos"], dtype=float)

    r_ours = float(do["settle_rmse"])
    r_edmd = float(de["settle_rmse"])

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    ax.plot(goal[:, 0], goal[:, 1], ls="--", lw=1.4, color=C_GOAL,
            marker="o", ms=3.0, mfc="none", mec=C_GOAL, mew=0.8,
            label="Goal", zorder=1)

    ax.plot(edmd[:, 0], edmd[:, 1], ls="-", lw=1.4, color=C_EDMD,
            marker="o", ms=3.0, mfc=C_EDMD, mec=C_EDMD,
            label="EDMD (RBF)", zorder=2)

    ax.plot(ours[:, 0], ours[:, 1], ls="-", lw=1.4, color=C_OURS,
            marker="o", ms=3.0, mfc=C_OURS, mec=C_OURS,
            label="Ours", zorder=3)

    # anchor node (index 29) marked once; it is common to all three
    ax.plot(goal[-1, 0], goal[-1, 1], marker="s", ms=5.5, mfc="k", mec="k",
            ls="none", zorder=5)

    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.4, alpha=0.35)

    # Legend below the axes: the EDMD shapes wander far enough that no
    # in-axes corner is reliably free across all four letters.
    ax.legend(frameon=False, ncol=3, handlelength=1.6, columnspacing=1.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.20))

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"shape_{letter}.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"{letter}: ours {r_ours:8.3f} mm | EDMD {r_edmd:8.3f} mm")


def plot_rmse(letter, ours_file, edmd_file):
    do = np.load(UP_OURS / ours_file, allow_pickle=True)
    de = np.load(UP_EDMD / edmd_file, allow_pickle=True)

    ro = np.asarray(do["rmse"], dtype=float)
    re = np.asarray(de["rmse"], dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    ax.plot(np.arange(re.size), re, ls="-", lw=1.4, color=C_EDMD,
            label="EDMD (RBF)", zorder=2)

    ax.plot(np.arange(ro.size), ro, ls="-", lw=1.4, color=C_OURS,
            label="Ours", zorder=3)

    ax.set_xlabel("Control step")
    ax.set_ylabel("RMSE [mm]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.4, alpha=0.35)

    ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=1.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.20))

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"rmse_{letter}.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def plot_command(letter, ours_file):
    do = np.load(UP_OURS / ours_file, allow_pickle=True)

    cmd = np.asarray(do["command"], dtype=float)
    nc = int(do["n_control"])
    u_max = float(do["u_max"])

    # Control phase only: during the hold phase u = 0 by construction, so
    # including it appends a flat zero tail (600 steps for L) that carries
    # no information and compresses the part that does.
    cmd = cmd[:nc]
    k = np.arange(nc)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    ax.axhline(u_max, ls=":", lw=0.9, color="0.5", zorder=1)
    ax.axhline(-u_max, ls=":", lw=0.9, color="0.5", zorder=1)

    ax.plot(k, cmd[:, 0], ls="-", lw=1.4, color=C_OURS,
            label="$u_x$", zorder=3)
    ax.plot(k, cmd[:, 1], ls="-", lw=1.4, color=C_EDMD,
            label="$u_y$", zorder=2)

    ax.set_xlabel("Control step")
    ax.set_ylabel("Command [mm/step]")
    ax.set_xlim(0, nc - 1)
    ax.set_ylim(-1.15 * u_max, 1.15 * u_max)
    ax.grid(True, lw=0.4, alpha=0.35)

    ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=1.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.20))

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cmd_{letter}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


for letter, a, b in PAIRS:
    plot_letter(letter, a, b)
    plot_rmse(letter, a, b)
    plot_command(letter, a)

print("\nWrote 12 PDF + 12 PNG to", OUT)