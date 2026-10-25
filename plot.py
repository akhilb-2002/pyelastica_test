"""
plot_coverage.py

Plot the workspace coverage of the training data: logged rope configurations overlaid,
optionally with a density heatmap underneath, plus any goal shapes so you can see at a
glance whether a target sits inside the region the model was trained on.

Modes
-----
    spaghetti  overlaid rope polylines, one translucent line per configuration
    heatmap    2D histogram of every node position (fast, handles millions of points)
    both       heatmap underlay + spaghetti on top          [default]

Why both: the spaghetti plot shows the SHAPES the rope took; the heatmap shows where
it spent its TIME. A goal can sit inside the spaghetti envelope while landing in a
region the data barely visited, which is the situation that produces a large
reconstruction floor and a controller with no gradient to follow.

Row filters
-----------
    (none)         every logged row
    --final-only   last row of each trajectory
    --settled-only last row of each trajectory that is also at rest

--settled-only is the one to judge letter reachability with: a letter has to be a
pose the rope can REST in, not merely one it passes through mid-swing. --final-only
keeps ramp endpoints that may still be ringing.

Usage
-----
    python plot_coverage.py --csv 'case_*/collected_*.csv'
    python plot_coverage.py --csv 'case_*/*.csv' --shape 'rope_all_shapes/rope_*.json'
    python plot_coverage.py --csv 'case_*/*.csv' --settled-only --alpha 0.3
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

ROPE_LENGTH_MM = 300.0


# --------------------------------------------------------------------------- #
def load_positions(csv_paths):
    """-> pos [R, N, 2] mm, vel [R, N, 2] mm/s or None, traj [R], case [R], names."""
    frames, names = [], []
    for i, p in enumerate(csv_paths):
        d = pd.read_csv(p)
        d["case_id"] = i
        # offset trajectory ids so ids restarting at 0 per file do not collide
        d["trajectory_id"] = d["trajectory_id"].astype(np.int64) + i * 1_000_000
        frames.append(d)
        names.append(Path(p).parent.name or Path(p).stem)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["case_id", "trajectory_id", "action_idx", "step_idx"],
                        kind="stable").reset_index(drop=True)

    n = len([c for c in df.columns if re.match(r"^pos_x_\d+$", c)])
    pos = np.stack([df[[f"pos_x_{i}" for i in range(n)]].to_numpy(np.float64),
                    df[[f"pos_y_{i}" for i in range(n)]].to_numpy(np.float64)], axis=2)

    vel = None
    if f"vel_x_0" in df.columns:
        vel = np.stack([df[[f"vel_x_{i}" for i in range(n)]].to_numpy(np.float64),
                        df[[f"vel_y_{i}" for i in range(n)]].to_numpy(np.float64)], axis=2)

    return (pos, vel, df["trajectory_id"].to_numpy(np.int64),
            df["case_id"].to_numpy(np.int64), names)


def load_shapes(shape_paths):
    out = []
    for p in shape_paths:
        nodes = np.asarray(json.load(open(p))["nodes"], dtype=np.float64)
        out.append((Path(p).stem, nodes))
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Plot training-data workspace coverage.")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="Training CSVs, space separated. Globs are expanded if your "
                         "shell did not (quote them to let this script do it).")
    ap.add_argument("--shape", nargs="*", default=[],
                    help="Goal shape JSONs to overlay.")
    ap.add_argument("--mode", default="both", choices=["spaghetti", "heatmap", "both"])
    ap.add_argument("--final-only", action="store_true",
                    help="Keep only the last configuration of each trajectory.")
    ap.add_argument("--settled-only", action="store_true",
                    help="Keep only trajectory-final configurations that are at rest "
                         "(max|v| < --settle-vel). Implies --final-only.")
    ap.add_argument("--settle-vel", type=float, default=5.0,
                    help="mm/s threshold for --settled-only.")
    ap.add_argument("--max-configs", type=int, default=6000,
                    help="Configurations drawn in spaghetti mode. All kept rows feed "
                         "the heatmap regardless; this only caps the line drawing, "
                         "which is what actually costs time.")
    ap.add_argument("--color-by", default="tip_y",
                    choices=["tip_y", "tip_angle", "case", "trajectory", "none"],
                    help="What varies the line colour.")
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="Per-line alpha. Raise it a lot with --final-only / "
                         "--settled-only: you go from ~10^5 lines to ~10^2.")
    ap.add_argument("--bins", type=int, default=420, help="Heatmap bins per axis.")
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--no-reach", action="store_true",
                    help="Skip the reach circle around the anchor.")
    ap.add_argument("--out", default="training_coverage.png")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = []
    for c in args.csv:
        paths.extend(sorted(glob.glob(c)) or [c])
    shapes = []
    for s in args.shape:
        shapes.extend(sorted(glob.glob(s)) or [s])

    pos, vel, traj, case, case_names = load_positions(paths)
    R_all, N, _ = pos.shape
    print(f"{len(paths)} file(s) | {R_all} configurations | {N} nodes | "
          f"{len(np.unique(traj))} trajectories")

    # ---- row filter --------------------------------------------------------
    # The anchor is computed from the FULL set before filtering, so it is the same
    # reference point regardless of which rows end up plotted.
    anchor = pos[:, -1, :].mean(axis=0)
    anchor_wander = np.linalg.norm(pos[:, -1, :] - anchor, axis=1).max()
    print(f"anchor at ({anchor[0]:.1f}, {anchor[1]:.1f}) mm, "
          f"max wander {anchor_wander:.3f} mm")

    label_suffix = "all logged states"

    if args.final_only or args.settled_only:
        ends = np.flatnonzero(np.r_[np.diff(traj) != 0, True])

        if args.settled_only:
            if vel is None:
                raise SystemExit("--settled-only needs vel_* columns in the CSVs.")
            speed = np.linalg.norm(vel[ends], axis=2).max(axis=1)
            keep = ends[speed < args.settle_vel]
            print(f"final configs: {len(ends)} | at rest "
                  f"(max|v| < {args.settle_vel} mm/s): {len(keep)}")
            if len(keep) == 0:
                raise SystemExit(
                    "No trajectory ends at rest. The collection ramps stop while the "
                    "rope is still moving -- raise --settle-vel or use --final-only.")
            label_suffix = f"settled trajectory endpoints (max|v| < {args.settle_vel} mm/s)"
        else:
            keep = ends
            print(f"final configs only: {len(keep)} rows (one per trajectory)")
            label_suffix = "trajectory endpoints (may still be ringing)"

        pos, traj, case = pos[keep], traj[keep], case[keep]

    R = len(pos)

    fig, ax = plt.subplots(figsize=(11, 9))

    # ---- heatmap: every node of every kept row -----------------------------
    if args.mode in ("heatmap", "both"):
        pts = pos.reshape(-1, 2)
        h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=args.bins)
        # LogNorm because occupancy spans orders of magnitude: the rope sits near
        # its rest shape most of the time and visits the extremes rarely. On a
        # linear scale the whole plot is one bright line and nothing else.
        im = ax.pcolormesh(xe, ye, h.T, cmap=args.cmap,
                           norm=LogNorm(vmin=1, vmax=max(h.max(), 2)),
                           shading="auto",
                           alpha=1.0 if args.mode == "heatmap" else 0.55)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.045)
        cb.set_label("node visits per bin (log)")
        print(f"heatmap over {len(pts):,} node positions")

    # ---- spaghetti: overlaid configurations --------------------------------
    if args.mode in ("spaghetti", "both"):
        if R > args.max_configs:
            idx = np.random.RandomState(args.seed).choice(R, args.max_configs,
                                                          replace=False)
        else:
            idx = np.arange(R)
        sub = pos[idx]

        if args.color_by == "tip_y":
            c = sub[:, 0, 1]
        elif args.color_by == "tip_angle":
            d = sub[:, 0, :] - anchor
            c = np.arctan2(d[:, 1], d[:, 0])
        elif args.color_by == "case":
            c = case[idx].astype(float)
        elif args.color_by == "trajectory":
            c = traj[idx].astype(float)
        else:
            c = None

        # LineCollection, not a loop of ax.plot: thousands of separate Line2D
        # artists take minutes to render and blow up the output file.
        lc = LineCollection(sub, linewidths=0.6, alpha=args.alpha,
                            cmap="viridis" if args.mode == "both" else args.cmap)
        if c is not None:
            lc.set_array(c)
        else:
            lc.set_color("0.25")
        ax.add_collection(lc)
        print(f"drew {len(sub)} configurations (of {R}), coloured by {args.color_by}")

    # ---- reach circle ------------------------------------------------------
    # Anything outside this cannot be reached by ANY rope configuration, so a goal
    # crossing it is geometrically impossible before dynamics enter the picture.
    if not args.no_reach:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(anchor[0] + ROPE_LENGTH_MM * np.cos(th),
                anchor[1] + ROPE_LENGTH_MM * np.sin(th),
                "w--", lw=1.2, alpha=0.8, zorder=4)
        ax.plot([], [], "w--", lw=1.2, label=f"reach ({ROPE_LENGTH_MM:.0f} mm)")

    # ---- goal shapes -------------------------------------------------------
    for name, nodes in load_shapes(shapes):
        d = np.linalg.norm(nodes - anchor, axis=1).max()
        reachable = d <= ROPE_LENGTH_MM + 1e-6
        ax.plot(nodes[:, 0], nodes[:, 1], "-", lw=3, zorder=7,
                color="red" if reachable else "magenta",
                label=name + ("" if reachable else "  (OUT OF REACH)"))
        flag = "ok" if reachable else f"OUT OF REACH by {d - ROPE_LENGTH_MM:.1f} mm"
        print(f"  goal {name:<14} max dist from anchor {d:7.1f} mm   {flag}")

    ax.scatter(*anchor, s=90, marker="s", c="lime", edgecolors="k", zorder=8,
               label="anchor (fixed node)")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title(f"Training-data coverage — {R:,} configurations\n{label_suffix}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
    ax.autoscale_view()

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"\nSaved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()