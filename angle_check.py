"""
angle_state_check.py

Quick feasibility test for switching the state from Cartesian [x, v] to
[L_i, theta_i, L_dot_i, theta_dot_i].

Runs five checks against your existing CSVs and prints a verdict for each. No training,
no model, no fitting -- about a minute on one case file.

Edge convention
---------------
    e_i = p_{i+1} - p_i,   i = 0 .. N-2      (29 edges for 30 nodes)
    L_i = |e_i|,  theta_i = atan2(e_i.y, e_i.x)

Node 29 is pinned, so positions are recovered by integrating BACKWARD from the anchor:

    p_29 = anchor (constant)
    p_i  = p_{i+1} - L_i * [cos theta_i, sin theta_i]

which means the representation needs no extra state for absolute position -- the anchor
is a constant of the problem, not a variable.

Usage
-----
    python angle_state_check.py --csv case_10/*.csv
"""

import argparse
import re

import numpy as np
import pandas as pd

CONTROL_DT = 0.01


def load(csv_paths):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    n = len([c for c in df.columns if re.match(r"^pos_x_\d+$", c)])
    df = df.sort_values(["trajectory_id", "action_idx", "step_idx"]).reset_index(drop=True)
    pos = np.stack([df[[f"pos_x_{i}" for i in range(n)]].to_numpy(np.float64),
                    df[[f"pos_y_{i}" for i in range(n)]].to_numpy(np.float64)], axis=2)
    vel = np.stack([df[[f"vel_x_{i}" for i in range(n)]].to_numpy(np.float64),
                    df[[f"vel_y_{i}" for i in range(n)]].to_numpy(np.float64)], axis=2)
    traj = df["trajectory_id"].to_numpy(np.int64)
    return pos, vel, traj, n                       # pos, vel: [R, N, 2]


def to_angles(pos):
    e = pos[:, 1:, :] - pos[:, :-1, :]             # [R, N-1, 2]
    L = np.linalg.norm(e, axis=2)                  # [R, N-1]
    th = np.arctan2(e[..., 1], e[..., 0])          # [R, N-1]
    return L, th


def from_angles(L, th, anchor):
    """Integrate backward from the pinned node. anchor: [R, 2] or [2]."""
    R, M = L.shape
    p = np.zeros((R, M + 1, 2))
    p[:, -1, :] = anchor
    e = np.stack([L * np.cos(th), L * np.sin(th)], axis=2)
    for i in range(M - 1, -1, -1):
        p[:, i, :] = p[:, i + 1, :] - e[:, i, :]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=20000,
                    help="rows to subsample for the heavier checks")
    args = ap.parse_args()

    pos, vel, traj, N = load(args.csv)
    L, th = to_angles(pos)
    R = len(pos)
    print(f"{R} rows | {N} nodes | {L.shape[1]} edges | "
          f"angle state dim = {2 * L.shape[1]} (+rates = {4 * L.shape[1]})\n")

    # ---------------------------------------------------------------- 1
    # Invertibility. If this is not ~0 the representation loses information and
    # nothing downstream can recover it.
    anchor = pos[:, -1, :]
    rec = from_angles(L, th, anchor)
    err = np.linalg.norm(rec - pos, axis=2)
    print(f"[1] round-trip pos -> (L,theta) -> pos")
    print(f"    mean {err.mean():.3e} mm | max {err.max():.3e} mm")
    print(f"    {'OK: lossless' if err.max() < 1e-6 else 'PROBLEM: not invertible'}\n")

    # ---------------------------------------------------------------- 2
    # Branch cuts. atan2 returns (-pi, pi]; a rope crossing that cut makes theta jump
    # by 2*pi between consecutive rows. The jump is not physical, but a Koopman
    # operator fitted on raw theta sees a huge discontinuous transition and a finite
    # difference sees an enormous theta_dot.
    d = np.diff(th, axis=0)
    same = traj[1:] == traj[:-1]
    d = d[same]
    jumps = int((np.abs(d) > np.pi).sum())
    print(f"[2] 2*pi wrap events across consecutive rows")
    print(f"    {jumps} of {d.size} transitions | max |d theta| = {np.abs(d).max():.4f} rad")
    if jumps:
        print(f"    PROBLEM: use unwrapped theta per trajectory, or switch to")
        print(f"    (cos theta, sin theta) -- continuous, and camera-constructible")
    else:
        print(f"    OK: no wrapping in this data (but a wider workspace may still wrap)")
    print(f"    max per-step |d theta| in degrees: {np.degrees(np.abs(d).max()):.3f}\n")

    # ---------------------------------------------------------------- 3
    # Is L worth carrying? A Cosserat rod at these loads is nearly inextensible, so
    # the L block may be 29 near-constant columns -- exactly the collinearity that
    # made the pinned anchor's coordinates blow up the EDMD normal matrix.
    rel = L.std(axis=0) / L.mean(axis=0)
    print(f"[3] segment length variation")
    print(f"    mean L = {L.mean():.4f} mm | rest = {300.0 / (N - 1):.4f} mm")
    print(f"    relative std per edge: min {rel.min():.2e}  max {rel.max():.2e}")
    if rel.max() < 1e-3:
        print(f"    L is effectively CONSTANT -> drop it. State becomes theta alone:")
        print(f"    {L.shape[1]} dims (+rates = {2 * L.shape[1]}), vs 120 for [x, v].")
        print(f"    Carrying it adds rank-deficient columns for no information.")
    else:
        print(f"    L varies enough to keep.")
    # what does fixing L at its mean cost you?
    rec_fixed = from_angles(np.broadcast_to(L.mean(axis=0), L.shape), th, anchor)
    e_fixed = np.linalg.norm(rec_fixed - pos, axis=2)
    print(f"    cost of freezing L at its mean: {e_fixed.mean():.4f} mm mean, "
          f"{e_fixed.max():.4f} mm max\n")

    # ---------------------------------------------------------------- 4
    # THE DECISIVE ONE. Angles compose multiplicatively along the chain: an error in
    # theta_28 (at the anchor) rotates every node beyond it, while an error in theta_0
    # moves one node. So position error is NOT uniformly related to angle error, and a
    # cost that is uniform in theta is badly non-uniform in mm.
    idx = np.random.RandomState(0).choice(R, size=min(args.sample, R), replace=False)
    Ls, ths = L[idx], th[idx]
    # d p_i / d theta_j, integrating backward: node i is affected by edges j < i
    # |d p_i / d theta_j| = L_j for j < i, so column j moves (j+1) nodes... take the
    # exact Jacobian norm per edge instead of hand-waving:
    sens = np.zeros(Ls.shape[1])
    for j in range(Ls.shape[1]):
        # rotating edge j by d theta swings nodes 0..j about node j+1
        arm = np.cumsum(Ls[:, j::-1], axis=1)      # distance from node j+1 back to node 0
        sens[j] = np.sqrt((arm ** 2).mean())
    print(f"[4] tip-lever amplification: mm of node motion per rad of edge angle")
    print(f"    edge 0 (at driven tip): {sens[0]:8.2f} mm/rad")
    print(f"    edge {Ls.shape[1]-1:2d} (at anchor)  : {sens[-1]:8.2f} mm/rad")
    print(f"    ratio: {sens[-1] / sens[0]:.1f}x")
    print(f"    -> a Q that is uniform in theta weights anchor-side error "
          f"{sens[-1] / sens[0]:.0f}x too little in mm.")
    print(f"    You would need a Jacobian-weighted Q, which is exactly the "
          f"rank-deficient\n    ARPACK problem you already hit.\n")

    # ---------------------------------------------------------------- 5
    # Rates. theta_dot from logged velocities vs from finite differences. If these
    # disagree the two blocks of the state are telling different stories.
    e = pos[:, 1:, :] - pos[:, :-1, :]
    dv = vel[:, 1:, :] - vel[:, :-1, :]
    cross = e[..., 0] * dv[..., 1] - e[..., 1] * dv[..., 0]
    th_dot_analytic = cross / (L ** 2 + 1e-12)                 # rad/s
    th_u = np.unwrap(th, axis=0)
    th_dot_fd = np.zeros_like(th)
    th_dot_fd[1:] = np.diff(th_u, axis=0) / CONTROL_DT
    m = np.r_[False, same]
    a, b = th_dot_analytic[m], th_dot_fd[m]
    rel_err = np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12)
    print(f"[5] theta_dot: analytic (from logged v) vs finite difference")
    print(f"    mean |analytic| {np.abs(a).mean():.4f} rad/s | "
          f"mean |fd| {np.abs(b).mean():.4f} rad/s")
    print(f"    mean relative disagreement: {rel_err:.3f}")
    print(f"    {'OK' if rel_err < 0.3 else 'LARGE -- expected if the rope rings faster than control_dt'}\n")

    # ---------------------------------------------------------------- verdict
    print("=" * 68)
    print("What this representation costs you, regardless of the numbers above:")
    print("  * Node 0's position becomes a NONLINEAR function of all 29 angles")
    print("    (p0 = anchor - sum L_i [cos th_i, sin th_i]). The exact integrator row")
    print("    you just measured (A[0,0] = 1.0 with p0 literally in the state) is gone.")
    print("  * The MPC cost is squared position error, which is nonlinear in theta,")
    print("    so Q is no longer a constant mask -- back to a Jacobian at z_goal.")
    print("  * Goal letters must be converted to theta, and unreachable-theta targets")
    print("    are harder to spot than unreachable positions.")
    print("What it buys: dimension (29 or 58 vs 120), translation/rotation invariance,")
    print("and a local shape description, which is the Wang et al. local-observable")
    print("argument in a different form.")
    print("=" * 68)


if __name__ == "__main__":
    main()