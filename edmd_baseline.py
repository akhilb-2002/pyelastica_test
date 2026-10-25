"""
edmd_baseline.py

EDMD / EDMDc baseline on the same [x, v] state the neural Koopman model uses.

Model
-----
    psi(x_k+1) ~= A psi(x_k) + B u_k

with a FIXED dictionary psi instead of a learned encoder. One least-squares solve,
no training loop, no learning rate, two hyperparameters (--lift, --ridge).

The state itself is always the first block of psi, so the "decoder" is exact:

    x = C psi(x),   C = [I_n  0]

i.e. the reconstruction floor is ZERO. That alone removes the ~10 mm error floor the
learned autoencoder imposes on every tracking number.

Dictionaries (--lift)
---------------------
    state   psi = [x]              -> DMDc, the no-lift baseline
    affine  psi = [x, 1]           -> adds the constant that absorbs gravity / settling
    quad    psi = [x, x*x, 1]      -> elementwise square; grows 2n+1, not n^2
    rbf     psi = [x, 1, phi_i(x)] -> Gaussian RBFs at centres sampled from the data

Note the elementwise square, not a full quadratic: a proper degree-2 polynomial on a
120-d state is 7,381 terms, which is more unknowns than you have transitions.

Usage
-----
    python edmd_baseline.py --csv case_10/*.csv case_11/*.csv --lift affine
    python edmd_baseline.py --csv ... --lift rbf --n-rbf 200 --ridge 1e-4
"""

import argparse
from pathlib import Path

import numpy as np

from ngk_simple import RopeKoopmanDataset


# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #
class Lift:
    """psi(x) with the raw state as its leading block, so C = [I 0] is exact."""

    def __init__(self, kind, state_dim, n_rbf=100, centers=None, sigma=None):
        self.kind = kind
        self.state_dim = state_dim
        self.n_rbf = n_rbf
        self.centers = centers
        self.sigma = sigma

        if kind == "state":
            self.dim = state_dim
        elif kind == "affine":
            self.dim = state_dim + 1
        elif kind == "quad":
            self.dim = 2 * state_dim + 1
        elif kind == "rbf":
            self.dim = state_dim + 1 + n_rbf
        else:
            raise ValueError(f"unknown lift '{kind}'")

    @classmethod
    def fit(cls, kind, X, n_rbf=100, seed=0):
        """Pick RBF centres and bandwidth from the data. No-op for other lifts."""
        state_dim = X.shape[1]
        if kind != "rbf":
            return cls(kind, state_dim)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=min(n_rbf, len(X)), replace=False)
        centers = X[idx]
        # Bandwidth = median pairwise distance between centres. In 120-d, distances
        # concentrate, so a fixed sigma either saturates every RBF to 1 or to 0.
        d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        sigma = np.median(d[d > 0])
        return cls(kind, state_dim, n_rbf=len(centers), centers=centers, sigma=sigma)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, np.float64))
        ones = np.ones((len(X), 1))
        if self.kind == "state":
            return X
        if self.kind == "affine":
            return np.hstack([X, ones])
        if self.kind == "quad":
            return np.hstack([X, X ** 2, ones])
        x2 = (X ** 2).sum(1, keepdims=True)                    # [M, 1]
        c2 = (self.centers ** 2).sum(1)[None, :]               # [1, K]
        d2 = np.maximum(x2 - 2.0 * (X @ self.centers.T) + c2, 0.0)   # [M, K]
        return np.hstack([X, ones, np.exp(-d2 / (2 * self.sigma ** 2))])


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_edmdc(Z, U, Zp, ridge=1e-6):
    """Least squares for [A B] in psi(x_k+1) = A psi(x_k) + B u_k.

    Z, Zp: [M, N] lifted states.  U: [M, m].
    The ridge is not optional here: the pinned anchor node's coordinates are constant
    across every row, so they are exactly collinear with the constant term in psi and
    the normal matrix is singular by construction.
    """
    O = np.hstack([Z, U])                                  # [M, N+m]
    G = O.T @ O + ridge * np.eye(O.shape[1])
    K = np.linalg.solve(G, O.T @ Zp).T                     # [N, N+m]
    return K[:, :Z.shape[1]], K[:, Z.shape[1]:]


def build_transitions(dataset):
    """(x_k, u_k, x_k+1) triples, using the dataset's own pairing and boundaries.

    horizon=1 windows already guarantee that no pair straddles a trajectory or case
    edge, and __getitem__ pairs u_k = cmd[k+1] with the transition k -> k+1.
    """
    s = dataset.window_starts.numpy()
    X = dataset.states.numpy()[s]
    Xp = dataset.states.numpy()[s + 1]
    U = dataset.cmds.numpy()[s + 1]
    return X.astype(np.float64), U.astype(np.float64), Xp.astype(np.float64)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def rollout_rmse_mm(A, B, lift, dataset, X, U, Xp, horizon, starts_ok):
    """Multi-step position RMSE in mm, so it is comparable to the neural model's
    rmse_1_mm / rmse_H_mm. Rolls psi forward under the recorded commands and reads
    the state back off the leading block."""
    sl = dataset.pos_slice
    std = dataset.state_std[0, sl]
    n = dataset.state_dim
    errs = []
    for h_target in (1, horizon):
        idx = starts_ok[h_target]
        if len(idx) == 0:
            errs.append(float("nan"))
            continue
        z = lift(X[idx])
        for h in range(h_target):
            z = z @ A.T + U[idx + h] @ B.T
        pred = z[:, :n][:, sl]
        true = Xp[idx + h_target - 1][:, sl]
        errs.append(float(np.sqrt((((pred - true) * std) ** 2).mean())))
    return errs


def contiguous_starts(dataset, horizon):
    """Window starts that have `horizon` further transitions inside the same run."""
    s = dataset.window_starts.numpy()
    valid = {}
    lookup = set(s.tolist())
    for h in (1, horizon):
        keep = [i for i, k in enumerate(s)
                if all((k + j) in lookup for j in range(h))]
        valid[h] = np.asarray(keep, dtype=np.int64)
    return valid


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="EDMD baseline on the [x, v] rope state.")
    p.add_argument("--csv", type=str, nargs="+", required=True)
    p.add_argument("--features", type=str, nargs="+", default=["pos", "vel"])
    p.add_argument("--lift", default="affine",
                   choices=["state", "affine", "quad", "rbf"])
    p.add_argument("--n-rbf", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--horizon", type=int, default=10,
                   help="Horizon for the multi-step report (matches the neural model's H).")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="edmd_model.npz")
    args = p.parse_args()

    ds = RopeKoopmanDataset(args.csv, features=tuple(args.features), horizon=1,
                            normalize=not args.no_normalize, clamp_tol_mm=None)
    X, U, Xp = build_transitions(ds)
    print(f"state_dim {ds.state_dim} | {len(X)} transitions | normalize {ds.normalize}")

    # Split by trajectory, not by row: consecutive rows are nearly identical, so a
    # random row split leaks almost perfectly and the val number becomes meaningless.
    traj = ds.traj_ids.numpy()
    uniq = np.unique(traj)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_traj = set(uniq[:n_val].tolist())
    is_val = np.array([t in val_traj for t in traj])
    print(f"train {int((~is_val).sum())} | val {int(is_val.sum())} transitions "
          f"from {len(uniq)} trajectories")

    lift = Lift.fit(args.lift, X[~is_val], n_rbf=args.n_rbf, seed=args.seed)
    print(f"lift '{args.lift}' -> N = {lift.dim}"
          + (f" (sigma {lift.sigma:.3f})" if lift.kind == "rbf" else ""))
    if lift.dim > 0.2 * int((~is_val).sum()):
        print("WARNING: dictionary is large relative to the number of transitions; "
              "raise --ridge or shrink the lift.")

    Z, Zp = lift(X[~is_val]), lift(Xp[~is_val])
    A, B = fit_edmdc(Z, U[~is_val], Zp, ridge=args.ridge)

    rho = float(np.abs(np.linalg.eigvals(A)).max())
    print(f"\nrho(A) = {rho:.6f}"
          + ("   <- contractive: a constant command gives a bounded lifted state, so "
             "the model cannot represent sustained travel" if rho < 0.95 else ""))

    # One-step residual in the lifted space, for reference against the neural 'lin' term
    resid = Z @ A.T + U[~is_val] @ B.T - Zp
    print(f"one-step lifted residual (train): {np.sqrt((resid ** 2).mean()):.6e}")

    starts = contiguous_starts(ds, args.horizon)
    for name, mask in (("train", ~is_val), ("val", is_val)):
        idx_all = np.flatnonzero(mask)
        sub = {h: np.intersect1d(starts[h], idx_all) for h in starts}
        e1, eH = rollout_rmse_mm(A, B, lift, ds, X, U, Xp, args.horizon, sub)
        print(f"{name:5s} position RMSE: {e1:8.3f} mm @h=1 | {eH:8.3f} mm @h={args.horizon}")

    # --- the integrator check ------------------------------------------------
    # Node 0 is kinematically driven: p0_k+1 = p0_k + u_k exactly. p0 sits in the
    # dictionary, so row 0 of [A B] can represent that as a literal unit row. If it
    # does, the near-unit mode you fought for in training is here by construction.
    # If it does not, suspect the (x_k, u_k, x_k+1) pairing before anything else.
    print(f"\nnode-0 x row:  A[0,0] = {A[0, 0]:.4f}   "
          f"max |A[0, 1:]| = {np.abs(A[0, 1:]).max():.4f}   "
          f"B[0] = {np.round(B[0], 4)}")

    np.savez(args.out, A=A, B=B, lift=args.lift, ridge=args.ridge,
             state_dim=ds.state_dim, lift_dim=lift.dim, rho=rho,
             centers=lift.centers if lift.centers is not None else np.zeros(0),
             sigma=lift.sigma if lift.sigma is not None else 0.0,
             state_mean=ds.state_mean, state_std=ds.state_std, cmd_scale=ds.cmd_scale,
             features=np.array(list(ds.features)))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()