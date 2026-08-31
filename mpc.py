"""
mpc_simple.py

Minimal condensed MPC in Koopman latent space, for the model trained by the *new*
neural_koopman.py (the one with an InputEncoder and no operator bias).

Why this file exists
--------------------
mpc.py evaluates its cost through the decoder and solves with Adam, so when tracking is
bad you cannot tell whether it is (a) the solver not converging, (b) the cost being wrong,
or (c) the model being wrong. This file separates those:

    decision variable = v_k, the LIFTED control (what B actually multiplies)
    dynamics          = z_{k+1} = A z_k + B v_k            (exactly linear)
    cost              = quadratic in z and v               (exactly quadratic)
    => J(V) is a convex quadratic in V, minimised by ONE linear solve.

So `solve_analytic` gives the true optimum. `solve_adam` runs the same cost through
gradient descent. If they disagree, it is a solver problem. If they agree and tracking is
still bad, it is a cost/model problem. Run `--mode diagnose` to print exactly that.

Horizon convention
------------------
  K = number of free input steps      (decision variables: V in R^{K x m})
  H = number of predicted state steps (K <= H)
  For k >= K the input is either zero (`--tail zero`, free evolution) or frozen at
  v_{K-1} (`--tail hold`). K == H recovers plain dense MPC.

The lifted-control caveat (READ THIS)
-------------------------------------
The plant does not accept v; it accepts a raw 2-d (cmd_dx, cmd_dy). The reachable set of
v is the image of InputEncoder, a 2-d manifold inside R^{encoded_control_dim} (16-d by
default). Optimising v freely over all of R^16 will generally land OFF that manifold, and
`B v` for such a v is a control authority the operator was never trained to produce. This
file therefore always reports the *realisability gap*: it projects v back to a raw command
u by minimising ||InputEncoder(u) - v||, then reports ||InputEncoder(u*) - v*||, and can
step the plant with the re-lifted command instead (`--apply relifted`, the honest mode).

A large gap means the convex-in-v formulation is a fiction and you should optimise over
raw u directly (`--param raw`, non-convex, Adam only) or retrain with a linear B u.

Usage
-----
  # is the solver converging? (analytic vs Adam on the identical cost)
  python mpc_simple.py --mode diagnose --K 5 --H 10

  # closed-loop track, analytic solve each tick, honest re-lifted commands
  python mpc_simple.py --mode track --K 5 --H 10 --apply relifted --steps 100

  # non-convex sanity check: same cost, but optimise raw cmd through the input encoder
  python mpc_simple.py --mode track --param raw --K 5 --H 10
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import torch

from ngk_simple import RopeKoopmanDataset, KoopmanModel

DT = torch.float64  # condensed matrices are solved in float64 on CPU; they are small


# --------------------------------------------------------------------------- #
# Checkpoint loading (mirrors mpc.py, plus the input encoder)
# --------------------------------------------------------------------------- #
def latest_checkpoint(results_dir):
    ckpt_root = Path(results_dir) / "checkpoints"
    dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_exp_args(results_dir, exp_name):
    log_path = Path(results_dir) / f"{exp_name}.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} not found; cannot recover the training args.")
    return ast.literal_eval(log_path.read_text().splitlines()[0].split("Args: ", 1)[1])


def load_koopman(results_dir="training_results", checkpoint=None, csv=None, device=None):
    """Rebuilds model + normalisation stats. Normalisation is re-fit from the same CSVs,
    exactly as in mpc.py, because the training script never writes it to disk."""
    results_dir = Path(results_dir)
    ckpt_dir = Path(checkpoint) if checkpoint else latest_checkpoint(results_dir)
    exp_name = ckpt_dir.name
    ta = load_exp_args(results_dir, exp_name)

    dataset = RopeKoopmanDataset(
        csv if csv else ta["csv"], features=tuple(ta["features"]), horizon=1,
        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None,
    )
    device = device or torch.device("cpu")  # everything here is small; CPU keeps it simple

    model = KoopmanModel(
        state_dim=dataset.state_dim, control_dim=dataset.control_dim,
        koopman_dim=ta["koopman_dim"],
        encoded_control_dim=ta.get("encoded_control_dim", 16),
        hidden_dim=ta["hidden_dim"], num_hidden=ta["num_hidden"],
        control_hidden_dim=ta.get("control_hidden_dim", 64),
        control_num_hidden=ta.get("control_num_hidden", 2),
        operator_bias=not ta.get("no_operator_bias", False),
    ).to(device)

    ld = lambda f: torch.load(ckpt_dir / f, map_location=device)
    model.encoder.load_state_dict(ld("koopman_enc.pt"))
    model.decoder.load_state_dict(ld("koopman_dec.pt"))
    model.operator.load_state_dict(ld("koopman_op.pt"))
    model.input_encoder.load_state_dict(ld("koopman_input_enc.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dataset, device, exp_name


# --------------------------------------------------------------------------- #
# Condensed prediction matrices
# --------------------------------------------------------------------------- #
def build_condensed(A, B, H, K, tail="zero"):
    """Stacks the rollout z_1..z_H as an affine function of the stacked lifted inputs.

        Z = S @ vec(V) + F @ z0,     Z in R^{H*n},  vec(V) in R^{K*m}

    z_k = A^k z0 + sum_{j<k} A^{k-1-j} B v_j, with v_j for j >= K set by `tail`.
    """
    n, m = A.shape[0], B.shape[1]
    S = torch.zeros(H * n, K * m, dtype=DT)
    F = torch.zeros(H * n, n, dtype=DT)
    Apow = [torch.eye(n, dtype=DT)]
    for _ in range(H):
        Apow.append(Apow[-1] @ A)
    for k in range(1, H + 1):
        F[(k - 1) * n:k * n] = Apow[k]
        for j in range(k):
            M = Apow[k - 1 - j] @ B
            if j < K:
                S[(k - 1) * n:k * n, j * m:(j + 1) * m] += M
            elif tail == "hold":
                S[(k - 1) * n:k * n, (K - 1) * m:K * m] += M
    return S, F


def rate_matrix(K, m):
    """D such that D @ vec(V) = vec([v_0 - v_{-1}, v_1 - v_0, ...]) with v_{-1} added
    separately as the offset d0."""
    D = torch.eye(K * m, dtype=DT)
    for k in range(1, K):
        D[k * m:(k + 1) * m, (k - 1) * m:k * m] -= torch.eye(m, dtype=DT)
    return D


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #
class SimpleKoopmanMPC:
    """Latent-space condensed MPC. All state/command I/O is in REAL (mm) units."""

    def __init__(self, model, dataset, H=10, K=5, tail="zero",
                 q=1.0, qf=10.0, r=1e-3, r_rate=1e-3):
        self.model = model
        self.H, self.K, self.tail = H, K, tail
        self.q, self.qf, self.r, self.r_rate = q, qf, r, r_rate

        self.state_mean = torch.as_tensor(dataset.state_mean[0])
        self.state_std = torch.as_tensor(dataset.state_std[0])
        self.cmd_scale = float(dataset.cmd_scale)
        self.pos_slice = dataset.pos_slice

        A = model.operator.A.detach().to(DT)
        B = model.operator.B.detach().to(DT)
        self.n, self.m = A.shape[0], B.shape[1]
        self.S, self.F = build_condensed(A, B, H, K, tail)
        self.D = rate_matrix(K, self.m)

        # per-step weight, ramped 1 -> qf over the horizon, expanded to R^{H*n}
        w = self.q * (1.0 + (qf - 1.0) * torch.linspace(0.0, 1.0, H, dtype=DT))
        self.W = w.repeat_interleave(self.n)                       # [H*n]

        # Constant Hessian: 2 (S' W S + r I + r_rate D' D). Prefactorised once.
        Hess = self.S.T @ (self.W[:, None] * self.S) \
            + self.r * torch.eye(K * self.m, dtype=DT) \
            + self.r_rate * (self.D.T @ self.D)
        self.Hess = Hess
        self.chol = torch.linalg.cholesky(Hess)
        self.cond = torch.linalg.cond(Hess).item()
        self.rho = torch.linalg.eigvals(A).abs().max().item()

    # -- conversions --------------------------------------------------------- #
    def normalize_state(self, x):
        return (x - self.state_mean) / self.state_std

    def encode(self, x_real):
        return self.model.encoder(self.normalize_state(x_real).float()).to(DT)

    def decode(self, z):
        return self.model.decoder(z.float()) * self.state_std + self.state_mean

    def lift(self, u_real):
        return self.model.input_encoder((u_real / self.cmd_scale).float()).to(DT)

    def position_error_mm(self, x, x_goal):
        d = (x - x_goal)[..., self.pos_slice]
        return d.pow(2).mean().sqrt().item()

    # -- the cost, as an explicit function of the stacked lifted inputs -------- #
    def cost(self, V, z0, z_goal, v_prev):
        """J(V). Quadratic and convex in V. Used by BOTH solvers, so they are
        provably minimising the same thing."""
        Z = self.S @ V + self.F @ z0
        e = Z - z_goal.repeat(self.H)
        dv = self.D @ V - torch.cat([v_prev, torch.zeros((self.K - 1) * self.m, dtype=DT)])
        return (self.W * e * e).sum() + self.r * (V * V).sum() + self.r_rate * (dv * dv).sum()

    # -- solvers ------------------------------------------------------------- #
    def solve_analytic(self, z0, z_goal, v_prev):
        """Exact minimiser. Hess V = S' W (Zg - F z0) + r_rate D' d0."""
        rhs = self.S.T @ (self.W * (z_goal.repeat(self.H) - self.F @ z0)) \
            + self.r_rate * (self.D.T @ torch.cat(
                [v_prev, torch.zeros((self.K - 1) * self.m, dtype=DT)]))
        return torch.cholesky_solve(rhs.unsqueeze(1), self.chol).squeeze(1)

    def solve_adam(self, z0, z_goal, v_prev, iters=300, lr=0.05, V_init=None, trace=False):
        """Same cost, gradient descent. Only here so you can compare against the exact
        answer; in production use solve_analytic."""
        V = (torch.zeros(self.K * self.m, dtype=DT) if V_init is None
             else V_init.clone()).requires_grad_(True)
        opt = torch.optim.Adam([V], lr=lr)
        hist = []
        for _ in range(iters):
            opt.zero_grad()
            J = self.cost(V, z0, z_goal, v_prev)
            J.backward()
            opt.step()
            if trace:
                hist.append(J.item())
        return V.detach(), hist

    # -- projecting a lifted control back to a raw command -------------------- #
    def unlift(self, v, iters=200, lr=0.05, u_max_mm=None):
        """min_u ||InputEncoder(u/scale) - v||^2. Returns (u_real, residual_norm).

        The residual is the realisability gap: how far the optimiser's chosen v is from
        anything the plant can actually be commanded to do."""
        u = torch.zeros(2, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([u], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            res = (self.model.input_encoder(u).to(DT) - v).pow(2).sum()
            res.backward()
            opt.step()
        with torch.no_grad():
            u_real = u.detach().to(DT) * self.cmd_scale
            if u_max_mm is not None:
                u_real = u_real.clamp(-u_max_mm, u_max_mm)
            resid = (self.lift(u_real.float()) - v).norm().item()
        return u_real, resid

    # -- raw-parameterisation solve (non-convex, for comparison) -------------- #
    def solve_raw(self, z0, z_goal, u_prev_real, iters=300, lr=0.05, u_max_mm=None):
        """Optimise the raw 2-d commands directly, lifting them through InputEncoder.
        Always realisable, but no longer convex."""
        scale = 1.0 if u_max_mm is None else min(1.0, u_max_mm / self.cmd_scale)
        theta = torch.zeros(self.K, 2, requires_grad=True)
        v_prev = self.lift(u_prev_real.float()) if u_prev_real is not None \
            else torch.zeros(self.m, dtype=DT)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            u = scale * torch.tanh(theta)
            V = self.model.input_encoder(u).to(DT).reshape(-1)
            self.cost(V, z0, z_goal, v_prev).backward()
            opt.step()
        with torch.no_grad():
            u = (scale * torch.tanh(theta)).to(DT) * self.cmd_scale
            V = self.model.input_encoder((u / self.cmd_scale).float()).to(DT).reshape(-1)
        return u, V

    # -- closed loop ---------------------------------------------------------- #
    def track(self, x0, x_goal, steps=100, param="lifted", apply="relifted",
              iters=300, lr=0.05, u_max_mm=None, tol_mm=1.0, plant_step=None, verbose=True):
        """Receding horizon: re-solve each tick, apply the first input only.

        apply='relifted'  -> project v_0 to a raw command and step with THAT (honest)
        apply='lifted'    -> step with v_0 directly (upper bound; plant cannot do this)
        """
        x_t = x0.clone().to(DT)
        z_goal = self.encode(x_goal.to(DT))
        traj = {"states": [x_t.clone()], "cmds": [], "err_mm": [self.position_error_mm(x_t, x_goal)],
                "gap": [], "cost": []}
        v_prev = torch.zeros(self.m, dtype=DT)
        u_prev = None

        for t in range(steps):
            z_t = self.encode(x_t)
            if param == "raw":
                u_seq, V = self.solve_raw(z_t, z_goal, u_prev, iters, lr, u_max_mm)
                v0, u0, gap = V[:self.m], u_seq[0], 0.0
            else:
                V = self.solve_analytic(z_t, z_goal, v_prev)
                v0 = V[:self.m]
                u0, gap = self.unlift(v0, u_max_mm=u_max_mm)
            J = self.cost(V, z_t, z_goal, v_prev).item()

            v_applied = self.lift(u0.float()) if apply == "relifted" else v0
            if plant_step is not None:
                x_t = plant_step(x_t, u0).to(DT)
            else:
                with torch.no_grad():
                    z_next = self.model.operator(z_t.float().unsqueeze(0),
                                                 v_applied.float().unsqueeze(0)).squeeze(0)
                    x_t = self.decode(z_next.to(DT)).to(DT)

            err = self.position_error_mm(x_t, x_goal)
            traj["states"].append(x_t.clone()); traj["cmds"].append(u0.clone())
            traj["err_mm"].append(err); traj["gap"].append(gap); traj["cost"].append(J)
            v_prev, u_prev = v_applied, u0
            if verbose:
                print(f"  t={t:03d}  u=({u0[0]:+7.3f},{u0[1]:+7.3f})mm  err {err:8.3f}mm  "
                      f"J {J:11.4f}  lift-gap {gap:.3e}")
            if tol_mm is not None and err < tol_mm:
                print(f"  reached {tol_mm} mm at t={t}")
                break

        traj["states"] = torch.stack(traj["states"])
        traj["cmds"] = torch.stack(traj["cmds"]) if traj["cmds"] else torch.zeros(0, 2, dtype=DT)
        return traj


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def diagnose(mpc, z0, z_goal, iters, lr):
    v_prev = torch.zeros(mpc.m, dtype=DT)
    V_star = mpc.solve_analytic(z0, z_goal, v_prev)
    J_star = mpc.cost(V_star, z0, z_goal, v_prev).item()
    J_zero = mpc.cost(torch.zeros_like(V_star), z0, z_goal, v_prev).item()
    V_adam, hist = mpc.solve_adam(z0, z_goal, v_prev, iters=iters, lr=lr, trace=True)
    J_adam = mpc.cost(V_adam, z0, z_goal, v_prev).item()

    print("\n--- problem conditioning -------------------------------------------")
    print(f"  rho(A)                      {mpc.rho:.4f}"
          + ("   <-- >1: rollouts of length H amplify by "
             f"{mpc.rho ** mpc.H:.2e}" if mpc.rho > 1 else ""))
    print(f"  cond(Hessian)               {mpc.cond:.3e}"
          + ("   <-- ill-conditioned; first-order solvers will crawl"
             if mpc.cond > 1e6 else ""))
    print(f"  decision variables          K*m = {mpc.K}*{mpc.m} = {mpc.K * mpc.m}")

    print("\n--- solver agreement (identical cost) ------------------------------")
    print(f"  J(0)                        {J_zero:.6e}")
    print(f"  J(V*) analytic              {J_star:.6e}")
    print(f"  J(V) Adam @{iters:4d} iters    {J_adam:.6e}")
    print(f"  suboptimality gap           {J_adam - J_star:.3e} "
          f"({(J_adam - J_star) / max(abs(J_star), 1e-12):.2%})")
    print(f"  ||V_adam - V*|| / ||V*||    {(V_adam - V_star).norm() / (V_star.norm() + 1e-12):.3e}")
    print(f"  ||grad J(V*)||              {(2 * (mpc.Hess @ V_star) - 2 * (mpc.S.T @ (mpc.W * (z_goal.repeat(mpc.H) - mpc.F @ z0)))).norm():.3e}")
    if len(hist) > 4:
        q = [hist[0], hist[len(hist) // 4], hist[len(hist) // 2], hist[-1]]
        print(f"  Adam cost trace             {q[0]:.4e} -> {q[1]:.4e} -> "
              f"{q[2]:.4e} -> {q[3]:.4e}")

    print("\n--- realisability of the optimal lifted input ----------------------")
    v0 = V_star[:mpc.m]
    u0, resid = mpc.unlift(v0)
    print(f"  ||v0*||                     {v0.norm():.4f}")
    print(f"  best raw cmd u0             ({u0[0]:+.4f}, {u0[1]:+.4f}) mm")
    print(f"  ||lift(u0) - v0*||          {resid:.4f}  "
          f"({resid / (v0.norm() + 1e-12):.1%} of ||v0*||)")
    if resid > 0.25 * v0.norm():
        print("  WARNING: the optimum is far off the reachable 2-d input manifold. The "
              "convex-in-v\n           relaxation is not a faithful surrogate here -- "
              "use --param raw, or retrain\n           with a linear B u so the lifted "
              "and raw problems coincide.")
    return V_star


# --------------------------------------------------------------------------- #
def _boundary(dataset, traj_id):
    rows = dataset.window_starts[dataset.traj_ids == traj_id]
    if rows.numel() == 0:
        raise ValueError(f"trajectory_id {traj_id} not in the loaded CSVs.")
    lo, hi = int(rows.min()), int(rows.max()) + dataset.horizon
    mean = torch.as_tensor(dataset.state_mean[0]); std = torch.as_tensor(dataset.state_std[0])
    return dataset.states[lo] * std + mean, dataset.states[hi] * std + mean, lo, hi


def main():
    p = argparse.ArgumentParser(description="Simple condensed Koopman MPC (debuggable).")
    p.add_argument("--results-dir", default="training_results")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--csv", nargs="+", default=None)
    p.add_argument("--traj-id", type=int, default=None)
    p.add_argument("--mode", choices=["diagnose", "track"], default="diagnose")
    p.add_argument("--param", choices=["lifted", "raw"], default="lifted",
                   help="lifted: convex QP over InputEncoder outputs. raw: non-convex over cmd.")
    p.add_argument("--apply", choices=["relifted", "lifted"], default="relifted")
    p.add_argument("--H", type=int, default=10, help="Predicted state steps.")
    p.add_argument("--K", type=int, default=5, help="Free input steps (K <= H).")
    p.add_argument("--tail", choices=["zero", "hold"], default="zero")
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--qf", type=float, default=10.0)
    p.add_argument("--r", type=float, default=1e-3)
    p.add_argument("--r-rate", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--u-max-mm", type=float, default=None)
    p.add_argument("--tol-mm", type=float, default=1.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    assert args.K <= args.H, "K must be <= H"

    model, dataset, device, exp_name = load_koopman(args.results_dir, args.checkpoint, args.csv)
    mpc = SimpleKoopmanMPC(model, dataset, H=args.H, K=args.K, tail=args.tail,
                           q=args.q, qf=args.qf, r=args.r, r_rate=args.r_rate)
    print(f"checkpoint '{exp_name}' | state_dim {dataset.state_dim} | koopman_dim {mpc.n} "
          f"| lifted control dim {mpc.m} | H={args.H} K={args.K} tail={args.tail}")

    traj_id = args.traj_id if args.traj_id is not None else int(dataset.traj_ids[0])
    x0, x_goal, lo, hi = _boundary(dataset, traj_id)
    x0, x_goal = x0.to(DT), x_goal.to(DT)
    print(f"trajectory {traj_id}: row {lo} -> {hi} | initial pos err "
          f"{mpc.position_error_mm(x0, x_goal):.3f} mm")

    if args.mode == "diagnose":
        diagnose(mpc, mpc.encode(x0), mpc.encode(x_goal), args.iters, args.lr)
        return

    traj = mpc.track(x0, x_goal, steps=args.steps, param=args.param, apply=args.apply,
                     iters=args.iters, lr=args.lr, u_max_mm=args.u_max_mm, tol_mm=args.tol_mm)
    print(f"\nfinal pos err {traj['err_mm'][-1]:.3f} mm after {len(traj['cmds'])} ticks "
          f"(started {traj['err_mm'][0]:.3f} mm)")
    if traj["gap"]:
        print(f"mean realisability gap {np.mean(traj['gap']):.4f}")
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, states=traj["states"].numpy(), cmds=traj["cmds"].numpy(),
                 err_mm=np.array(traj["err_mm"]), cost=np.array(traj["cost"]),
                 gap=np.array(traj["gap"]), x_goal=x_goal.numpy())
        print(f"saved {out}")


if __name__ == "__main__":
    main()