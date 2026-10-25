"""
edmd_lqr.py

Constrained finite-horizon LQR for rope shaping, driven by the EDMD model fitted in
edmd_baseline.py instead of a learned encoder/decoder.

State:
    s = [x, v]      x : 60 positions, v : 60 velocities, s : 120-d

Lifted state:
    z = psi(s)      psi's leading block IS s, so the "decoder" is the exact selection
                    z[:120] -> s. There is no reconstruction floor.

Model:
    z_{k+1} = A z_k + B u_k

Control:
    u = [dx, dy]  [mm / step], box-constrained

What changes vs. the neural version (const_lqr.py)
-------------------------------------------------
  * No checkpoints, no torch. A, B and the lift come out of edmd_model.npz.
  * Q is a plain diagonal that selects the position block. No decoder Jacobian, no
    rank-deficient matrix, no ARPACK failure.
  * z_goal = psi(goal_state), built directly. No encoder round-trip.

Solver notes
------------
The goal is fixed for a whole run, so z_goal is baked in as a CONSTANT rather than a
cp.Parameter. Only z0 remains parameterised, which keeps the problem DPP-compliant and
lets OSQP warm-start; with z_goal as a parameter the problem is not DPP, every solve
starts cold, and at 321 latent dims x H steps that is what exhausted the iteration cap.

Q is also normalised to mean weight 1 rather than left in raw mm^2. OSQP is first-order
and a cost with a ~1e6 dynamic range between Qf entries and R conditions it badly.
Absolute scale is carried by --qf-weight / --r-weight instead, where it is visible.

Usage
-----
    python edmd_lqr.py --model edmd_model.npz --shape rope_all_shapes/rope_C.json --H 10
"""

import argparse
import json
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from edmd_baseline import Lift          # reuse the fitted lift, never reimplement it
from rope_manip_env import RopeManipEnv, MM


# ============================================================
# Model
# ============================================================
class EDMDController:
    """Wraps the fitted EDMD model: normalise -> lift -> (A, B) -> unlift -> denormalise."""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)

        self.A = d["A"].astype(np.float64)
        self.B = d["B"].astype(np.float64)

        self.state_mean = d["state_mean"][0].astype(np.float64)
        self.state_std = d["state_std"][0].astype(np.float64)
        self.cmd_scale = float(d["cmd_scale"])

        self.state_dim = int(d["state_dim"])
        self.z_dim = self.A.shape[0]
        self.features = [str(f) for f in d["features"]]

        kind = str(d["lift"])
        centers = d["centers"]
        self.lift = Lift(
            kind,
            self.state_dim,
            n_rbf=len(centers) if centers.size else 0,
            centers=centers if centers.size else None,
            sigma=float(d["sigma"]) if float(d["sigma"]) > 0 else None,
        )
        if self.lift.dim != self.z_dim:
            raise RuntimeError(
                f"Lift dim {self.lift.dim} != A dim {self.z_dim}; the npz and the "
                f"Lift class have drifted apart.")

        # B was fitted against u / cmd_scale, so undo that here and keep u in raw mm
        # everywhere downstream (the box constraint and env.drive_step both expect mm).
        self.B = self.B / self.cmd_scale

        self.rho = float(np.abs(np.linalg.eigvals(self.A)).max())

    def encode(self, state):
        """Physical [x, v] in mm, mm/s -> lifted z."""
        s = (np.asarray(state, np.float64) - self.state_mean) / self.state_std
        return self.lift(s[None, :]).ravel()

    def decode(self, z):
        """Lifted z -> physical [x, v]. Exact: the state is psi's leading block."""
        return np.asarray(z, np.float64)[: self.state_dim] * self.state_std + self.state_mean

    def clip_spectrum(self):
        """Push any eigenvalue outside the unit circle back onto it.

        Only worth doing if rho > 1: over an H-step horizon the equality constraints
        compound A exactly H times, and a divergent mode makes the solver chase
        terminal error that the real rope will never produce.
        """
        w, V = np.linalg.eig(self.A)
        if np.abs(w).max() <= 1.0:
            return False
        w = np.where(np.abs(w) > 1.0, w / np.abs(w), w)
        self.A = np.real(V @ np.diag(w) @ np.linalg.inv(V))
        self.rho = float(np.abs(np.linalg.eigvals(self.A)).max())
        return True


# ============================================================
# Finite-horizon LQR
# ============================================================
class FiniteHorizonLQR:
    """z_goal is a compile-time constant; only z0 is a Parameter, so the problem is DPP."""

    def __init__(self, A, B, Q, R, Qf, z_goal, horizon=20, u_max=1.5,
                 max_iter=20000, eps=1e-4):
        self.A, self.B = A, B
        self.Q, self.R, self.Qf = Q, R, Qf
        self.H = horizon
        self.u_max = u_max
        self.max_iter, self.eps = max_iter, eps

        self.z_dim = A.shape[0]
        self.u_dim = B.shape[1]

        self.z = cp.Variable((self.z_dim, self.H + 1))
        self.u = cp.Variable((self.u_dim, self.H))
        self.z0 = cp.Parameter(self.z_dim)

        zg = np.asarray(z_goal, np.float64)          # CONSTANT, not a Parameter

        constraints = [self.z[:, 0] == self.z0]
        for k in range(self.H):
            constraints.append(
                self.z[:, k + 1] == self.A @ self.z[:, k] + self.B @ self.u[:, k])
        constraints += [self.u <= self.u_max, self.u >= -self.u_max]

        # Q is diagonal PSD by construction, so quad_form is safe here -- unlike the
        # rank-60-of-256 decoder-Jacobian Q, which sent cvxpy's PSD check to ARPACK.
        cost = 0
        for k in range(self.H):
            cost += cp.quad_form(self.z[:, k] - zg, self.Q)
            cost += cp.quad_form(self.u[:, k], self.R)
        cost += cp.quad_form(self.z[:, self.H] - zg, self.Qf)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self.n_inaccurate = 0

    def solve(self, z_current):
        self.z0.value = z_current
        self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                           max_iter=self.max_iter, eps_abs=self.eps, eps_rel=self.eps,
                           polish=True)

        u = self.u[:, 0].value
        # A capped or loosely-converged solve still returns a usable first action.
        # Dying here would throw away the whole trajectory, so only genuinely
        # unusable outcomes raise.
        if u is None or self.problem.status in (cp.INFEASIBLE, cp.UNBOUNDED,
                                                cp.INFEASIBLE_INACCURATE,
                                                cp.UNBOUNDED_INACCURATE):
            raise RuntimeError(f"LQR solver failed: {self.problem.status}")
        if self.problem.status != cp.OPTIMAL:
            self.n_inaccurate += 1
        return np.asarray(u).ravel()


def build_Q(ctrl, pos_weight=1.0, ridge=1e-4):
    """Relative weight on squared position error; velocities and lift terms unweighted.

    The lift's leading block is the raw state, so selecting positions out of z is a
    constant mask -- no Jacobian, no linearisation, and the same matrix is valid at
    every point in the workspace (the Jacobian Q was only valid near z_goal).

    Weights are normalised to mean 1. Leaving them in raw mm^2 gives entries of ~1e3,
    Qf entries of ~1e5 and an R of 0.1, and OSQP does not survive that spread.

    The ridge is doing real work: only n_pos of z_dim diagonal entries are nonzero,
    so every other latent direction is constrained by the ridge alone. Too small and
    the QP is numerically rank-deficient.

    Velocities are left unweighted deliberately: the goal is a shape, not a
    shape-at-a-velocity, and penalising v fights the settling the rope must do anyway.
    """
    n_pos = ctrl.state_dim // 2 if "vel" in ctrl.features else ctrl.state_dim
    Q = np.zeros((ctrl.z_dim, ctrl.z_dim))
    idx = np.arange(n_pos)
    w = ctrl.state_std[:n_pos] ** 2
    w = w / w.mean()                                  # relative node weighting, mean 1
    Q[idx, idx] = pos_weight * w
    return Q + ridge * np.eye(ctrl.z_dim)


# ============================================================
# Run
# ============================================================
def run(args):
    ctrl = EDMDController(args.model)
    print(f"Model: {args.model}")
    print(f"  lift '{ctrl.lift.kind}' | state_dim {ctrl.state_dim} | z_dim {ctrl.z_dim}")
    print(f"  features {ctrl.features} | cmd_scale {ctrl.cmd_scale:.6f}")
    print(f"  rho(A) = {ctrl.rho:.6f}   (as fitted)")
    if args.clip_spectrum and ctrl.clip_spectrum():
        print(f"  clipped onto the unit circle -> rho(A) = {ctrl.rho:.6f}")

    n_nodes = ctrl.state_dim // 4 if "vel" in ctrl.features else ctrl.state_dim // 2

    env = RopeManipEnv(n_nodes=n_nodes, base_length=0.30, base_radius=0.003,
                       youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01, seed=0)
    env.reset()

    # ---- goal -------------------------------------------------------------
    goal_pos = np.asarray(json.load(open(args.shape))["nodes"], dtype=np.float64)
    assert goal_pos.shape == (n_nodes, 2), f"shape file is {goal_pos.shape}, expected {(n_nodes, 2)}"
    goal_state = np.concatenate([goal_pos.reshape(-1), np.zeros(2 * n_nodes)])
    z_goal = ctrl.encode(goal_state)

    # decode(encode(.)) is exact for EDMD; assert it rather than trusting it, because
    # a silent mismatch here would look exactly like a controller failure.
    floor = np.sqrt(np.mean((ctrl.decode(z_goal)[:2 * n_nodes] - goal_pos.reshape(-1)) ** 2))
    print(f"  reconstruction floor decode(encode(x_goal)): {floor:.6f} mm")

    # sanity: is the measured state anywhere near the training distribution?
    s0 = np.concatenate([env.get_positions(mm=True)[:, :2].reshape(-1),
                         env.get_velocities(mm=True)[:, :2].reshape(-1)])
    z_score = np.abs((s0 - ctrl.state_mean) / ctrl.state_std).max()
    print(f"  max |z-score| of the initial state: {z_score:.2f}"
          + ("   <- WAY out of distribution, check units before reading any result"
             if z_score > 5 else ""))

    # ---- weights ----------------------------------------------------------
    Q = build_Q(ctrl, pos_weight=1.0, ridge=args.q_ridge)
    R = args.r_weight * np.eye(2)
    Qf = args.qf_weight * Q

    n_var = ctrl.z_dim * (args.H + 1) + 2 * args.H
    print(f"\nBuilding QP: z_dim {ctrl.z_dim} x H {args.H}  ({n_var} variables)...",
          flush=True)
    controller = FiniteHorizonLQR(A=ctrl.A, B=ctrl.B, Q=Q, R=R, Qf=Qf, z_goal=z_goal,
                                  horizon=args.H, u_max=args.u_max,
                                  max_iter=args.max_iter, eps=args.eps)
    print(f"  DPP-compliant: {controller.problem.is_dcp(dpp=True)}  "
          f"(False here means no warm start and every solve starts cold)")

    # ---- loop -------------------------------------------------------------
    print("\n" + "=" * 66)
    print("Starting EDMD-LQR")
    print("=" * 66)

    state_history, command_history, rmse_history, max_vel_history = [], [], [], []
    settled_for = 0
    best_rmse, best_step = np.inf, -1
    interrupted = False

    # Ctrl+C breaks out of the loop and falls through to the save + plot below,
    # so a long run can be stopped early without losing what it has logged.
    try:
        for step in range(args.steps):
            pos = env.get_positions(mm=True)[:, :2].reshape(-1)
            vel = env.get_velocities(mm=True)[:, :2].reshape(-1)
            state = np.concatenate([pos, vel])
            state_history.append(state.copy())

            u = controller.solve(ctrl.encode(state))
            u = np.clip(u, -args.u_max, args.u_max)
            env.drive_step(u / MM)

            new_pos = env.get_positions(mm=True)[:, :2]
            rmse = float(np.sqrt(np.mean(np.sum((new_pos - goal_pos) ** 2, axis=1))))
            max_vel = float(np.max(np.linalg.norm(env.get_velocities(mm=True)[:, :2], axis=1)))

            rmse_history.append(rmse)
            command_history.append(u.copy())
            max_vel_history.append(max_vel)
            if rmse < best_rmse:
                best_rmse, best_step = rmse, step

            if step % 10 == 0:
                print(f"step {step:4d} | u = [{u[0]:7.3f}, {u[1]:7.3f}] | "
                      f"RMSE = {rmse:8.3f} mm | max|v| = {max_vel:8.3f} mm/s")

            if env.is_exploded():
                print("Simulation exploded.")
                break

            if step >= 120:
                print("Simulation did not settle within 160 steps; aborting.")
                break

            # Settling test over a WINDOW, not a single sample. The rope rings with a
            # ~7-8 ms period against control_dt = 10 ms, so instantaneous speed dips deep
            # at oscillation troughs while the envelope is still high; a one-shot
            # threshold can fire at a trough and report a random phase as "settled".
            settled_for = settled_for + 1 if max_vel < args.settle_vel else 0
            if settled_for >= 100:
                print(f"Settled: max|v| < {args.settle_vel} mm/s for {args.settle_steps} "
                      f"consecutive steps.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nInterrupted by user (Ctrl+C) after {len(rmse_history)} steps; "
              f"saving and plotting what was logged.")

    # An interrupt between logging the state and logging the command leaves one extra
    # state; trim so every logged array has the same length.
    state_history = state_history[:len(command_history)]
    if not rmse_history:
        print("No steps completed; nothing to save.")
        return

    # ---- results ----------------------------------------------------------
    final_pos = env.get_positions(mm=True)[:, :2]
    final_rmse = float(np.sqrt(np.mean(np.sum((final_pos - goal_pos) ** 2, axis=1))))
    w = min(args.settle_steps, len(rmse_history))
    settle_rmse = float(np.mean(rmse_history[-w:]))
    per_node = np.linalg.norm(final_pos - goal_pos, axis=1)

    print("\n" + "=" * 66)
    print(f"Shape               : {Path(args.shape).stem}")
    print(f"Steps run           : {len(rmse_history)}")
    print(f"Final RMSE          : {final_rmse:.3f} mm   ({100 * final_rmse / 300:.2f}% of L)")
    print(f"Settle RMSE (last {w:2d}) : {settle_rmse:.3f} mm   <- report this, it is "
          f"phase-averaged")
    print(f"Best RMSE           : {best_rmse:.3f} mm at step {best_step}")
    print(f"Worst node error    : {per_node.max():.3f} mm at node {int(per_node.argmax())}")
    print(f"Reconstruction floor: {floor:.6f} mm")
    if controller.n_inaccurate:
        print(f"NOTE: {controller.n_inaccurate}/{len(rmse_history)} solves returned "
              f"inaccurate; raise --max-iter or loosen --eps if this is most of them.")
    print("=" * 66)

    np.savez(
        args.log_npz,
        command=np.asarray(command_history), state=np.asarray(state_history),
        rmse=np.asarray(rmse_history), max_vel=np.asarray(max_vel_history),
        goal_state=goal_state, goal_pos=goal_pos, z_goal=z_goal,
        final_pos=final_pos, per_node_error=per_node,
        final_rmse=final_rmse, settle_rmse=settle_rmse, recon_floor=floor,
        A=ctrl.A, B=ctrl.B, Q=Q, R=R, Qf=Qf, rho=ctrl.rho, lift=ctrl.lift.kind,
        horizon=np.array(args.H), u_max=np.array(args.u_max),
        control_dt=np.array(0.01), shape=Path(args.shape).stem,
        n_inaccurate=np.array(controller.n_inaccurate),
        interrupted=np.array(interrupted),
    )
    print(f"Saved {args.log_npz}")

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], "r--", lw=2, label="goal")
    ax[0].plot(final_pos[:, 0], final_pos[:, 1], "b-", lw=2, label="final")
    ax[0].scatter(*final_pos[0], s=60, c="k", zorder=5, label="node 0 (driven)")
    ax[0].scatter(*final_pos[-1], s=60, c="g", zorder=5, label="anchor")
    ax[0].set_xlabel("X [mm]"); ax[0].set_ylabel("Y [mm]")
    ax[0].set_title(f"final shape - settle RMSE {settle_rmse:.2f} mm")
    ax[0].axis("equal"); ax[0].grid(True); ax[0].legend()

    ax[1].plot(rmse_history, lw=1.5)
    ax[1].axhline(settle_rmse, color="r", ls="--", lw=1, label=f"settle {settle_rmse:.2f} mm")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("RMSE [mm]")
    ax[1].set_title("tracking error"); ax[1].grid(True); ax[1].legend()

    ax[2].plot(per_node, "o-", ms=4)
    ax[2].set_xlabel("node index (0 = driven, 29 = anchor)")
    ax[2].set_ylabel("final error [mm]")
    ax[2].set_title("per-node error"); ax[2].grid(True)

    plt.tight_layout()
    plt.savefig(args.log_npz.replace(".npz", ".png"), dpi=130)
    print(f"Saved {args.log_npz.replace('.npz', '.png')}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Finite-horizon LQR on an EDMD rope model.")
    p.add_argument("--model", default="edmd_model.npz")
    p.add_argument("--shape", default="rope_all_shapes/rope_I.json")
    p.add_argument("--H", type=int, default=10, help="LQR prediction horizon")
    p.add_argument("--u-max", type=float, default=1.5, help="max command [mm/step]")
    p.add_argument("--r-weight", type=float, default=0.1, help="control penalty")
    p.add_argument("--qf-weight", type=float, default=200.0, help="terminal cost multiplier")
    p.add_argument("--q-ridge", type=float, default=1e-4,
                   help="Ridge on Q. Most latent directions are constrained by this alone.")
    p.add_argument("--max-iter", type=int, default=20000, help="OSQP iteration cap")
    p.add_argument("--eps", type=float, default=1e-4, help="OSQP eps_abs / eps_rel")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--settle-vel", type=float, default=5.0,
                   help="max|v| threshold [mm/s] for the settling test")
    p.add_argument("--settle-steps", type=int, default=10,
                   help="consecutive steps below the threshold before stopping "
                        "(>= one ring period, so a trough cannot trip it)")
    p.add_argument("--clip-spectrum", action="store_true",
                   help="Project eigenvalues onto the unit circle if rho(A) > 1.")
    p.add_argument("--log-npz", default="edmd_lqr_run.npz")
    run(p.parse_args())


if __name__ == "__main__":
    main()