"""
letter_lqr.py

Simple constrained finite-horizon LQR for rope shaping.

State:
    s = [x, v]
    x : 60 position values
    v : 60 velocity values
    s : 120 dimensional

Control:
    u = [dx, dy]   [mm / step]

Koopman model:
    z_{k+1} = A z_k + B u_k

Controller:
    finite-horizon quadratic optimal control
    with simple box constraints on u.

The encoder/decoder are used outside the optimization.

Cost
----
Q is decoder-Jacobian weighted rather than the identity. J = d(decode)/dz evaluated at
z_goal maps latent perturbations to millimetre position perturbations, so e^T J^T J e is
a second-order-accurate proxy for squared mm position error -- which is the quantity
actually being reported. Q = I instead measures distance in arbitrary latent
coordinates, weighting directions the decoder barely reads as heavily as ones that move
the rope a centimetre.

The approximation is local to z_goal, which is exactly where settle error lives.

Run structure
-------------
Two phases, logged into one continuous history but separable via `n_control`:

  1. CONTROL. MPC runs until one of:
       - the RMSE plateaus (range over a window < --plateau-tol),
       - the RMSE diverges (windowed mean rises --diverge-tol above its best),
       - --steps is exhausted, or the sim explodes.
  2. HOLD. u = 0, so node 0 is frozen where the controller left it and the rope
     rings down under its own damping. Nothing here is control -- it separates
     "did the controller get there" from "was the rope still moving when measured".

Every stop test operates on a WINDOW, never on consecutive samples. The rope rings at
~7-8 ms against a 10 ms control step, so rmse[k] vs rmse[k-1] oscillates even after
convergence: a naive plateau test would never fire, and a naive "RMSE went up" test
would fire on the first upswing of every cycle.

Report `settle_rmse` (phase-averaged over the settling window), not the single-sample
final value, for the same reason.
"""

import argparse
import ast
import json
from collections import deque
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM
import matplotlib.pyplot as plt

# ============================================================
# Koopman model
# ============================================================


class KoopmanController:

    def __init__(self, dataset, model, A, B, mean, std):
        self.dataset = dataset
        self.model = model

        self.A = A
        self.B = B

        self.mean = mean
        self.std = std

        self.z_dim = A.shape[0]

    def encode(self, state):
        """
        Convert physical state [x, v] into Koopman state z.
        """

        state = np.asarray(state, dtype=np.float32)

        state_norm = (torch.as_tensor(state).unsqueeze(0) - self.mean) / self.std

        with torch.no_grad():
            z = self.model.encoder(state_norm)

        return z.numpy().ravel().astype(np.float64)

    def decode(self, z):
        """
        Decode Koopman state back to physical state.
        """

        z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            state_norm = self.model.decoder(z_t)

        state = state_norm * self.std + self.mean

        return state.numpy().ravel()


# ============================================================
# Finite-horizon LQR
# ============================================================


class FiniteHorizonLQR:

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        Qf,
        horizon=20,
        u_max=1.5,
    ):
        self.A = A
        self.B = B

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.H = horizon
        self.u_max = u_max

        self.z_dim = A.shape[0]
        self.u_dim = B.shape[1]

        # ----------------------------------------------------
        # Optimization variables
        # ----------------------------------------------------

        self.z = cp.Variable((self.z_dim, self.H + 1))

        self.u = cp.Variable((self.u_dim, self.H))

        # These are updated every control step
        self.z0 = cp.Parameter(self.z_dim)
        self.z_goal = cp.Parameter(self.z_dim)

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        constraints = []

        # Initial condition
        constraints.append(self.z[:, 0] == self.z0)

        # Koopman dynamics
        for k in range(self.H):

            constraints.append(
                self.z[:, k + 1] == self.A @ self.z[:, k] + self.B @ self.u[:, k]
            )

        # Input limits
        constraints.append(self.u <= self.u_max)

        constraints.append(self.u >= -self.u_max)

        # ----------------------------------------------------
        # Cost
        #
        # Q = J^T J + ridge is positive definite by construction, but it is a dense
        # 256x256 with a large condition number, and cvxpy's numerical PSD check
        # hands that to ARPACK shift-invert, which fails to converge. psd_wrap
        # asserts what is already true and skips the check.
        # ----------------------------------------------------

        Qw = cp.psd_wrap(self.Q)
        Qfw = cp.psd_wrap(self.Qf)

        cost = 0

        for k in range(self.H):

            error = self.z[:, k] - self.z_goal

            cost += cp.quad_form(error, Qw)

            cost += cp.quad_form(self.u[:, k], self.R)

        # Terminal cost
        terminal_error = self.z[:, self.H] - self.z_goal

        cost += cp.quad_form(terminal_error, Qfw)

        # ----------------------------------------------------
        # Problem
        # ----------------------------------------------------

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, z_current, z_goal):

        self.z0.value = z_current
        self.z_goal.value = z_goal

        self.problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
        )

        if self.problem.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ):
            raise RuntimeError(f"LQR solver failed: {self.problem.status}")

        # Only apply the first control action
        u = self.u[:, 0].value

        return np.asarray(u).ravel()


# ============================================================
# Load training experiment
# ============================================================


def load_experiment(args):

    device = torch.device("cpu")

    results_dir = Path(args.results_dir)

    log_path = results_dir / f"{args.exp_name}.txt"

    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find training log:\n{log_path}")

    # Same mechanism as the existing MPC code
    ta = ast.literal_eval(log_path.read_text().splitlines()[0].split("Args: ", 1)[1])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = RopeKoopmanDataset(
        ta["csv"],
        features=tuple(ta["features"]),
        horizon=1,
        normalize=not ta.get("no_normalize", False),
        clamp_tol_mm=None,
    )

    print("Dataset state dimension:", dataset.state_dim)

    print("Dataset features:", ta["features"])

    # We expect position + velocity
    if "vel" not in dataset.feature_slices:
        raise RuntimeError("This LQR script expects a position + velocity model.")

    mean = torch.as_tensor(dataset.state_mean[0])

    std = torch.as_tensor(dataset.state_std[0])

    # --------------------------------------------------------
    # Koopman model
    # --------------------------------------------------------

    model = KoopmanModel(
        state_dim=dataset.state_dim,
        control_dim=dataset.control_dim,
        koopman_dim=ta["koopman_dim"],
        hidden_dim=ta["hidden_dim"],
        num_hidden=ta["num_hidden"],
        operator_bias=not ta.get("no_operator_bias", False),
    ).to(device)

    ckpt_dir = results_dir / "checkpoints" / args.exp_name

    def load_checkpoint(filename):

        return torch.load(
            ckpt_dir / filename,
            map_location=device,
        )

    model.encoder.load_state_dict(load_checkpoint("koopman_enc.pt"))

    model.decoder.load_state_dict(load_checkpoint("koopman_dec.pt"))

    model.operator.load_state_dict(load_checkpoint("koopman_op.pt"))

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # Koopman matrices
    # --------------------------------------------------------

    A = model.operator.A.detach().cpu().numpy().astype(np.float64)

    B = model.operator.B.detach().cpu().numpy().astype(np.float64)

    # The operator was trained using u / cmd_scale
    B = B / dataset.cmd_scale

    print("A shape:", A.shape)
    print("B shape:", B.shape)

    print("rho(A):", np.max(np.abs(np.linalg.eigvals(A))))

    return (
        dataset,
        model,
        mean,
        std,
        A,
        B,
    )


# ============================================================
# Main control loop
# ============================================================


def run(args):

    # --------------------------------------------------------
    # Load Koopman model
    # --------------------------------------------------------

    (
        dataset,
        model,
        mean,
        std,
        A,
        B,
    ) = load_experiment(args)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    n_nodes = dataset.state_dim // 4

    env = RopeManipEnv(
        n_nodes=n_nodes,
        base_length=0.30,
        base_radius=0.003,
        youngs_modulus=1e6,
        damping=2.0,
        dt=5e-5,
        control_dt=0.01,
        seed=0,
    )

    env.reset()

    # --------------------------------------------------------
    # Koopman controller
    # --------------------------------------------------------

    koopman = KoopmanController(
        dataset=dataset,
        model=model,
        A=A,
        B=B,
        mean=mean,
        std=std,
    )

    # ========================================================
    # Goal
    # ========================================================

    with open(args.shape, "r") as f:
        goal_json = json.load(f)

    goal_pos = np.asarray(
        goal_json["nodes"],
        dtype=np.float32,
    )

    assert goal_pos.shape == (
        n_nodes,
        2,
    )

    # --------------------------------------------------------
    # Goal state
    #
    # s_goal = [x_goal, v_goal]
    #
    # Desired velocity = 0
    # --------------------------------------------------------

    goal_x = goal_pos.reshape(-1)

    goal_v = np.zeros_like(goal_x)

    goal_state = np.concatenate(
        [
            goal_x,
            goal_v,
        ]
    )

    print("Goal state dimension:", goal_state.shape)

    # Encode target
    z_goal = koopman.encode(goal_state)

    print("Goal latent dimension:", z_goal.shape)

    # --------------------------------------------------------
    # Reconstruction floor.
    #
    # decode(encode(x_goal)) bounds how well the goal can be represented at all.
    # It is NOT a bound on closed-loop tracking -- the decoder never runs inside
    # the MPC loop -- but it is the number to quote alongside every result.
    # --------------------------------------------------------

    recon = koopman.decode(z_goal)[: 2 * n_nodes]

    recon_floor = float(np.sqrt(np.mean(np.sum(
        (recon.reshape(n_nodes, 2) - goal_pos) ** 2, axis=1))))

    print(f"Reconstruction floor decode(encode(x_goal)): {recon_floor:.3f} mm")

    # ========================================================
    # LQR weights -- Jacobian-weighted cost
    # ========================================================

    z_dim = A.shape[0]

    # J maps latent perturbations to mm position perturbations, so e^T J^T J e is
    # a second-order-accurate proxy for squared mm error near the goal.
    zg_t = torch.tensor(z_goal, dtype=torch.float32)

    Jt = torch.autograd.functional.jacobian(
        lambda z: model.decoder(z.unsqueeze(0))[0], zg_t)          # (state_dim, z_dim)

    # Positions only. The decoder emits [x(60), v(60)]; keeping the velocity rows
    # would penalise v toward the zero goal velocity, which fights the settling the
    # rope has to do anyway.
    pos_sl = dataset.pos_slice

    Jd = (Jt.numpy()[pos_sl]
          * dataset.state_std[0][pos_sl][:, None]).astype(np.float64)   # -> mm

    Q = Jd.T @ Jd

    # rank(J^T J) <= 60 << z_dim, so without a floor the solver can drift freely
    # through the ~200 latent directions that cost nothing.
    Q += 1e-4 * (np.trace(Q) / z_dim) * np.eye(z_dim)

    Q /= (np.trace(Q) / z_dim)             # unit mean eigenvalue: r-weight portable
    Q = 0.5 * (Q + Q.T)                    # kill float asymmetry before psd_wrap

    print(f"Q cond {np.linalg.cond(Q):.2e}  rank(Jd) {np.linalg.matrix_rank(Jd)} "
          f"(expected {Jd.shape[0]})")

    R = args.r_weight * np.eye(2)

    Qf = args.qf_weight * Q

    # ========================================================
    # LQR
    # ========================================================

    controller = FiniteHorizonLQR(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Qf=Qf,
        horizon=args.H,
        u_max=args.u_max,
    )

    # ========================================================
    # Phase 1: control
    # ========================================================

    print()
    print("=" * 60)
    print("Phase 1: LQR control")
    print("=" * 60)

    # Full physical state logging:
    # state = [x(60), v(60)] -> 120 dimensions
    state_history = []
    rmse_history = []
    command_history = []
    max_vel_history = []

    plateau_win = deque(maxlen=args.plateau_steps)
    best_win = float("inf")            # best windowed-MEAN RMSE seen so far
    stop_reason = "step budget exhausted"

    for step in range(args.steps):

        # ----------------------------------------------------
        # Measure position
        # ----------------------------------------------------

        pos = env.get_positions(mm=True)[:, :2].reshape(-1)

        # ----------------------------------------------------
        # Measure velocity
        # ----------------------------------------------------

        vel = env.get_velocities(mm=True)[:, :2].reshape(-1)

        # ----------------------------------------------------
        # Full state
        #
        # s = [x, v]
        # ----------------------------------------------------

        state = np.concatenate(
            [
                pos,
                vel,
            ]
        )

        # ----------------------------------------------------
        # Encode current state
        # ----------------------------------------------------

        z_current = koopman.encode(state)

        # Save the complete measured 120-D physical state BEFORE applying u.
        state_history.append(state.copy())

        # ----------------------------------------------------
        # Solve finite-horizon LQR
        # ----------------------------------------------------

        u = controller.solve(
            z_current=z_current,
            z_goal=z_goal,
        )

        # ----------------------------------------------------
        # Safety clipping
        # ----------------------------------------------------

        u = np.clip(
            u,
            -args.u_max,
            args.u_max,
        )

        # ----------------------------------------------------
        # Apply first action
        # ----------------------------------------------------

        env.drive_step(u / MM)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        new_pos = env.get_positions(mm=True)[:, :2]

        rmse = float(np.sqrt(
            np.mean(
                np.sum(
                    (new_pos - goal_pos) ** 2,
                    axis=1,
                )
            )
        ))

        max_vel = float(np.max(np.linalg.norm(env.get_velocities(mm=True)[:, :2], axis=1)))

        rmse_history.append(rmse)
        command_history.append(u.copy())
        max_vel_history.append(max_vel)

        if step % 10 == 0:

            print(
                f"step {step:4d} | "
                f"u = [{u[0]:7.3f}, {u[1]:7.3f}] | "
                f"RMSE = {rmse:8.3f} mm | "
                f"Max Vel = {max_vel:8.3f} mm/s"
            )

        # ----------------------------------------------------
        # Stop if simulation exploded
        # ----------------------------------------------------

        if env.is_exploded():

            print("Simulation exploded.")
            stop_reason = "exploded"

            break

        # ----------------------------------------------------
        # Stop tests, both on a WINDOW rather than consecutive samples.
        #
        # The rope rings at ~7-8 ms against a 10 ms control step, so rmse[k]
        # oscillates even after convergence. A step-to-step plateau test would
        # never fire; a step-to-step "RMSE went up" test would fire on the first
        # upswing of every cycle.
        #
        # NOTE: a plateau means "no longer improving", NOT "converged". If MPC
        # stalls at 40 mm this fires there too, which is why rmse_at_stop is
        # reported alongside the settle value rather than instead of it.
        # ----------------------------------------------------

        plateau_win.append(rmse)

        if len(plateau_win) == args.plateau_steps:

            win_mean = sum(plateau_win) / len(plateau_win)

            # Plateau: RMSE has stopped changing.
            if max(plateau_win) - min(plateau_win) < args.plateau_tol:

                print(f"RMSE flat within {args.plateau_tol} mm over "
                      f"{args.plateau_steps} steps -- stopping control.")
                stop_reason = "plateau"

                break

            # Divergence: RMSE is now meaningfully worse than the best it reached.
            # Compared against best_win BEFORE updating it, so the first full window
            # can never trip this.
            # if win_mean > best_win + args.diverge_tol:

            #     print(f"RMSE rose {win_mean - best_win:.3f} mm above its best "
            #           f"({best_win:.3f} mm) -- stopping control.")
            #     stop_reason = "diverging"

            #     break

            best_win = min(best_win, win_mean)

    n_control = len(rmse_history)
    rmse_at_stop = float(rmse_history[-1]) if n_control else float("nan")

    # ========================================================
    # Phase 2: hold
    #
    # u = 0, so node 0 stays frozen where the controller left it and the rope
    # rings down under its own damping. Everything measured here is settling,
    # not control.
    # ========================================================

    print()
    print("=" * 60)
    print(f"Phase 2: hold (u = 0), up to {args.hold_steps} steps")
    print("=" * 60)

    settled_for = 0

    for h in range(args.hold_steps):

        pos = env.get_positions(mm=True)[:, :2].reshape(-1)
        vel = env.get_velocities(mm=True)[:, :2].reshape(-1)
        state_history.append(np.concatenate([pos, vel]))

        env.drive_step(np.zeros(2))

        new_pos = env.get_positions(mm=True)[:, :2]

        rmse = float(np.sqrt(np.mean(np.sum((new_pos - goal_pos) ** 2, axis=1))))
        max_vel = float(np.max(np.linalg.norm(env.get_velocities(mm=True)[:, :2], axis=1)))

        rmse_history.append(rmse)
        command_history.append(np.zeros(2))
        max_vel_history.append(max_vel)

        if h % 10 == 0:
            print(f"hold {h:4d} | RMSE = {rmse:8.3f} mm | Max Vel = {max_vel:8.3f} mm/s")

        if env.is_exploded():
            print("Simulation exploded during hold.")
            break

        # Settling test over a WINDOW, not a single sample: instantaneous speed
        # dips deep at oscillation troughs while the envelope is still high, so a
        # one-shot threshold can fire at a trough and call a ringing rope settled.
        settled_for = settled_for + 1 if max_vel < args.settle_vel else 0

        if settled_for >= args.settle_steps:
            print(f"Settled: max|v| < {args.settle_vel} mm/s for "
                  f"{args.settle_steps} consecutive steps.")
            break

    # ========================================================
    # Final result
    # ========================================================

    final_pos = env.get_positions(mm=True)[:, :2]

    final_rmse = float(np.sqrt(
        np.mean(
            np.sum(
                (final_pos - goal_pos) ** 2,
                axis=1,
            )
        )
    ))

    w = min(args.settle_steps, len(rmse_history))
    settle_rmse = float(np.mean(rmse_history[-w:]))
    per_node = np.linalg.norm(final_pos - goal_pos, axis=1)
    best_rmse = float(min(rmse_history))
    best_step = int(np.argmin(rmse_history))

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(f"Shape                     : {Path(args.shape).stem}")
    print(f"Control steps             : {n_control}  ({stop_reason})")
    print(f"Hold steps                : {len(rmse_history) - n_control}")
    print(f"RMSE when control stopped : {rmse_at_stop:.3f} mm")
    print(f"Settle RMSE (last {w:2d})      : {settle_rmse:.3f} mm   <- report this, "
          f"it is phase-averaged")
    print(f"Final RMSE                : {final_rmse:.3f} mm   "
          f"({100 * final_rmse / 300:.2f}% of L)")
    print(f"Best RMSE                 : {best_rmse:.3f} mm at step {best_step}")
    print(f"Worst node error          : {per_node.max():.3f} mm at node "
          f"{int(per_node.argmax())}")
    print(f"Reconstruction floor      : {recon_floor:.3f} mm")

    if stop_reason == "diverging" and best_rmse < settle_rmse - args.diverge_tol:
        print()
        print(f"NOTE: control diverged, and the rope is now sitting where divergence "
              f"caught it,\n      not at its best pose. PyElastica has no state rewind, "
              f"so to recover the\n      best configuration re-run deterministically "
              f"with --steps {best_step + 1}.")

    # ========================================================
    # Save run data
    # ========================================================

    state_history = np.asarray(state_history, dtype=np.float64)
    command_history = np.asarray(command_history, dtype=np.float64)
    rmse_history = np.asarray(rmse_history, dtype=np.float64)
    max_vel_history = np.asarray(max_vel_history, dtype=np.float64)

    np.savez(
        args.log_npz,
        # Per-step data (control phase then hold phase, concatenated)
        command=command_history,  # [N, 2] = [cmd_x, cmd_y] [mm/step]
        cmd_x=command_history[:, 0],
        cmd_y=command_history[:, 1],
        state=state_history,  # [N, 120] = [x(60), v(60)]
        rmse=rmse_history,  # [N]
        max_vel=max_vel_history,  # [N]
        # Phase split: rows [:n_control] are controlled, [n_control:] are hold
        n_control=np.array(n_control),
        stop_reason=stop_reason,
        # Headline numbers
        rmse_at_stop=np.array(rmse_at_stop),
        settle_rmse=np.array(settle_rmse),
        final_rmse=np.array(final_rmse),
        best_rmse=np.array(best_rmse),
        best_step=np.array(best_step),
        recon_floor=np.array(recon_floor),
        per_node_error=per_node,
        # Target / metadata
        goal_state=goal_state,  # [120]
        goal_pos=goal_pos,  # [n_nodes, 2]
        z_goal=z_goal,
        final_pos=final_pos,
        shape=Path(args.shape).stem,
        # LQR parameters
        A=A,
        B=B,
        Q=Q,
        R=R,
        Qf=Qf,
        Jd=Jd,
        horizon=np.array(args.H),
        u_max=np.array(args.u_max),
        # Useful experiment metadata
        control_dt=np.array(0.01),
        state_dim=np.array(state_history.shape[1] if state_history.ndim == 2 else 120),
        control_dim=np.array(2),
    )

    print(f"\nSaved LQR run data: {args.log_npz}")
    print(f"  command: {command_history.shape}")
    print(f"  state:   {state_history.shape}")
    print(f"  RMSE:    {rmse_history.shape}")

    # ========================================================
    # Plots
    # ========================================================

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # -- final shape --------------------------------------------------------
    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], "r--", linewidth=2, label="Goal")
    ax[0].plot(final_pos[:, 0], final_pos[:, 1], "b-", linewidth=2, label="Final")
    ax[0].scatter(final_pos[0, 0], final_pos[0, 1], s=60, c="k", zorder=5,
                  label="Node 0 (driven)")
    ax[0].scatter(final_pos[-1, 0], final_pos[-1, 1], s=60, c="g", zorder=5,
                  label="Anchor")
    ax[0].set_xlabel("X [mm]")
    ax[0].set_ylabel("Y [mm]")
    ax[0].set_title(f"Final shape — settle RMSE {settle_rmse:.2f} mm")
    ax[0].axis("equal")
    ax[0].grid(True)
    ax[0].legend()

    # -- error trace, with the control/hold boundary marked -----------------
    ax[1].plot(rmse_history, lw=1.5)
    ax[1].axvline(n_control, color="k", ls="--", lw=1,
                  label=f"control → hold ({stop_reason})")
    ax[1].axhline(settle_rmse, color="r", ls="--", lw=1,
                  label=f"settle {settle_rmse:.2f} mm")
    ax[1].axhline(recon_floor, color="gray", ls=":", lw=1,
                  label=f"recon floor {recon_floor:.2f} mm")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("RMSE [mm]")
    ax[1].set_title("tracking error")
    ax[1].grid(True)
    ax[1].legend()

    # -- per-node error -----------------------------------------------------
    ax[2].plot(per_node, "o-", ms=4)
    ax[2].set_xlabel("node index (0 = driven, 29 = anchor)")
    ax[2].set_ylabel("final error [mm]")
    ax[2].set_title("per-node error")
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig(args.log_npz.replace(".npz", ".png"), dpi=130)
    print(f"Saved plot: {args.log_npz.replace('.npz', '.png')}")
    plt.show()


# ============================================================
# Arguments
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shape",
        default="rope_all_shapes/rope_L.json",
    )

    parser.add_argument(
        "--results-dir",
        default="training_results",
    )

    parser.add_argument(
        "--exp-name",
        default="exp_simple_1788069999",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=20,
        help="LQR prediction horizon",
    )

    parser.add_argument(
        "--u-max",
        type=float,
        default=1.5,
        help="maximum command [mm/step]",
    )

    parser.add_argument(
        "--r-weight",
        type=float,
        default=0.1,
        help="control penalty. Q now has unit mean eigenvalue, so this is comparable "
             "to its old value against Q = I -- but Q's SHAPE changed, so retune.",
    )

    parser.add_argument(
        "--qf-weight",
        type=float,
        default=200.0,
        help="terminal cost multiplier on Q",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="maximum control steps before the hold phase",
    )

    parser.add_argument(
        "--plateau-steps",
        type=int,
        default=10,
        help="window length for both the plateau and divergence tests",
    )

    parser.add_argument(
        "--plateau-tol",
        type=float,
        default=0.05,
        help="mm; RMSE range over the window below this counts as converged. "
             "Set just above the ringing amplitude in your traces.",
    )

    parser.add_argument(
        "--diverge-tol",
        type=float,
        default=10,
        help="mm; stop control once the windowed-mean RMSE exceeds its best by this "
             "much. Set above the ringing amplitude or it will trip on oscillation.",
    )

    parser.add_argument(
        "--hold-steps",
        type=int,
        default=200,
        help="maximum hold (u = 0) steps after control stops",
    )

    parser.add_argument(
        "--settle-vel",
        type=float,
        default=5.0,
        help="max|v| threshold [mm/s] for the settling test",
    )

    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="consecutive steps below the threshold before the hold ends; also the "
             "window settle_rmse is averaged over",
    )

    parser.add_argument(
        "--log-npz",
        default="lqr_run.npz",
        help="output NPZ file for commands, states, and LQR parameters",
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()