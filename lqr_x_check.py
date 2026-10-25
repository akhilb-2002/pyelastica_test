"""
letter_lqr_xonly.py

Simple constrained finite-horizon LQR for rope shaping
using an x-only Koopman model.

State:
    s = x

    x : 60 position values
    s : 60 dimensional

Control:
    u = [dx, dy]   [mm / step]

Koopman model:
    z_{k+1} = A z_k + B u_k

Controller:
    finite-horizon quadratic optimal control
    with simple box constraints on u.

Important:
    Velocity is NOT part of the Koopman state.

    Velocity is measured from the simulator only for:
        - logging
        - evaluating settling behavior

    The controller only sees x.
"""

import argparse
import ast
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch
import matplotlib.pyplot as plt

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM

# ============================================================
# Koopman model
# ============================================================


class KoopmanController:

    def __init__(
        self,
        dataset,
        model,
        A,
        B,
        mean,
        std,
    ):
        self.dataset = dataset
        self.model = model

        self.A = A
        self.B = B

        self.mean = mean
        self.std = std

        self.z_dim = A.shape[0]

    def encode(self, state):
        """
        Convert x-only physical state into Koopman state.

        state:
            [x] -> 60 dimensions
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        state_norm = (torch.as_tensor(state).unsqueeze(0) - self.mean) / self.std

        with torch.no_grad():

            z = self.model.encoder(state_norm)

        return z.numpy().ravel().astype(np.float64)

    def decode(self, z):
        """
        Decode Koopman state back to x-only physical state.
        """

        z_t = torch.as_tensor(
            z,
            dtype=torch.float32,
        ).unsqueeze(0)

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

        # Updated every control step
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
        # ----------------------------------------------------

        cost = 0

        for k in range(self.H):

            error = self.z[:, k] - self.z_goal

            cost += cp.quad_form(
                error,
                self.Q,
            )

            cost += cp.quad_form(
                self.u[:, k],
                self.R,
            )

        # Terminal cost
        terminal_error = self.z[:, self.H] - self.z_goal

        cost += cp.quad_form(
            terminal_error,
            self.Qf,
        )

        # ----------------------------------------------------
        # Problem
        # ----------------------------------------------------

        self.problem = cp.Problem(
            cp.Minimize(cost),
            constraints,
        )

    def solve(
        self,
        z_current,
        z_goal,
    ):

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
            raise RuntimeError(f"LQR solver failed: " f"{self.problem.status}")

        # Only apply first control action
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

        raise FileNotFoundError(f"Cannot find training log:\n" f"{log_path}")

    # Same mechanism as existing training code
    ta = ast.literal_eval(log_path.read_text().splitlines()[0].split("Args: ", 1)[1])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = RopeKoopmanDataset(
        ta["csv"],
        features=tuple(ta["features"]),
        horizon=1,
        normalize=not ta.get(
            "no_normalize",
            False,
        ),
        clamp_tol_mm=None,
    )

    print(
        "Dataset state dimension:",
        dataset.state_dim,
    )

    print(
        "Dataset features:",
        ta["features"],
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # This must be an x-only model.
    # --------------------------------------------------------

    if "vel" in dataset.feature_slices:

        raise RuntimeError(
            "This script expects an x-only Koopman model, "
            "but the loaded dataset contains velocity."
        )

    print("Using x-only Koopman state.")

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
        operator_bias=not ta.get(
            "no_operator_bias",
            False,
        ),
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

    # Operator was trained using u / cmd_scale
    B = B / dataset.cmd_scale

    print(
        "A shape:",
        A.shape,
    )

    print(
        "B shape:",
        B.shape,
    )

    print(
        "rho(A):",
        np.max(np.abs(np.linalg.eigvals(A))),
    )

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
    # Load x-only Koopman model
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

    n_nodes = dataset.state_dim // 2

    print(
        "Number of rope nodes:",
        n_nodes,
    )

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

    with open(
        args.shape,
        "r",
    ) as f:

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
    # X-only goal state
    # --------------------------------------------------------

    goal_x = goal_pos.reshape(-1)

    goal_state = goal_x.copy()

    print(
        "Goal state dimension:",
        goal_state.shape,
    )

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    z_goal = koopman.encode(goal_state)

    print(
        "Goal latent dimension:",
        z_goal.shape,
    )

    # ========================================================
    # LQR weights
    # ========================================================

    z_dim = A.shape[0]

    # --------------------------------------------------------
    # Simple latent-space weighting
    # --------------------------------------------------------

    Q = np.eye(z_dim)

    R = 0.01 * np.eye(2)

    Qf = 500 * Q

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
    # Control loop
    # ========================================================

    print()
    print("=" * 60)
    print("Starting X-only LQR")
    print("=" * 60)

    print(f"State dimension : {dataset.state_dim}")

    print(f"Latent dimension: {z_dim}")

    print(f"Horizon         : {args.H}")

    print(f"u_max           : {args.u_max}")

    print()

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    rmse_history = []
    command_history = []

    # IMPORTANT:
    # State is now only x -> [N, 60]
    state_history = []

    # Velocity is measured only for evaluation
    velocity_history = []
    max_vel_history = []

    # ========================================================
    # Loop
    # ========================================================

    for step in range(args.steps):

        # ----------------------------------------------------
        # Measure position
        # ----------------------------------------------------

        pos = env.get_positions(mm=True)[:, :2].reshape(-1)

        # ----------------------------------------------------
        # Measure velocity
        #
        # NOT used by the controller.
        # Only used for evaluation.
        # ----------------------------------------------------

        vel = env.get_velocities(mm=True)[:, :2]

        max_vel = np.max(
            np.linalg.norm(
                vel,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # X-only state
        # ----------------------------------------------------

        state = pos.copy()

        # ----------------------------------------------------
        # Log x-only state
        # ----------------------------------------------------

        state_history.append(state.copy())

        # ----------------------------------------------------
        # Encode current state
        # ----------------------------------------------------

        z_current = koopman.encode(state)

        # ----------------------------------------------------
        # Solve LQR
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
        # Measure new position
        # ----------------------------------------------------

        new_pos = env.get_positions(mm=True)[:, :2]

        # ----------------------------------------------------
        # Physical position RMSE
        # ----------------------------------------------------

        rmse = np.sqrt(
            np.mean(
                np.sum(
                    (new_pos - goal_pos) ** 2,
                    axis=1,
                )
            )
        )

        # ----------------------------------------------------
        # Velocity after action
        # ----------------------------------------------------

        new_vel = env.get_velocities(mm=True)[:, :2]

        max_vel = np.max(
            np.linalg.norm(
                new_vel,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        rmse_history.append(rmse)

        command_history.append(u.copy())

        max_vel_history.append(max_vel)

        velocity_history.append(new_vel.reshape(-1).copy())

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if step % 10 == 0:

            print(
                f"step {step:4d} | "
                f"u = [{u[0]:7.3f}, "
                f"{u[1]:7.3f}] | "
                f"RMSE = {rmse:8.3f} mm | "
                f"Max Vel = {max_vel:8.3f} mm/s"
            )

        # ----------------------------------------------------
        # Stop if simulation exploded
        # ----------------------------------------------------

        if env.is_exploded():

            print("Simulation exploded.")

            break

        # ----------------------------------------------------
        # Optional velocity stopping condition
        # ----------------------------------------------------

        if args.stop_vel is not None and max_vel < args.stop_vel:

            print()
            print(
                f"Max velocity "
                f"{max_vel:.3f} mm/s "
                f"is below threshold "
                f"{args.stop_vel:.3f} mm/s."
            )

            print("Stopping.")

            break

    # ========================================================
    # Final result
    # ========================================================

    final_pos = env.get_positions(mm=True)[:, :2]

    final_rmse = np.sqrt(
        np.mean(
            np.sum(
                (final_pos - goal_pos) ** 2,
                axis=1,
            )
        )
    )

    final_vel = env.get_velocities(mm=True)[:, :2]

    final_max_vel = np.max(
        np.linalg.norm(
            final_vel,
            axis=1,
        )
    )

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(f"Final RMSE: " f"{final_rmse:.3f} mm")

    print(f"Final Max Velocity: " f"{final_max_vel:.3f} mm/s")

    # ========================================================
    # Convert logs
    # ========================================================

    state_history = np.asarray(
        state_history,
        dtype=np.float64,
    )

    command_history = np.asarray(
        command_history,
        dtype=np.float64,
    )

    rmse_history = np.asarray(
        rmse_history,
        dtype=np.float64,
    )

    max_vel_history = np.asarray(
        max_vel_history,
        dtype=np.float64,
    )

    velocity_history = np.asarray(
        velocity_history,
        dtype=np.float64,
    )

    # ========================================================
    # Save
    # ========================================================

    np.savez(
        args.log_npz,
        # ----------------------------------------------------
        # Per-step data
        # ----------------------------------------------------
        command=command_history,
        cmd_x=command_history[:, 0],
        cmd_y=command_history[:, 1],
        # X-only state
        state=state_history,
        rmse=rmse_history,
        # Velocity is logged separately
        max_vel=max_vel_history,
        velocity=velocity_history,
        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------
        goal_state=goal_state,
        goal_pos=goal_pos,
        z_goal=z_goal,
        # ----------------------------------------------------
        # LQR parameters
        # ----------------------------------------------------
        A=A,
        B=B,
        Q=Q,
        R=R,
        Qf=Qf,
        horizon=np.array(args.H),
        u_max=np.array(args.u_max),
        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------
        control_dt=np.array(0.01),
        state_dim=np.array(dataset.state_dim),
        control_dim=np.array(2),
    )

    print(f"Saved LQR run data: " f"{args.log_npz}")

    print(f"  command:  " f"{command_history.shape}")

    print(f"  state:    " f"{state_history.shape}")

    print(f"  velocity: " f"{velocity_history.shape}")

    print(f"  RMSE:     " f"{rmse_history.shape}")

    # ========================================================
    # Plot final rope shape
    # ========================================================

    plt.figure(figsize=(8, 6))

    # Goal
    plt.plot(
        goal_pos[:, 0],
        goal_pos[:, 1],
        "r--",
        linewidth=2,
        label="Goal",
    )

    # Final
    plt.plot(
        final_pos[:, 0],
        final_pos[:, 1],
        "b-",
        linewidth=2,
        label="Final",
    )

    # Base
    plt.scatter(
        final_pos[0, 0],
        final_pos[0, 1],
        s=60,
        label="Base",
    )

    # Tip
    plt.scatter(
        final_pos[-1, 0],
        final_pos[-1, 1],
        s=60,
        label="Final tip",
    )

    plt.xlabel("X [mm]")

    plt.ylabel("Y [mm]")

    plt.title(f"Final Rope Shape — " f"RMSE = {final_rmse:.2f} mm")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Arguments
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shape",
        default="rope_all_shapes/rope_C.json",
    )

    parser.add_argument(
        "--results-dir",
        default="training_results",
    )

    parser.add_argument(
        "--exp-name",
        default="exp_simple_1788269928",
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
        "--steps",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--stop-vel",
        type=float,
        default=None,
        help=(
            "stop when max velocity falls below "
            "this value [mm/s]. Disabled by default."
        ),
    )

    parser.add_argument(
        "--log-npz",
        default="lqr_xonly_run.npz",
        help="output NPZ file",
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
