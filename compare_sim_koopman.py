"""
compare_sim_koopman.py

Drives the PyElastica rope (rope_manip_env.RopeManipEnv) with a random sequence of
per-step commands (cmd_dx, cmd_dy) and, from the same starting state, rolls the
trained Koopman model (ngk_simple.KoopmanModel, loaded via mpc.load_koopman) forward
under the identical *applied* commands. Plots the two rope trajectories together so
model drift from the real simulator is visible directly, both as an animation and as
a position-error-vs-time curve.

The command sequence is clipped to stay within the rope's reach exactly as
koopman_data_gen.py does during data collection, so both rollouts see the same
actually-applied commands (not the raw, possibly-clipped, random draw).

Usage
-----
    python compare_sim_koopman.py --steps 150 --seed 0
    python compare_sim_koopman.py --steps 300 --action-hold 5 --mag-max 2.0 --out-dir vis_koopman
"""

import argparse
from pathlib import Path

import numpy as np
import torch


from rope_manip_env import RopeManipEnv, MM
from mpc import load_koopman


# --------------------------------------------------------------------------- #
# Random command generation (mirrors the "turn_short"-style per-step cases in
# koopman_data_gen.py: a fixed (magnitude, angle) held for `action_hold` steps,
# then resampled).
# --------------------------------------------------------------------------- #
def random_command_sequence(steps, rng, mag_range=(0.2, 1.5), action_hold=10):
    cmds = np.zeros((steps, 2))
    mag = angle = None
    for k in range(steps):
        if k % action_hold == 0:
            mag = rng.uniform(*mag_range)
            angle = rng.uniform(0.0, 2 * np.pi)
        cmds[k] = [mag * np.cos(angle), mag * np.sin(angle)]
    return cmds


def clip_to_reach(target, fixed_node, max_reach):
    d = target - fixed_node
    n = np.linalg.norm(d)
    if n > max_reach:
        return fixed_node + d * (max_reach / n)
    return target


def build_applied_commands(start_target, fixed_node, max_reach, raw_cmds):
    """Resolves the raw random draws against the reach limit, exactly as the
    collector does, so both rollouts below see identical applied commands."""
    applied = np.zeros_like(raw_cmds)
    target = start_target.copy()
    for k, cmd in enumerate(raw_cmds):
        clipped = clip_to_reach(target + cmd, fixed_node, max_reach)
        applied[k] = clipped - target
        target = clipped
    return applied


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def state_vector_from_env(env, dataset):
    """Packs the sim's current positions+velocities into the model's state layout
    (node-major x,y within each feature block; see RopeKoopmanDataset)."""
    x = np.zeros(dataset.state_dim, dtype=np.float32)
    x[dataset.feature_slices["pos"]] = env.get_positions(mm=True)[:, :2].reshape(-1)
    if "vel" in dataset.feature_slices:
        x[dataset.feature_slices["vel"]] = env.get_velocities(mm=True)[:, :2].reshape(-1)
    return x


def rollout_sim(env, cmds_mm):
    """Applies cmds_mm (mm/step) to the real PyElastica sim. Returns positions
    [T+1, N, 2] mm, index 0 = the state before any command."""
    pos = [env.get_positions(mm=True)[:, :2].copy()]
    for cmd in cmds_mm:
        env.drive_step(cmd / MM)
        pos.append(env.get_positions(mm=True)[:, :2].copy())
    return np.stack(pos)


def rollout_koopman(model, dataset, x0_mm, cmds_mm):
    """Rolls the Koopman latent dynamics forward from x0 (mm, full state vector)
    under the same applied commands. Returns decoded positions [T+1, N, 2] mm."""
    mean = torch.as_tensor(dataset.state_mean[0])
    std = torch.as_tensor(dataset.state_std[0])
    pos_slice = dataset.pos_slice
    n_nodes = (pos_slice.stop - pos_slice.start) // 2

    x0 = torch.as_tensor(x0_mm, dtype=torch.float32)
    positions = [x0[pos_slice].reshape(n_nodes, 2).numpy().copy()]

    with torch.no_grad():
        z = model.encoder(((x0 - mean) / std).unsqueeze(0))
        for cmd in cmds_mm:
            u = torch.as_tensor(cmd, dtype=torch.float32).unsqueeze(0) / dataset.cmd_scale
            u_lifted = model.input_encoder(u)
            z = model.operator(z, u_lifted)
            x_hat = model.decoder(z) * std + mean
            positions.append(x_hat[0, pos_slice].reshape(n_nodes, 2).numpy().copy())
    return np.stack(positions)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def animate(sim_pos, koop_pos, out_path, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    allp = np.concatenate([sim_pos.reshape(-1, 2), koop_pos.reshape(-1, 2)], axis=0)
    lo, hi = allp.min(0) - 10, allp.max(0) + 10

    fig, ax = plt.subplots(figsize=(6, 6))
    sim_line, = ax.plot([], [], "-o", ms=3, color="tab:blue", label="actual sim (PyElastica)")
    koop_line, = ax.plot([], [], "--o", ms=3, color="tab:orange", label="Koopman prediction")
    tip_sim, = ax.plot([], [], "o", ms=9, color="tab:blue")
    tip_koop, = ax.plot([], [], "o", ms=9, color="tab:orange")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")

    T = len(sim_pos)

    def upd(i):
        sim_line.set_data(sim_pos[i, :, 0], sim_pos[i, :, 1])
        koop_line.set_data(koop_pos[i, :, 0], koop_pos[i, :, 1])
        tip_sim.set_data([sim_pos[i, 0, 0]], [sim_pos[i, 0, 1]])
        tip_koop.set_data([koop_pos[i, 0, 0]], [koop_pos[i, 0, 1]])
        rmse = np.sqrt(((sim_pos[i] - koop_pos[i]) ** 2).mean())
        title.set_text(f"t = {i}/{T - 1}   node-position RMSE = {rmse:.2f} mm")
        return sim_line, koop_line, tip_sim, tip_koop

    FuncAnimation(fig, upd, frames=T, blit=False).save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved animation   -> {out_path}")


def plot_error(sim_pos, koop_pos, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmse = np.sqrt(((sim_pos - koop_pos) ** 2).mean(axis=(1, 2)))
    node0_err = np.linalg.norm(sim_pos[:, 0] - koop_pos[:, 0], axis=-1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rmse, label="whole-rope position RMSE (mm)")
    ax.plot(node0_err, label="driven node-0 error (mm)")
    ax.set_xlabel("step"); ax.set_ylabel("mm"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Koopman rollout drift vs. actual PyElastica simulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved error plot  -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Compare a random-command rollout in the real PyElastica rope sim "
                    "(rope_manip_env) against the trained Koopman model's (ngk_simple) "
                    "latent-space prediction, starting from the same state.")
    p.add_argument("--results-dir", default="training_results")
    p.add_argument("--checkpoint", default=None, help="Checkpoint dir; default = most recent.")
    p.add_argument("--csv", nargs="+", default=None,
                   help="Override the CSVs used to recover normalisation stats "
                        "(default: whatever the checkpoint's run was trained on).")
    p.add_argument("--steps", type=int, default=150, help="Number of random commands to apply.")
    p.add_argument("--action-hold", type=int, default=10,
                   help="Steps between random direction/magnitude resamples.")
    p.add_argument("--mag-min", type=float, default=0.2, help="Min per-step cmd magnitude (mm).")
    p.add_argument("--mag-max", type=float, default=1.5, help="Max per-step cmd magnitude (mm).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-randomize-shape", action="store_true",
                   help="Skip the initial random-force shape kick; start from the straight rope.")
    p.add_argument("--out-dir", default="koopman_vis", help="Directory for the gif/png outputs.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Koopman model + normalisation stats ...")
    model, dataset, device, exp_name = load_koopman(args.results_dir, args.checkpoint, args.csv)
    print(f"  checkpoint '{exp_name}' | state_dim {dataset.state_dim} | "
          f"features {list(dataset.features)}")

    print("Building PyElastica rope env ...")
    env = RopeManipEnv(seed=args.seed)
    env.reset()
    if not args.no_randomize_shape:
        env.randomize_shape()

    fixed_node = env.fixed_position(mm=True)[:2]
    max_reach = 0.9 * env.rope_reach(mm=True)
    start_target = env.driven_position(mm=True)[:2].copy()

    raw_cmds = random_command_sequence(
        args.steps, rng, mag_range=(args.mag_min, args.mag_max), action_hold=args.action_hold)
    cmds_mm = build_applied_commands(start_target, fixed_node, max_reach, raw_cmds)

    x0_mm = state_vector_from_env(env, dataset)

    print(f"Rolling out {args.steps} random-command steps in the real sim ...")
    sim_pos = rollout_sim(env, cmds_mm)
    if env.is_exploded():
        print("WARNING: the real sim exploded during this rollout; results past that "
              "point are meaningless.")

    print("Rolling out the same commands through the Koopman model ...")
    koop_pos = rollout_koopman(model, dataset, x0_mm, cmds_mm)

    rmse_h1 = np.sqrt(((sim_pos[1] - koop_pos[1]) ** 2).mean())
    rmse_final = np.sqrt(((sim_pos[-1] - koop_pos[-1]) ** 2).mean())
    print(f"position RMSE: {rmse_h1:.3f} mm @ h=1, {rmse_final:.3f} mm @ h={args.steps}")

    animate(sim_pos, koop_pos, out_dir / "sim_vs_koopman.gif")
    plot_error(sim_pos, koop_pos, out_dir / "sim_vs_koopman_error.png")


if __name__ == "__main__":
    main()
