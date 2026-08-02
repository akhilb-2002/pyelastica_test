"""
Matplotlib animation of the PyElastica rope-manipulation env.

Adapted from the standard PyElastica plotting utilities (plot_position /
plot_video_* ) for the horizontal table rope in `rope_manip_env.py`:

  * the rope lies on the z = 0 table and moves in the xy-plane, so xy is the
    primary view (xz / 3d are provided too, useful for the live rollout which
    carries z; a CSV trajectory has no z and shows flat);
  * nodes are drawn as a connected rope with the driven end (Node 0) and the
    fixed end highlighted, plus a fading trail of the driven-node path;
  * when animating a collected CSV, ALL trajectories are played back-to-back and
    the current trajectory id / action id are shown as live legend entries;
  * the CSV animation is written into the CSV's own folder;
  * saving falls back from mp4 (ffmpeg) to gif (pillow) automatically.

`plot_params` follows the framework layout, extended with per-frame metadata:
    {"time": [...], "position": [(3, N), ...], "traj_id": [...], "action_id": [...]}
with each position frame shaped (3, n_nodes) == (x/y/z, node).

Run directly to render a demo rollout:
    python animate_rope.py                 # -> rope_xy.gif
Animate every trajectory in a collected CSV (gif saved next to the CSV):
    python animate_rope.py --csv csv_timeseries_pyelastica/case_9_long_mixed/collected_trajectories_koopman_timeseries.csv
"""

import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as manimation

from rope_manip_env import RopeManipEnv, MM


# --------------------------------------------------------------------------- #
# Rollout collection  ->  plot_params
# --------------------------------------------------------------------------- #
def collect_rollout(env, n_actions=8, T=20, mag_range_mm=(5.0, 45.0),
                    max_reach_frac=0.9, randomize=True, seed=0):
    """Drive Node 0 through reach-limited random ramps; record one frame/step.

    Returns a plot_params dict with position frames shaped (3, n_nodes) in metres,
    plus per-frame traj_id (all 1 here) and action_id metadata.
    """
    rng = np.random.default_rng(seed)
    env.reset()
    if randomize:
        env.randomize_shape()

    fixed_xy = env.fixed_position()[:2]
    reach = env.rope_reach()
    limit = max_reach_frac * reach

    times, positions, traj_id, action_id = [], [], [], []
    t = 0.0
    times.append(t)
    positions.append(env.get_positions().T.copy())      # (3, N)
    traj_id.append(1)
    action_id.append(0)

    target_xy = env.driven_position()[:2].copy()
    for a in range(n_actions):
        for _ in range(25):
            mag = rng.uniform(*mag_range_mm) / MM        # mm -> m
            ang = rng.uniform(0.0, 2 * np.pi)
            endpoint = target_xy + mag * np.array([np.cos(ang), np.sin(ang)])
            if np.linalg.norm(endpoint - fixed_xy) <= limit:
                break
        step = np.array([mag * np.cos(ang), mag * np.sin(ang)]) / T
        for _ in range(T):
            env.drive_step(step)
            target_xy = target_xy + step
            t += env.control_dt
            times.append(t)
            positions.append(env.get_positions().T.copy())
            traj_id.append(1)
            action_id.append(a)
    return {"time": times, "position": positions,
            "traj_id": traj_id, "action_id": action_id}


def load_rollout_from_csv(csv_path, trajectory_id=None, z_mm=3.0):
    """Build plot_params from a collected CSV (positions in mm).

    trajectory_id : None -> use every trajectory (played back-to-back);
                    an int -> only that trajectory_id.
    The CSV stores only x, y per node; z is set to the rope radius for display.
    Carries per-frame traj_id / action_id so the animation can label them.
    """
    import csv as _csv

    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    header, data = rows[0], np.array(rows[1:], dtype=float)
    col = {h: i for i, h in enumerate(header)}
    n_nodes = sum(h.startswith("pos_x_") for h in header)

    if trajectory_id is not None:
        data = data[data[:, col["trajectory_id"]] == trajectory_id]
    # keep CSV order (trajectory, action, step are already sequential)

    xs = data[:, [col[f"pos_x_{i}"] for i in range(n_nodes)]]   # (T, N) mm
    ys = data[:, [col[f"pos_y_{i}"] for i in range(n_nodes)]]
    frames = [np.vstack([xs[k], ys[k], np.full(n_nodes, z_mm)]) / MM
              for k in range(len(data))]                        # (3, N) metres
    times = list(np.arange(len(data)) * 0.01)
    return {"time": times, "position": frames,
            "traj_id": data[:, col["trajectory_id"]].astype(int).tolist(),
            "action_id": data[:, col["action_idx"]].astype(int).tolist()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save(anim, path, fps):
    path = str(path)
    ext = path.rsplit(".", 1)[-1].lower()
    writers = manimation.writers.list()
    try:
        if ext == "mp4" and "ffmpeg" in writers:
            writer = manimation.FFMpegWriter(
                fps=fps, metadata=dict(artist="rope_manip_env"), bitrate=2400)
        else:
            if ext == "mp4":
                path = path[:-4] + ".gif"
                print("ffmpeg unavailable -> writing gif instead:", path)
            writer = manimation.PillowWriter(fps=fps)
        anim.save(path, writer=writer)
        print("saved", path)
    except Exception as e:  # pragma: no cover
        print("save failed:", e)
    return path


def _traj_block_starts(traj_id):
    """First frame index of each frame's contiguous trajectory block."""
    starts = np.zeros(len(traj_id), dtype=int)
    cur = 0
    for i in range(1, len(traj_id)):
        if traj_id[i] != traj_id[i - 1]:
            cur = i
        starts[i] = cur
    return starts


def _dynamic_legend(ax, static_handles, static_labels, has_meta, loc="upper right"):
    """Build a legend; if has_meta, append live traj/action entries and return
    their text objects for per-frame updates."""
    handles = list(static_handles)
    labels = list(static_labels)
    if has_meta:
        handles += [Line2D([], [], color="none"), Line2D([], [], color="none")]
        labels += ["traj —", "action —"]
    leg = ax.legend(handles=handles, labels=labels, loc=loc)
    if has_meta:
        return leg.get_texts()[-2], leg.get_texts()[-1]
    return None, None


def _limits(pos, ax_a, ax_b, pad=0.03):
    a, b = pos[:, ax_a, :], pos[:, ax_b, :]
    return (a.min() - pad, a.max() + pad), (b.min() - pad, b.max() + pad)


# --------------------------------------------------------------------------- #
# Static plot: driven / fixed / tip trajectory  (analogue of plot_position)
# --------------------------------------------------------------------------- #
def plot_tip_trajectory(plot_params, filename="tip_trajectory.png", save=False):
    pos = np.array(plot_params["position"])           # (T, 3, N)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.grid(which="major", color="0.7", linestyle="-")
    ax.plot(pos[:, 0, 0], pos[:, 1, 0], "g-", lw=1.5, label="driven node (0)")
    ax.plot(pos[0, 0, -1], pos[0, 1, -1], "rs", ms=10, label="fixed end")
    ax.plot(pos[0, 0], pos[0, 1], "-", color="0.6", label="initial shape")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    if save:
        fig.savefig(filename)
    return fig


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #
def animate_xy(plot_params, video_name="rope_xy.gif", fps=30, trail=40, save=True):
    """Top-down xy animation; plays all trajectories with a live traj/action legend."""
    time_list = plot_params["time"]
    pos = np.array(plot_params["position"])           # (T, 3, N)
    traj_id = plot_params.get("traj_id")
    action_id = plot_params.get("action_id")
    has_meta = traj_id is not None and action_id is not None
    starts = _traj_block_starts(traj_id) if has_meta else np.zeros(len(pos), int)
    (xlim, ylim) = _limits(pos, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(color="0.85", linestyle="-")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    (rope_line,) = ax.plot([], [], "-", color="0.35", lw=2, zorder=2)
    (nodes,) = ax.plot([], [], "o", color="0.35", ms=3, zorder=3)
    (driven,) = ax.plot([], [], "o", color="tab:green", ms=10, zorder=5)
    (fixed,) = ax.plot([], [], "s", color="tab:red", ms=10, zorder=5)
    (trail_line,) = ax.plot([], [], "-", color="tab:green", lw=1, alpha=0.5, zorder=1)
    title = ax.set_title("")
    traj_text, action_text = _dynamic_legend(
        ax, [driven, fixed], ["driven (0)", "fixed end"], has_meta)

    def update(i):
        f = pos[i]
        rope_line.set_data(f[0], f[1])
        nodes.set_data(f[0], f[1])
        driven.set_data([f[0, 0]], [f[1, 0]])
        fixed.set_data([f[0, -1]], [f[1, -1]])
        j = max(i - trail, int(starts[i]))            # trail resets per trajectory
        trail_line.set_data(pos[j:i + 1, 0, 0], pos[j:i + 1, 1, 0])
        title.set_text(f"t = {time_list[i]:.2f} s")
        if has_meta:
            traj_text.set_text(f"traj {traj_id[i]}")
            action_text.set_text(f"action {action_id[i]}")
        return rope_line, nodes, driven, fixed, trail_line, title

    anim = manimation.FuncAnimation(
        fig, update, frames=len(time_list), interval=1000 / fps, blit=False)
    if save:
        _save(anim, video_name, fps)
    return anim


def animate_xz(plot_params, video_name="rope_xz.gif", fps=30, save=True):
    """Side xz view (shows the rope on the table; ~flat for CSV trajectories)."""
    time_list = plot_params["time"]
    pos = np.array(plot_params["position"])
    traj_id = plot_params.get("traj_id")
    action_id = plot_params.get("action_id")
    has_meta = traj_id is not None and action_id is not None
    (xlim, zlim) = _limits(pos, 0, 2, pad=0.02)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.axhline(0.0, color="0.7", lw=1)                # table surface
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    (rope_line,) = ax.plot([], [], "-o", color="0.35", ms=3, lw=2)
    title = ax.set_title("")
    traj_text, action_text = _dynamic_legend(ax, [rope_line], ["rope"], has_meta)

    def update(i):
        f = pos[i]
        rope_line.set_data(f[0], f[2])
        title.set_text(f"t = {time_list[i]:.2f} s")
        if has_meta:
            traj_text.set_text(f"traj {traj_id[i]}")
            action_text.set_text(f"action {action_id[i]}")
        return rope_line, title

    anim = manimation.FuncAnimation(
        fig, update, frames=len(time_list), interval=1000 / fps, blit=False)
    if save:
        _save(anim, video_name, fps)
    return anim


def animate_3d(plot_params, video_name="rope_3d.gif", fps=30, save=True):
    time_list = plot_params["time"]
    pos = np.array(plot_params["position"])
    traj_id = plot_params.get("traj_id")
    action_id = plot_params.get("action_id")
    has_meta = traj_id is not None and action_id is not None
    (xlim, ylim) = _limits(pos, 0, 1)
    zlim = (0.0, max(0.02, pos[:, 2, :].max() + 0.02))
    fig = plt.figure(figsize=(8, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    (rope_line,) = ax.plot([], [], [], "-o", color="0.35", ms=3, lw=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    title = ax.set_title("")
    traj_text, action_text = _dynamic_legend(ax, [rope_line], ["rope"], has_meta)

    def update(i):
        f = pos[i]
        rope_line.set_data(f[0], f[1])
        rope_line.set_3d_properties(f[2])
        title.set_text(f"t = {time_list[i]:.2f} s")
        if has_meta:
            traj_text.set_text(f"traj {traj_id[i]}")
            action_text.set_text(f"action {action_id[i]}")
        return rope_line, title

    anim = manimation.FuncAnimation(
        fig, update, frames=len(time_list), interval=1000 / fps, blit=False)
    if save:
        _save(anim, video_name, fps)
    return anim


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Animate the PyElastica rope manipulation env.")
    ap.add_argument("--csv", default=None, help="Animate trajectories from a collected CSV.")
    ap.add_argument("--traj", type=int, default=None,
                    help="Restrict to one trajectory_id (default: all trajectories).")
    ap.add_argument("--view", choices=["xy", "xz", "3d"], default="xy")
    ap.add_argument("--out", default=None, help="Output filename (.gif or .mp4).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1,
                    help="Keep every Nth frame (shrinks long multi-trajectory gifs).")
    ap.add_argument("--n-actions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="Display interactively instead of saving.")
    args = ap.parse_args()

    if args.csv:
        scope = f"trajectory {args.traj}" if args.traj is not None else "all trajectories"
        print(f"Loading {scope} from {args.csv} ...")
        params = load_rollout_from_csv(args.csv, trajectory_id=args.traj)
    else:
        print("Running a demo rollout ...")
        env = RopeManipEnv(seed=args.seed)
        params = collect_rollout(env, n_actions=args.n_actions, seed=args.seed)

    if args.stride > 1:  # subsample frames uniformly
        for k in ("time", "position", "traj_id", "action_id"):
            params[k] = params[k][:: args.stride]
    print(f"{len(params['time'])} frames"
          + (f" across {len(set(params['traj_id']))} trajectories" if params.get('traj_id') else ""))

    default_out = {"xy": "rope_xy.gif", "xz": "rope_xz.gif", "3d": "rope_3d.gif"}[args.view]
    out_name = args.out or default_out
    # CSV animations are written into the CSV's own folder.
    out = (Path(args.csv).resolve().parent / Path(out_name).name) if args.csv else out_name

    animate = {"xy": animate_xy, "xz": animate_xz, "3d": animate_3d}[args.view]
    anim = animate(params, video_name=out, fps=args.fps, save=not args.show)
    if args.show:
        plt.show()