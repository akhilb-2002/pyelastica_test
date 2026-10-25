"""
render_lqr_animation.py

Read an LQR run NPZ (as written by const_lqr.py) and turn it into an animation.

The commands stored in the NPZ (`cmd_x`, `cmd_y`, in mm/step) are replayed
through the PyElastica environment (`rope_manip_env.RopeManipEnv`) with the same
parameters const_lqr.py used, so the rope shape for every frame comes out of the
physics engine, not out of the log. The replayed positions are checked against
the logged `state` array and the mismatch is reported.

Two rendering backends:

  povray      ray-traced frames using the Case1_rendering scene
              (camera_position.inc + the sphere_sweep rope of
              generate_povray_rope.py). Needs the `povray` binary.
  matplotlib  fast top-down 2D plot of the rope, the goal shape and the
              driven-end trail. No external binary needed.

Frames are assembled into a GIF with ffmpeg (palettegen/paletteuse) when ffmpeg
is available, otherwise with Pillow.

Examples
--------
    python render_lqr_animation.py                       # povray, default npz
    python render_lqr_animation.py --backend matplotlib  # quick preview
    python render_lqr_animation.py --npz lqr_runs/lqr_run_C.npz \
        --out lqr_runs/lqr_run_C.gif --stride 2 --fps 25
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rope_manip_env import MM, RopeManipEnv  # noqa: E402

RENDER_DIR = (
    HERE
    / "GazzolaLab Elastica-RL-control 1e708f889b81c572ba4d78d2df7213688e0e7ee3 "
      "visualization-Case1_rendering"
)
CAMERA_INC = RENDER_DIR / "camera_position.inc"

# ---------------------------------------------------------------------------
# POV-Ray scene constants (kept in sync with generate_povray_rope.py)
# ---------------------------------------------------------------------------
SCALE = 100.0  # metres -> POV units; must match camera_position.inc

# Named colours for --rope-color / --goal-color (POV pigments + matplotlib).
COLORS = {
    "green": ("rgb<0.15,0.75,0.35>", "#26bf59"),
    "purple": ("rgb<0.45,0.39,1>", "#736aff"),
    "blue": ("rgb<0.2,0.45,0.95>", "#3373f2"),
    "red": ("rgb<0.9,0.2,0.2>", "#e63333"),
    "orange": ("rgb<1,0.55,0.1>", "#ff8c1a"),
    "yellow": ("rgb<0.95,0.85,0.15>", "#f2d926"),
    "cyan": ("rgb<0.15,0.8,0.85>", "#26ccd9"),
    "magenta": ("rgb<0.9,0.25,0.75>", "#e640bf"),
    "grey": ("rgb<0.55,0.55,0.58>", "#8c8c94"),
}

ROPE_COLOR = COLORS["green"][0]
GOAL_COLOR = COLORS["orange"][0]
DRIVEN_COLOR = "rgb<1,0.5,0.4>"
FIXED_COLOR = "rgb<0.75,0.75,0.75>"
POLE_COLOR = "rgb<0.3,0.3,0.32>"

MARKER_RADIUS = 0.5  # POV units
POLE_HEIGHT = 4.0
AXIS_LENGTH = 4.0
AXIS_RADIUS = 0.12


# ---------------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------------
def load_run(npz_path: Path):
    """Pull the pieces of an LQR run we need to replay and draw it."""
    data = np.load(npz_path, allow_pickle=False)

    cmd_x = np.asarray(data["cmd_x"], dtype=float)
    cmd_y = np.asarray(data["cmd_y"], dtype=float)
    commands = np.stack([cmd_x, cmd_y], axis=1)  # [N, 2] mm/step

    state = np.asarray(data["state"], dtype=float)  # [N, 4*n_nodes] mm
    n_nodes = state.shape[1] // 4

    goal_pos = np.asarray(data["goal_pos"], dtype=float)  # [n_nodes, 2] mm
    rmse = np.asarray(data["rmse"], dtype=float)  # [N] mm, AFTER each command
    control_dt = float(data["control_dt"])

    # state[k] is the measurement taken BEFORE command k is applied.
    logged_pos = state[:, : 2 * n_nodes].reshape(-1, n_nodes, 2)

    return {
        "commands": commands,
        "logged_pos": logged_pos,
        "goal_pos": goal_pos,
        "rmse": rmse,
        "control_dt": control_dt,
        "n_nodes": n_nodes,
    }


# ---------------------------------------------------------------------------
# PyElastica replay
# ---------------------------------------------------------------------------
def replay(run, max_steps=None, verbose=True):
    """Re-run the logged commands through PyElastica.

    Returns positions in mm, shape [N+1, n_nodes, 3]: the settled initial shape
    followed by the shape after each command.
    """
    n_nodes = run["n_nodes"]
    commands = run["commands"]
    if max_steps is not None:
        commands = commands[:max_steps]

    # Same environment const_lqr.py drove.
    env = RopeManipEnv(
        n_nodes=n_nodes,
        base_length=0.30,
        base_radius=0.003,
        youngs_modulus=1e6,
        damping=2.0,
        dt=5e-5,
        control_dt=run["control_dt"],
        seed=0,
    )
    env.reset()

    frames = [env.get_positions(mm=True)]
    for step, u in enumerate(commands):
        env.drive_step(u / MM)
        frames.append(env.get_positions(mm=True))
        if env.is_exploded():
            print(f"simulation exploded at step {step}; truncating replay")
            break
        if verbose and step % 50 == 0:
            print(f"  replay step {step:4d} / {len(commands)}")

    frames = np.asarray(frames)

    # Sanity check against the log: replayed frame k is the state the logger
    # recorded at the start of step k.
    n_cmp = min(len(frames), len(run["logged_pos"]))
    if n_cmp > 1:
        err = np.linalg.norm(
            frames[:n_cmp, :, :2] - run["logged_pos"][:n_cmp], axis=2
        )
        print(
            f"replay vs log: mean {err.mean():.3f} mm, max {err.max():.3f} mm "
            f"over {n_cmp} frames"
        )

    return frames


def positions_from_log(run):
    """Frames straight from the logged state (z pinned at the rope radius)."""
    pos2d = run["logged_pos"]
    z = np.full(pos2d.shape[:2] + (1,), 0.003 * MM)
    return np.concatenate([pos2d, z], axis=2)


# ---------------------------------------------------------------------------
# POV-Ray backend
# ---------------------------------------------------------------------------
def to_pov(p_mm):
    """sim (x, y, z) in mm -> POV (x, z, y) in POV units."""
    p = np.asarray(p_mm) / MM
    return SCALE * p[0], SCALE * p[2], SCALE * p[1]


def _sphere_sweep(points_pov, radius, color, transmit):
    s = "sphere_sweep\n{b_spline %d" % len(points_pov)
    for px, py, pz in points_pov:
        s += ",\n<%f,%f,%f>,%f" % (px, py, pz, radius)
    s += "\ntexture{ pigment{ color %s transmit %g } finish{ phong 1 } }\n}\n" % (
        color,
        transmit,
    )
    return s


def _fixed_pole(origin):
    x, y, z = origin
    top = y + POLE_HEIGHT
    knob_r = 1.8 * AXIS_RADIUS
    s = "cylinder{<%f,%f,%f>,<%f,%f,%f>,%f\n" % (
        x, 0.0, z, x, top, z, 0.9 * AXIS_RADIUS
    )
    s += "texture{ pigment{ color %s } finish{ phong 0.3 } }\n}\n" % POLE_COLOR
    s += "sphere{<%f,%f,%f>,%f\n" % (x, top, z, knob_r)
    s += "texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % POLE_COLOR
    return s


def _axis_gizmo(origin):
    x, y, z = origin
    axes = [
        ((AXIS_LENGTH, 0.0, 0.0), "rgb<0.9,0.15,0.15>"),
        ((0.0, AXIS_LENGTH, 0.0), "rgb<0.15,0.75,0.15>"),
        ((0.0, 0.0, AXIS_LENGTH), "rgb<0.15,0.35,0.95>"),
    ]
    parts = []
    for (dx, dy, dz), color in axes:
        tip = (x + dx, y + dy, z + dz)
        parts.append(
            "cylinder{<%f,%f,%f>,<%f,%f,%f>,%f\n"
            % (x, y, z, tip[0], tip[1], tip[2], AXIS_RADIUS)
        )
        parts.append(
            "texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % color
        )
        cone_tip = (
            tip[0] + 0.3 * dx / AXIS_LENGTH,
            tip[1] + 0.3 * dy / AXIS_LENGTH,
            tip[2] + 0.3 * dz / AXIS_LENGTH,
        )
        parts.append(
            "cone{<%f,%f,%f>,%f,<%f,%f,%f>,0\n"
            % (tip[0], tip[1], tip[2], 2.5 * AXIS_RADIUS, *cone_tip)
        )
        parts.append(
            "texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % color
        )
    return "".join(parts)


def set_colors(rope, goal):
    """Apply the --rope-color / --goal-color choices to the POV pigments."""
    global ROPE_COLOR, GOAL_COLOR
    ROPE_COLOR = COLORS[rope][0]
    GOAL_COLOR = COLORS[goal][0]


def fit_camera(frames_mm, goal_pos_mm, aspect, angle_deg=35.0, margin=1.12,
               view="3q"):
    """A fixed camera that frames the whole run.

    view="3q"  keeps the elevated 3/4 viewing direction of camera_position.inc's
               case 1; view="top" looks straight down at the table with sim +x
               to the right and sim +y up the screen. Either way the camera is
               re-centred and pulled back so every frame of this run fits.
    Returns a POV `camera{}` block, or None if the geometry is degenerate.
    """
    pts = [to_pov(p) for p in frames_mm.reshape(-1, 3)]
    if goal_pos_mm is not None:
        pts += [to_pov((gx, gy, 0.0)) for gx, gy in goal_pos_mm]
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return None

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    # The axis gizmo and end markers stick out past the rope itself.
    gizmo = AXIS_LENGTH + 2.0
    lo[[0, 2]] -= gizmo
    hi[[0, 2]] += gizmo
    hi[1] = max(hi[1], POLE_HEIGHT + 1.0)  # the pinned-end pole sticks up
    lo[1] = min(lo[1], 0.0)                # and the table is at y = 0
    look_at = 0.5 * (lo + hi)

    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
         for z in (lo[2], hi[2])],
        dtype=float,
    )

    if view == "top":
        # Straight down on the table. POV's default sky is parallel to the view
        # direction here, so pick sim +y (POV +z) as the screen-up axis.
        offset = np.array([0.0, 1.0, 0.0])
        sky = np.array([0.0, 0.0, 1.0])
        look_at = np.array([look_at[0], 0.0, look_at[2]])
    else:
        # Same view direction as case 1: <15,35,-55> -> <15,0,0>.
        offset = np.array([0.0, 35.0, -55.0])
        sky = np.array([0.0, 1.0, 0.0])
    offset = offset / np.linalg.norm(offset)

    tan_h = np.tan(np.radians(angle_deg) / 2.0)
    tan_v = tan_h / aspect

    def fits(dist):
        """True if every bbox corner falls inside the frustum at this distance."""
        pos = look_at + offset * dist
        f = look_at - pos
        f /= np.linalg.norm(f)
        r = np.cross(sky, f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)

        v = corners - pos
        depth = v @ f
        if np.any(depth <= 1e-6):
            return False
        return (np.max(np.abs(v @ r) / depth) <= tan_h
                and np.max(np.abs(v @ u) / depth) <= tan_v)

    # Smallest distance that still frames the whole run: the predicate is
    # monotone in dist, so bisect between "too close" and "certainly fits".
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    far = max(radius / tan_v, 1.0)
    while not fits(far) and far < 1e6:
        far *= 2.0
    near = 1e-3 * far
    for _ in range(60):
        mid = 0.5 * (near + far)
        if fits(mid):
            far = mid
        else:
            near = mid
    dist = far * margin

    pos = look_at + offset * dist
    sky_line = ""
    if view == "top":
        sky_line = "        sky      <%f,%f,%f>\n" % (sky[0], sky[1], sky[2])
    return (
        "// auto-framed camera for this run (overrides camera_position.inc)\n"
        "camera{ location <%f,%f,%f>\n"
        "%s"
        "        right    x*image_width/image_height\n"
        "        angle    %f\n"
        "        look_at  <%f,%f,%f>\n"
        "      }\n"
        "light_source{ <%f,%f,%f> color rgb<0.9,0.9,1>*0.25 }\n"
        % (pos[0], pos[1], pos[2], sky_line, angle_deg,
           look_at[0], look_at[1], look_at[2],
           pos[0], pos[1] + 20.0, pos[2])
    )


def write_pov_frame(idx, positions_mm, goal_pos_mm, radius_m, work_dir: Path,
                    camera_block=None):
    """Write moving_arm%04d.inc/.pov for one frame."""
    inc_path = work_dir / ("moving_arm%04d.inc" % idx)
    rope_pov = [to_pov(p) for p in positions_mm]
    rope_radius = SCALE * radius_m

    body = _sphere_sweep(rope_pov, rope_radius, ROPE_COLOR, 0.1)

    # Goal shape: a thin translucent ribbon lying on the table.
    if goal_pos_mm is not None:
        goal_pov = [to_pov((gx, gy, 0.001 * MM)) for gx, gy in goal_pos_mm]
        body += _sphere_sweep(goal_pov, 0.45 * rope_radius, GOAL_COLOR, 0.55)

    dx, dy, dz = to_pov(positions_mm[0])
    body += "sphere\n{<%f,%f,%f>,%f\n" % (dx, dy, dz, 1.5 * rope_radius)
    body += (
        "texture{ pigment{ color %s transmit 0.1 } finish{ phong 1 } }\n}\n"
        % DRIVEN_COLOR
    )

    fx, fy, fz = to_pov(positions_mm[-1])
    body += "sphere\n{<%f,%f,%f>,%f\n" % (fx, fy, fz, MARKER_RADIUS)
    body += (
        "texture{ pigment{ color %s transmit 0.1 } finish{ phong 1 } }\n}\n"
        % FIXED_COLOR
    )
    body += _fixed_pole((fx, fy, fz))
    body += _axis_gizmo((dx, dy, dz))

    inc_path.write_text(body)

    pov_path = work_dir / ("moving_arm%04d.pov" % idx)
    scene = '#include "%s"\n' % CAMERA_INC.as_posix()
    if camera_block:  # a later camera{} wins in POV-Ray
        scene += camera_block
    scene += '#include "%s"\n' % inc_path.name
    pov_path.write_text(scene)
    return pov_path


def _run_povray(job):
    pov_path, width, height, work_dir = job
    png_path = Path(pov_path).with_suffix(".png")
    cmd = [
        "povray",
        "-W%d" % width,
        "-H%d" % height,
        "Quality=11",
        "Antialias=on",
        "Output_File_Name=%s" % png_path.name,
        "Verbose=Off",
        Path(pov_path).name,
    ]
    proc = subprocess.run(
        cmd, cwd=work_dir, capture_output=True, text=True
    )
    if proc.returncode != 0 or not png_path.exists():
        return pov_path, proc.stderr[-2000:]
    return pov_path, None


def render_povray(frames_mm, goal_pos, radius_m, work_dir: Path, width, height,
                  jobs, auto_camera=True, view="3q"):
    if shutil.which("povray") is None:
        raise RuntimeError(
            "povray not found on PATH; use --backend matplotlib instead"
        )
    if not CAMERA_INC.exists():
        raise FileNotFoundError(f"camera include not found: {CAMERA_INC}")

    camera_block = None
    if auto_camera:
        camera_block = fit_camera(frames_mm, goal_pos, aspect=width / height,
                                  view=view)
    elif view == "top":
        print("--scene-camera overrides --view top; using the scene camera")

    pov_paths = [
        write_pov_frame(i, pos, goal_pos, radius_m, work_dir, camera_block)
        for i, pos in enumerate(frames_mm)
    ]

    jobs_list = [(str(p), width, height, str(work_dir)) for p in pov_paths]
    print(f"rendering {len(jobs_list)} frames with povray ({jobs} workers)...")
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for pov_path, err in pool.map(_run_povray, jobs_list):
            done += 1
            if err:
                raise RuntimeError(f"povray failed on {pov_path}:\n{err}")
            if done % 25 == 0 or done == len(jobs_list):
                print(f"  {done} / {len(jobs_list)} frames")

    return [Path(p).with_suffix(".png") for p in pov_paths]


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------
def render_matplotlib(frames_mm, goal_pos, work_dir: Path, width, height, dpi=100,
                      rope_color=COLORS["green"][1],
                      goal_color=COLORS["orange"][1]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_xy = frames_mm[:, :, :2].reshape(-1, 2)
    pad = 20.0
    xmin = min(all_xy[:, 0].min(), goal_pos[:, 0].min()) - pad
    xmax = max(all_xy[:, 0].max(), goal_pos[:, 0].max()) + pad
    ymin = min(all_xy[:, 1].min(), goal_pos[:, 1].min()) - pad
    ymax = max(all_xy[:, 1].max(), goal_pos[:, 1].max()) + pad

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    paths = []
    for i, pos in enumerate(frames_mm):
        ax.clear()
        ax.plot(goal_pos[:, 0], goal_pos[:, 1], "--", color=goal_color,
                lw=2, label="goal")
        trail = frames_mm[: i + 1, 0, :2]
        ax.plot(trail[:, 0], trail[:, 1], "-", color="0.7", lw=1,
                label="driven-end path")
        ax.plot(pos[:, 0], pos[:, 1], "-", color=rope_color, lw=3, label="rope")
        ax.plot(pos[0, 0], pos[0, 1], "o", color="tab:red", ms=9,
                label="driven end")
        ax.plot(pos[-1, 0], pos[-1, 1], "s", color="0.35", ms=8,
                label="fixed end")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        fig.tight_layout()
        out = work_dir / ("frame%04d.png" % i)
        fig.savefig(out)
        paths.append(out)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Overlay + GIF assembly
# ---------------------------------------------------------------------------
def annotate(png_paths, labels):
    """Burn a text label into the top-left corner of each frame."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    for path, text in zip(png_paths, labels):
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        box = draw.textbbox((0, 0), text, font=font)
        draw.rectangle(
            (8, 8, 8 + box[2] + 16, 8 + box[3] + 12), fill=(255, 255, 255)
        )
        draw.text((16, 12), text, fill=(20, 20, 20), font=font)
        img.save(path)


def make_gif(png_paths, out_path: Path, fps, width=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is not None:
        pattern = png_paths[0].parent / (
            "moving_arm%04d.png"
            if png_paths[0].name.startswith("moving_arm")
            else "frame%04d.png"
        )
        scale = "scale=%d:-1:flags=lanczos," % width if width else ""
        vf = (
            "fps=%d,%ssplit[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3" % (fps, scale)
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(pattern),
            "-vf", vf,
            "-loop", "0",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        print("ffmpeg failed, falling back to Pillow:\n" + proc.stderr[-1000:])

    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in png_paths]
    if width:
        h = int(round(imgs[0].height * width / imgs[0].width))
        imgs = [im.resize((width, h), Image.LANCZOS) for im in imgs]
    imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in imgs]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        disposal=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", default="lqr_runs/lqr_run_C.npz")
    parser.add_argument("--out", default=None,
                        help="output GIF (default: <npz stem>.gif next to the npz)")
    parser.add_argument("--backend", choices=("povray", "matplotlib"),
                        default="povray")
    parser.add_argument("--source", choices=("replay", "log"), default="replay",
                        help="replay the commands in PyElastica (default) or "
                             "draw the logged states directly")
    parser.add_argument("--stride", type=int, default=2,
                        help="keep every Nth control step (default 2)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="only replay/draw the first N control steps")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gif-width", type=int, default=720,
                        help="downscale width of the GIF (0 = keep render size)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="parallel povray processes")
    parser.add_argument("--work-dir", default=None,
                        help="where frames are written (default: <out>_frames)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep the per-frame PNG/POV files")
    parser.add_argument("--no-goal", action="store_true",
                        help="do not draw the goal shape")
    parser.add_argument("--scene-camera", action="store_true",
                        help="use camera_position.inc's camera verbatim instead "
                             "of auto-framing this run")
    parser.add_argument("--view", choices=("3q", "top"), default="top",
                        help="povray viewpoint: elevated 3/4 view (default) or "
                             "'top' looking straight down at the table "
                             "(the matplotlib backend is always top-down)")
    parser.add_argument("--rope-color", choices=sorted(COLORS), default="green")
    parser.add_argument("--goal-color", choices=sorted(COLORS), default="orange")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = HERE / npz_path
    if not npz_path.exists():
        parser.error(f"npz not found: {npz_path}")

    out_path = Path(args.out) if args.out else npz_path.with_suffix(".gif")
    if not out_path.is_absolute():
        out_path = HERE / out_path

    set_colors(args.rope_color, args.goal_color)

    run = load_run(npz_path)
    n_steps = len(run["commands"])
    print(f"loaded {npz_path.name}: {n_steps} control steps, "
          f"{run['n_nodes']} nodes, control_dt = {run['control_dt']} s")
    print(f"RMSE: {run['rmse'][0]:.2f} mm -> {run['rmse'][-1]:.2f} mm")

    # ---- rope shapes ----------------------------------------------------
    if args.source == "replay":
        frames_mm = replay(run, max_steps=args.max_steps)
    else:
        frames_mm = positions_from_log(run)
        if args.max_steps:
            frames_mm = frames_mm[: args.max_steps + 1]

    idx = np.arange(0, len(frames_mm), max(1, args.stride))
    if idx[-1] != len(frames_mm) - 1:
        idx = np.append(idx, len(frames_mm) - 1)  # always show the final shape
    frames_mm = frames_mm[idx]
    print(f"{len(frames_mm)} frames after stride {args.stride}")

    goal_pos = None if args.no_goal else run["goal_pos"]

    # Frame k of the full trajectory is the shape after k commands; the RMSE
    # logged for command k-1 describes exactly that shape.
    rmse_full = np.concatenate([[run["rmse"][0]], run["rmse"]])
    labels = []
    for k in idx:
        k = int(k)
        rm = rmse_full[min(k, len(rmse_full) - 1)]
        labels.append(
            "step %4d   t = %5.2f s   RMSE = %6.2f mm"
            % (k, k * run["control_dt"], rm)
        )

    # ---- render ---------------------------------------------------------
    work_dir = Path(args.work_dir) if args.work_dir else out_path.with_name(
        out_path.stem + "_frames"
    )
    if not work_dir.is_absolute():
        work_dir = HERE / work_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    if args.backend == "povray":
        png_paths = render_povray(
            frames_mm,
            goal_pos if goal_pos is not None else None,
            radius_m=0.003,
            work_dir=work_dir,
            width=args.width,
            height=args.height,
            jobs=args.jobs,
            auto_camera=not args.scene_camera,
            view=args.view,
        )
    else:
        png_paths = render_matplotlib(
            frames_mm,
            run["goal_pos"],
            work_dir=work_dir,
            width=args.width,
            height=args.height,
            rope_color=COLORS[args.rope_color][1],
            goal_color=COLORS[args.goal_color][1],
        )

    annotate(png_paths, labels)

    make_gif(png_paths, out_path, args.fps,
             width=args.gif_width if args.gif_width > 0 else None)

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({len(png_paths)} frames, {args.fps} fps, "
          f"{size_mb:.1f} MB)")

    if not args.keep_frames:
        shutil.rmtree(work_dir)
    else:
        print(f"frames kept in {work_dir}")


if __name__ == "__main__":
    main()
