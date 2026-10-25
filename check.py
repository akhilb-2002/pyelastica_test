"""
Collect Koopman time-series rope data from the PyElastica manipulation env.

PyElastica port of the SOFA `collect ... koopman_timeseries` script. Same cases,
motion profiles, reach-limiting, CSV schema, and CLI, but driven by
`rope_manip_env.RopeManipEnv`:

  * Node 0 is kinematically driven; the control input u_k = Node 0's per-step
    displacement (cmd_dx, cmd_dy). The command logged on step k is exactly the
    increment applied on step k -- steps where Node 0 is held carry u = 0, and the
    rope's continued motion there is autonomous dynamics (what A must explain).
  * The last node is fixed. Interior nodes are free on a frictional table.

Everything works in millimetres in the CSV (positions, velocities, commands), to
match the SOFA output. The simulation itself runs in SI; conversion is handled at
the env boundary (mm=True readouts / MM factor).

MAGNITUDE CONVENTION
--------------------
Cases 1-9 (legacy) sample a TOTAL displacement per action, which the profile then
divides across T steps. Cases 10-16 sample a PER-STEP displacement (mm/step)
directly and hold it for T steps -- this decouples speed from hold length so T can
be varied independently. A case declares which convention it uses via `per_step`.

Usage:
    python collect_koopman_pyelastica.py all --num-trajectories 100 --seed 0
    python collect_koopman_pyelastica.py 10 11 12 13 14 15 --num-trajectories 300
    python collect_koopman_pyelastica.py new --num-trajectories 300   # cases 10-16
"""

import sys
import time
import csv
import argparse
import functools
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rope_manip_env import RopeManipEnv, MM


# ─────────────────────────────────────────────
# Control bound
#
# u_max is the per-step displacement ceiling used when collecting. Constrain the
# MPC/MPPI solver to ~0.75 * U_MAX_MM so the controller never extrapolates off the
# edge of the training distribution.
#
# At control_dt = 0.01 s: 2.0 mm/step = 200 mm/s. The rest segment length is
# base_length / (n_nodes - 1) ~ 10.3 mm, so 2 mm/step is ~0.2 segment/step.
# ─────────────────────────────────────────────
U_MAX_MM = 2.0


# ─────────────────────────────────────────────
# Motion profiles
#
# A profile turns one action into a per-step displacement increment of Node 0.
# All magnitudes are in millimetres.
#
#   legacy profiles (cases 1-9) : `mag` is the TOTAL displacement of the action
#   step profiles  (cases 10-16): `mag` is the PER-STEP displacement (mm/step)
# ─────────────────────────────────────────────
def _ramp(mag, angle, T):
    """Spread the total displacement evenly across the hold: smooth, MPC-like."""
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)]) / T
    return lambda k: step


def _teleport(mag, angle, T):
    """All displacement on the first step, then hold (impulse response)."""
    jump = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: jump if k == 0 else np.zeros(2)


def _ramp_step(mag, angle, T):
    """Constant per-step displacement `mag` (mm/step) held for T steps.

    Decouples speed from hold length: pair a small T with many actions to get
    frequent direction changes at a fixed speed, which is what a shape-tracing
    controller actually emits.
    """
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: step


def _drive_hold(mag, angle, T, T_drive=20):
    """Drive at `mag` mm/step for T_drive steps, then hold (u = 0) for the rest.

    Unlike case 8 (which settles BEFORE recording, so its transitions are ~identity
    and push the spectrum toward the unit circle), this records relaxation from
    energetic states -- the decay data A needs to be learned as contractive.
    """
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: step if k < T_drive else np.zeros(2)


def _pname(profile):
    """Profile name, unwrapping functools.partial."""
    p = profile
    while isinstance(p, functools.partial):
        p = p.func
    return p.__name__.lstrip("_")


def _u(lo, hi):
    """Uniform sampler factory (mm or mm/step depending on the case)."""
    return lambda: float(np.random.uniform(lo, hi))


CASES = {
    # ======================= LEGACY: total-displacement cases =================
    # ---- impulse response, wide magnitude coverage --------------------------
    1: dict(name="fixed_10mm",      magnitude=lambda: 10.0,
            profile=_teleport, T=20, n_actions=5),
    2: dict(name="fixed_50mm",      magnitude=lambda: 50.0,
            profile=_teleport, T=20, n_actions=5),
    3: dict(name="random_1_10mm",   magnitude=_u(1, 10),
            profile=_teleport, T=20, n_actions=5),
    4: dict(name="random_10_80mm",  magnitude=_u(10, 80),
            profile=_teleport, T=20, n_actions=5),
    # ---- smooth motion: matches how MPC will actually drive the rope --------
    5: dict(name="ramp_1_10mm",     magnitude=_u(1, 10),
            profile=_ramp, T=20, n_actions=5),
    6: dict(name="ramp_10_80mm",    magnitude=_u(10, 80),
            profile=_ramp, T=20, n_actions=5),
    # ---- small-signal: the regime a linear model should nail ---------------
    7: dict(name="micro_0.1_1mm",   magnitude=_u(0.1, 1.0),
            profile=_ramp, T=10, n_actions=10),
    # ---- autonomous dynamics: u = 0, isolates A from B ---------------------
    8: dict(name="free_relaxation", magnitude=lambda: 0.0,
            profile=_teleport, T=100, n_actions=2),
    # ---- long-horizon held-out set for rollout evaluation ------------------
    9: dict(name="long_mixed",      magnitude=_u(1, 60),
            profile=_ramp, T=20, n_actions=40),

    # ======================= PER-STEP displacement cases ======================
    # `magnitude` is mm/STEP. Total action displacement = magnitude * T.
    # ---- speed sweep at the nominal hold length (T = 20) --------------------
    10: dict(name="micro_step",      magnitude=_u(0.02, 0.2),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    11: dict(name="slow_step",       magnitude=_u(0.2, 0.8),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    12: dict(name="fast_step",       magnitude=_u(0.8, U_MAX_MM),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    # ---- frequent direction changes: closest match to a tracing controller --
    13: dict(name="turn_short",      magnitude=_u(0.2, 1.5),
             profile=_ramp_step, T=5, n_actions=40, per_step=True),
    14: dict(name="turn_v_short",    magnitude=_u(0.2, 1.5),
             profile=_ramp_step, T=2, n_actions=100, per_step=True),
    # ---- relaxation from MOVING states: the decay data A needs -------------
    15: dict(name="drive_hold",      magnitude=_u(0.5, 1.5),
             profile=functools.partial(_drive_hold, T_drive=20),
             T=60, n_actions=5, per_step=True, T_drive=20),
    # ---- long-horizon held-out set for rollout evaluation ------------------
    16: dict(name="long_mixed_step", magnitude=_u(0.1, 1.5),
             profile=_ramp_step, T=10, n_actions=80, per_step=True),
}

# Cases 10-16 are the recommended training/eval set for the shape-tracing MPC.
NEW_CASES = [10, 11, 12, 13, 14, 15, 16]

_case_help = ", ".join(f"{k}={v['name']}" for k, v in sorted(CASES.items()))
parser = argparse.ArgumentParser(description="Collect Koopman time-series rope data (PyElastica).")
parser.add_argument("cases", nargs="+",
                    help=f"Case number(s), 'all', or 'new' (={NEW_CASES}). Available: {_case_help}")
parser.add_argument("--num-trajectories", type=int, default=100,
                    help="Number of independent multi-action rollouts per case.")
parser.add_argument("--seed", type=int, default=None, help="RNG seed (recorded in description.txt).")
args = parser.parse_args()

_sel = args.cases[0].lower() if len(args.cases) == 1 else None
if _sel == "all":
    SELECTED_CASES = sorted(CASES.keys())
elif _sel == "new":
    SELECTED_CASES = list(NEW_CASES)
else:
    try:
        SELECTED_CASES = [int(c) for c in args.cases]
    except ValueError:
        parser.error(f"Cases must be integers, 'all', or 'new'. Received: {args.cases}")
    unknown = [c for c in SELECTED_CASES if c not in CASES]
    if unknown:
        parser.error(f"Unknown case(s) {unknown}. Available: {sorted(CASES.keys())}")

if args.seed is not None:
    np.random.seed(args.seed)

NUM_TRAJECTORIES = args.num_trajectories
print("Cases to collect: " + ", ".join(f"{c}={CASES[c]['name']}" for c in SELECTED_CASES))

# ─────────────────────────────────────────────
# Collection parameters
# ─────────────────────────────────────────────
MAX_REACH_FRAC = 0.9       # reject action targets beyond this fraction of the rope reach
MAX_TARGET_RESAMPLES = 25
RANDOMIZE_SCALE = 0.1      # shape-randomization force std (N); see env.randomize_shape

OUT_ROOT = Path(__file__).resolve().parent / "csv_timeseries_pyelastica"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Rope physics parameters (metres internally; see below for the mm summary).
print("Initializing PyElastica RopeManipEnv ...")
env = RopeManipEnv(
    n_nodes=30,
    base_length=0.30,
    base_radius=0.003,
    youngs_modulus=1e6,
    damping=2.0,
    dt=5e-5,
    control_dt=0.01,
    seed=args.seed,
)

env.reset()
num_points = env.n_nodes
DT = env.control_dt                      # one recorded timestep, in seconds
print(f"Rope has {num_points} nodes. Recorded timestep dt = {DT:.4f} s.")

# Rope reach and the fixed-node position, measured on the settled rest config.
FIXED_NODE_POS = env.fixed_position(mm=True)[:2].copy()
ROPE_REACH = env.rope_reach(mm=True)
REST_SEGMENT = env.base_length * MM / (env.n_nodes - 1)
print(f"Rope reach (rest arc length): {ROPE_REACH:.2f} mm | "
      f"target limit: {MAX_REACH_FRAC * ROPE_REACH:.2f} mm from fixed node")
print(f"u_max = {U_MAX_MM:.2f} mm/step = {U_MAX_MM / DT:.0f} mm/s "
      f"({U_MAX_MM / REST_SEGMENT:.2f} rest-segments/step)")

# ─────────────────────────────────────────────
# CSV schema  (identical to the SOFA long format)
# ─────────────────────────────────────────────
headers = ["trajectory_id", "action_idx", "step_idx"]
for i in range(num_points):
    headers += [f"pos_x_{i}", f"pos_y_{i}"]
for i in range(num_points):
    headers += [f"vel_x_{i}", f"vel_y_{i}", f"ang_vel_z_{i}"]
for i in range(num_points):
    headers += [f"curvature_{i}"]
for i in range(num_points - 1):
    headers += [f"edge_dx_{i}", f"edge_dy_{i}", f"edge_length_{i}"]
headers += ["cmd_dx", "cmd_dy", "target_x", "target_y"]


def compute_curvature(positions, node_idx):
    N = len(positions)
    if node_idx == 0 or node_idx == N - 1:
        return 0.0
    v1 = (positions[node_idx] - positions[node_idx - 1])[:2]
    v2 = (positions[node_idx + 1] - positions[node_idx])[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def build_row(traj_id, action_idx, step_idx, positions, velocities, ang_vel_z, cmd, target):
    """positions/velocities in mm, mm/s; cmd/target in mm."""
    row = [traj_id, action_idx, step_idx]
    for p in positions:
        row += [float(p[0]), float(p[1])]
    for v, wz in zip(velocities, ang_vel_z):
        row += [float(v[0]), float(v[1]), float(wz)]
    for i in range(num_points):
        row.append(compute_curvature(positions, i))
    for i in range(num_points - 1):
        edge = positions[i + 1] - positions[i]
        row += [float(edge[0]), float(edge[1]), float(np.linalg.norm(edge[:2]))]
    row += [float(cmd[0]), float(cmd[1]), float(target[0]), float(target[1])]
    assert len(row) == len(headers), f"Row {len(row)} != headers {len(headers)}"
    return row


def action_total_mm(case, magnitude):
    """Total in-plane displacement of one action, in mm.

    For legacy cases `magnitude` IS the total. For per-step cases it is mm/step,
    so the total is magnitude * (number of DRIVEN steps) -- which is T_drive for
    drive-hold profiles and T otherwise.
    """
    if not case.get("per_step", False):
        return magnitude
    return magnitude * case.get("T_drive", case["T"])


def sample_action(case, current_target_mm):
    """Sample (magnitude, angle) whose action endpoint stays within the rope's reach."""
    magnitude = angle = None
    for _ in range(MAX_TARGET_RESAMPLES):
        magnitude = case["magnitude"]()
        angle = np.random.uniform(0.0, 2 * np.pi)
        total = action_total_mm(case, magnitude)
        endpoint = current_target_mm[:2] + total * np.array([np.cos(angle), np.sin(angle)])
        if np.linalg.norm(endpoint - FIXED_NODE_POS) <= MAX_REACH_FRAC * ROPE_REACH:
            break
    return magnitude, angle


def write_description(path, case_id, case, csv_name):
    per_step = case.get("per_step", False)
    unit = "mm/step" if per_step else "mm (total per action)"
    with open(path, "w") as f:
        f.write(f"Case {case_id}: {case['name']}\n")
        f.write(f"Motion profile: {_pname(case['profile'])}\n")
        f.write(f"Magnitude convention: {unit}\n\n")
        f.write("Collection parameters:\n")
        f.write(f"  num_trajectories       : {NUM_TRAJECTORIES}\n")
        f.write(f"  actions_per_trajectory : {case['n_actions']}\n")
        f.write(f"  timesteps_per_action   : {case['T']}\n")
        if per_step:
            f.write(f"  driven_steps_per_action: {case.get('T_drive', case['T'])}\n")
            f.write(f"  u_max_mm_per_step      : {U_MAX_MM}\n")
        f.write(f"  koopman_step_dt        : {DT:.4f} s\n")
        f.write(f"  rows_per_trajectory    : {case['n_actions'] * case['T']}\n")
        f.write(f"  rope_reach_mm          : {ROPE_REACH:.4f}\n")
        f.write(f"  max_reach_frac         : {MAX_REACH_FRAC}\n")
        f.write(f"  seed                   : {args.seed}\n\n")
        f.write("Rope physics parameters (PyElastica, SI internally):\n")
        f.write(f"  num_rope_points    : {num_points}\n")
        f.write(f"  base_length        : {env.base_length * MM:.1f} mm\n")
        f.write(f"  radius             : {env.base_radius * MM:.1f} mm\n")
        f.write(f"  youngs_modulus     : {env.youngs_modulus:.3e} (SI)\n")
        f.write(f"  damping            : {env.damping}\n")
        f.write(f"  sim_dt / substeps  : {env.dt:.1e} s / {env.substeps} per step\n\n")
        f.write(f"CSV: {csv_name} (long format, one row per (trajectory_id, action_idx, step_idx)):\n")
        f.write("  trajectory_id, action_idx, step_idx : rollout / action / within-action step\n")
        f.write("  pos_x_i, pos_y_i (i=0..N-1)          : rope node positions (mm)\n")
        f.write("  vel_x_i, vel_y_i, ang_vel_z_i        : node linear/angular velocities (mm/s, rad/s)\n")
        f.write("  curvature_i                          : local turning angle at each node (rad)\n")
        f.write("  edge_dx_i, edge_dy_i, edge_length_i  : edge vectors/lengths (mm)\n")
        f.write("  cmd_dx, cmd_dy                       : Node-0 displacement applied ON THIS STEP (mm).\n")
        f.write("                                         Zero on held steps; rope motion there is autonomous.\n")
        f.write("  target_x, target_y                   : commanded absolute Node-0 position this step (mm).\n")


def collect_case(case_id):
    case = CASES[case_id]
    T, n_actions = case["T"], case["n_actions"]

    dest_dir = OUT_ROOT / f"case_{case_id}_{case['name']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "collected_trajectories_koopman_timeseries.csv"
    write_description(dest_dir / "description.txt", case_id, case, csv_path.name)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)

    print(f"\n=== Case {case_id}: {case['name']} "
          f"(profile={_pname(case['profile'])}, "
          f"{n_actions} actions x {T} steps = {n_actions * T} rows/traj) ===")
    print(f"    -> {csv_path}")

    case_start = time.time()
    traj_idx = 1
    pbar = tqdm(total=NUM_TRAJECTORIES, desc=f"case {case_id} ({case['name']})")

    while traj_idx <= NUM_TRAJECTORIES:
        # 1. fresh straight rope, settled
        env.reset()
        # 2. randomize the starting shape for state-space coverage
        env.randomize_shape(force_scale=RANDOMIZE_SCALE)
        if env.is_exploded():
            continue

        # driven-node target (absolute, mm) starts at the current driven position
        target_node_pos = env.driven_position(mm=True)[:2].copy()

        traj_rows = []
        exploded = False
        for action_idx in range(n_actions):
            magnitude, angle = sample_action(case, target_node_pos)
            step_command = case["profile"](magnitude, angle, T)  # returns mm increments

            for step_idx in range(T):
                cmd_mm = np.asarray(step_command(step_idx), dtype=float)   # (dx, dy) mm
                target_node_pos = target_node_pos + cmd_mm

                # drive the node (env works in metres)
                env.drive_step(cmd_mm / MM)

                if env.is_exploded():
                    exploded = True
                    break

                positions = env.get_positions(mm=True)          # (N, 3) mm
                velocities = env.get_velocities(mm=True)         # (N, 3) mm/s
                ang_vel_z = env.get_node_angular_velocity_z()    # (N,) rad/s
                traj_rows.append(
                    build_row(traj_idx, action_idx, step_idx,
                              positions, velocities, ang_vel_z,
                              cmd_mm, target_node_pos))
            if exploded:
                break

        if exploded:
            continue  # retry this rollout

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(traj_rows)
        pbar.update(1)
        traj_idx += 1

    pbar.close()
    elapsed = time.time() - case_start
    print(f"    case {case_id} done in {elapsed / 60:.2f} min "
          f"({NUM_TRAJECTORIES * n_actions * T} rows)")
    return elapsed


start_time = time.time()
timings = {c: collect_case(c) for c in SELECTED_CASES}
total = time.time() - start_time
print("\n─── Summary ───")
for c, el in timings.items():
    rows = NUM_TRAJECTORIES * CASES[c]["n_actions"] * CASES[c]["T"]
    print(f"  case {c:>2} {CASES[c]['name']:<16} {el / 60:6.2f} min  ({rows} rows)")
print(f"Data collection completed in {total / 60:.2f} minutes.")