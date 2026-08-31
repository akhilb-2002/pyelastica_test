"""
Collect Koopman time-series rope data from the PyElastica manipulation env.

Node 0 is kinematically driven; the control input u_k is its per-step displacement
(cmd_dx, cmd_dy). The last node is fixed. Interior nodes are free on a frictional
table. Everything in the CSV is millimetres; the sim runs in SI internally.

CASE DESIGN
-----------
The reachable set of node 0 is a disc of radius L = 300 mm centred on the fixed
node. Distance from that centre determines how curled the rope must be: near the
rim it is nearly straight, near the centre it must coil. So covering the DISC is
the same thing as covering rope configurations, and that is what these cases do.

Each trajectory samples waypoints uniformly BY AREA in the annulus and drives
node 0 toward them across several ramped actions. This is the key difference from
the earlier random-angle design, which redrew a fresh bearing every action and so
random-walked (net travel ~ sqrt(n) x action size) instead of covering ground.

All motion is ramped -- no teleports. Impulsive inputs were measured to give ~50x
higher MSE than ramps at matched amplitude, and the MPC only ever emits smooth
per-step commands, so impulse data actively pollutes the operator.

Magnitude convention: `magnitude` is the TOTAL action displacement in mm, spread
evenly over the driven steps by _ramp. Per-step motion is magnitude / T_drive,
which is the quantity that must stay under U_MAX_MM.

Usage:
    python koopman_data_gen.py all --num-trajectories 100 --seed 0
    python koopman_data_gen.py 1 2 --num-trajectories 20
"""

import time
import csv
import argparse
import functools
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rope_manip_env import RopeManipEnv, MM


# ─────────────────────────────────────────────
# Motion profiles -- ramped only
# ─────────────────────────────────────────────
U_MAX_MM = 2.0        # per-step ceiling (mm/step). 2.0 mm/step = 200 mm/s at
                      # control_dt = 0.01, ~0.19 of the 10.345 mm rest segment.
                      # Constrain the MPC to ~0.75 * U_MAX_MM so it never
                      # extrapolates off the edge of the training set.

def _ramp(mag, angle, T):
    """Spread the total displacement `mag` evenly over T steps."""
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)]) / T
    return lambda k: step

def _ramp_hold(mag, angle, T, T_drive=20):
    """Ramp `mag` over T_drive steps, then hold (u = 0) for the rest of T.

    The held steps are autonomous dynamics -- the rope keeps moving with no
    command. This is the decay data A needs to be learned as contractive.
    """
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)]) / T_drive
    return lambda k: step if k < T_drive else np.zeros(2)


def _pname(profile):
    p = profile
    while isinstance(p, functools.partial):
        p = p.func
    return p.__name__.lstrip("_")


# ─────────────────────────────────────────────
# Cases
#
# `radius_frac` is the annulus (as a fraction of rope length) waypoints are drawn
# from: 0.95 is nearly straight, 0.25 is a deep coil. Cases differ mainly in
# action size (hence per-step speed) and in which part of the disc they visit.
# ─────────────────────────────────────────────
CASES = {
    # ---- the workhorse: full-disc coverage at nominal speed ----------------
    1: dict(name="disc_nominal",  magnitude=lambda: float(np.random.uniform(5.0, 20.0)),
            profile=_ramp, T=20, n_actions=40,          # 0.25 - 1.0 mm/step
            waypoints=dict(radius_frac=(0.30, 0.95))),
    # ---- fine motion: the precision regime every trace ENDS in -------------
    2: dict(name="disc_fine",     magnitude=lambda: float(np.random.uniform(0.5, 5.0)),
            profile=_ramp, T=20, n_actions=40,          # 0.025 - 0.25 mm/step
            waypoints=dict(radius_frac=(0.30, 0.95))),
    # ---- upper speed band: shorter holds, larger steps --------------------
    3: dict(name="disc_fast",     magnitude=lambda: float(np.random.uniform(10.0, 20.0)),
            profile=_ramp, T=10, n_actions=40,          # 1.0 - 2.0 mm/step
            waypoints=dict(radius_frac=(0.30, 0.95))),
    # ---- deep interior + rim, alternating: waypoints near the centre force a
    #      tight coil, and swinging back out forces the UNCURL that letter
    #      tracing needs. Outward-only data leaves the operator one-directional.
    4: dict(name="radial_sweep",  magnitude=lambda: float(np.random.uniform(5.0, 20.0)),
            profile=_ramp, T=20, n_actions=50,
            waypoints=dict(radius_frac=(0.25, 0.95), alternate=True)),
    # ---- relaxation from ENERGETIC states: drive, then let it ring down ----
    5: dict(name="disc_hold",     magnitude=lambda: float(np.random.uniform(5.0, 20.0)),
            profile=functools.partial(_ramp_hold, T_drive=20),
            T=80, n_actions=10, T_drive=20,
            waypoints=dict(radius_frac=(0.30, 0.95))),

    # ---- HELD OUT: long rollouts for open-loop drift evaluation ------------
    6: dict(name="disc_long",     magnitude=lambda: float(np.random.uniform(1.0, 20.0)),
            profile=_ramp, T=10, n_actions=120,
            waypoints=dict(radius_frac=(0.25, 0.95), alternate=True)),
}

TRAIN_CASES = [1, 2, 3, 4, 5]
HELDOUT_CASES = [6]


_case_help = ", ".join(f"{k}={v['name']}" for k, v in sorted(CASES.items()))
parser = argparse.ArgumentParser(description="Collect Koopman time-series rope data (PyElastica).")
parser.add_argument("cases", nargs="+",
                    help=f"Case number(s), 'all', or 'train' (={TRAIN_CASES}). "
                         f"Available: {_case_help}")
parser.add_argument("--num-trajectories", type=int, default=100)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

_sel = args.cases[0].lower() if len(args.cases) == 1 else None
if _sel == "all":
    SELECTED_CASES = sorted(CASES.keys())
elif _sel == "train":
    SELECTED_CASES = list(TRAIN_CASES)
else:
    try:
        SELECTED_CASES = [int(c) for c in args.cases]
    except ValueError:
        parser.error(f"Cases must be integers, 'all', or 'train'. Received: {args.cases}")
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
MAX_REACH_FRAC = 0.95      # hard ceiling on |node0 - fixed| / rope_reach
WAYPOINT_TOL = 8.0         # mm; within this, the waypoint counts as reached
WAYPOINT_JITTER = 0.25     # rad; spread around the exact bearing to the waypoint
RANDOMIZE_SCALE = 0.1      # shape-randomization force std (N)

OUT_ROOT = Path(__file__).resolve().parent / "csv_timeseries_pyelastica_simple"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Initializing PyElastica RopeManipEnv ...")

env = RopeManipEnv(
    n_nodes=30, base_length=0.30, base_radius=0.003,
    youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01, seed=args.seed,
)

env.reset()
num_points = env.n_nodes
DT = env.control_dt
print(f"Rope has {num_points} nodes. Recorded timestep dt = {DT:.4f} s.")

FIXED_NODE_POS = env.fixed_position(mm=True)[:2].copy()
ROPE_REACH = env.rope_reach(mm=True)
print(f"Rope reach: {ROPE_REACH:.2f} mm | fixed node at {FIXED_NODE_POS} mm")
print(f"Waypoint disc spans {0.25 * ROPE_REACH:.1f} - {0.95 * ROPE_REACH:.1f} mm from centre")


def sample_waypoint(r_lo_frac, r_hi_frac, want_inner=None):
    """A point in the reachable annulus, uniform BY AREA.

    r = R * sqrt(U), not r = R * U -- sampling the radius uniformly would
    concentrate points near the centre, over-representing deep coils and
    starving the mid-radius band where most letter shapes actually live.

    want_inner True/False forces the inner or outer third of the annulus; used
    by the alternating radial case to guarantee curl <-> uncurl transitions.
    """
    lo, hi = r_lo_frac, r_hi_frac
    if want_inner is True:
        hi = lo + 0.35 * (hi - lo)
    elif want_inner is False:
        lo = hi - 0.35 * (hi - lo)
    u = np.random.uniform(lo ** 2, hi ** 2)
    r = ROPE_REACH * np.sqrt(u)
    th = np.random.uniform(0.0, 2 * np.pi)
    return FIXED_NODE_POS + r * np.array([np.cos(th), np.sin(th)])


# ─────────────────────────────────────────────
# CSV schema
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


def write_description(path, case_id, case, csv_name):
    T = case["T"]
    T_drive = case.get("T_drive", T)
    wp = case["waypoints"]
    with open(path, "w") as f:
        f.write(f"Case {case_id}: {case['name']}\n")
        f.write(f"Motion profile: {_pname(case['profile'])} (ramped; no impulses)\n")
        f.write("Magnitude convention: mm total per action, spread over T_drive steps\n\n")
        f.write("Collection parameters:\n")
        f.write(f"  num_trajectories       : {NUM_TRAJECTORIES}\n")
        f.write(f"  actions_per_trajectory : {case['n_actions']}\n")
        f.write(f"  timesteps_per_action   : {T}\n")
        f.write(f"  driven_steps_per_action: {T_drive}\n")
        f.write(f"  u_max_mm_per_step      : {U_MAX_MM}\n")
        f.write(f"  koopman_step_dt        : {DT:.4f} s\n")
        f.write(f"  rows_per_trajectory    : {case['n_actions'] * T}\n")
        f.write(f"  rope_reach_mm          : {ROPE_REACH:.4f}\n")
        f.write(f"  max_reach_frac         : {MAX_REACH_FRAC}\n")
        f.write(f"  waypoint_radius_frac   : {wp['radius_frac'][0]:.2f} - {wp['radius_frac'][1]:.2f}\n")
        f.write(f"  waypoint_alternate     : {wp.get('alternate', False)}\n")
        f.write(f"  waypoint_tol_mm        : {WAYPOINT_TOL}\n")
        f.write(f"  waypoint_jitter_rad    : {WAYPOINT_JITTER}\n")
        f.write(f"  seed                   : {args.seed}\n\n")
        f.write("Rope physics parameters (PyElastica, SI internally):\n")
        f.write(f"  num_rope_points    : {num_points}\n")
        f.write(f"  base_length        : {env.base_length * MM:.1f} mm\n")
        f.write(f"  radius             : {env.base_radius * MM:.1f} mm\n")
        f.write(f"  youngs_modulus     : {env.youngs_modulus:.3e} (SI)\n")
        f.write(f"  damping            : {env.damping}\n")
        f.write(f"  sim_dt / substeps  : {env.dt:.1e} s / {env.substeps} per step\n\n")
        f.write(f"CSV: {csv_name} (long format, one row per (trajectory_id, action_idx, step_idx)):\n")
        f.write("  pos_x_i, pos_y_i                     : node positions (mm)\n")
        f.write("  vel_x_i, vel_y_i, ang_vel_z_i        : velocities (mm/s, rad/s)\n")
        f.write("  curvature_i                          : local turning angle (rad)\n")
        f.write("  edge_dx_i, edge_dy_i, edge_length_i  : edge vectors/lengths (mm)\n")
        f.write("  cmd_dx, cmd_dy                       : Node-0 displacement applied ON THIS STEP (mm)\n")
        f.write("                                         Zero on held steps; motion there is autonomous.\n")
        f.write("  target_x, target_y                   : commanded absolute Node-0 position (mm)\n")


def collect_case(case_id):
    case = CASES[case_id]
    T, n_actions = case["T"], case["n_actions"]
    wp_spec = case["waypoints"]
    r_lo, r_hi = wp_spec["radius_frac"]
    alternate = wp_spec.get("alternate", False)

    dest_dir = OUT_ROOT / f"case_{case_id}_{case['name']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "collected_trajectories_koopman_timeseries.csv"
    write_description(dest_dir / "description.txt", case_id, case, csv_path.name)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)

    print(f"\n=== Case {case_id}: {case['name']} "
          f"({n_actions} actions x {T} steps = {n_actions * T} rows/traj) ===")
    print(f"    -> {csv_path}")

    radii = []          # |node0 - centre| visited, for the coverage report
    case_start = time.time()
    traj_idx = 1
    pbar = tqdm(total=NUM_TRAJECTORIES, desc=f"case {case_id} ({case['name']})")

    while traj_idx <= NUM_TRAJECTORIES:
        env.reset()
        env.randomize_shape(force_scale=RANDOMIZE_SCALE)
        if env.is_exploded():
            continue

        target_node_pos = env.driven_position(mm=True)[:2].copy()
        traj_rows = []
        exploded = False

        want_inner = True if alternate else None
        waypoint = sample_waypoint(r_lo, r_hi, want_inner)

        for action_idx in range(n_actions):
            # --- new waypoint once the current one is reached ---------------
            d = waypoint - target_node_pos
            if np.linalg.norm(d) < WAYPOINT_TOL:
                if alternate:
                    want_inner = not want_inner
                waypoint = sample_waypoint(r_lo, r_hi, want_inner)
                d = waypoint - target_node_pos

            # --- head toward it; never overshoot it or the reach limit ------
            angle = (float(np.arctan2(d[1], d[0]))
                     + float(np.random.uniform(-WAYPOINT_JITTER, WAYPOINT_JITTER)))
            magnitude = min(float(case["magnitude"]()), float(np.linalg.norm(d)))

            direction = np.array([np.cos(angle), np.sin(angle)])
            endpoint = target_node_pos + magnitude * direction
            if np.linalg.norm(endpoint - FIXED_NODE_POS) > MAX_REACH_FRAC * ROPE_REACH:
                waypoint = sample_waypoint(r_lo, r_hi, want_inner)
                continue          # skip this action rather than pull past the limit

            step_command = case["profile"](magnitude, angle, T)

            for step_idx in range(T):
                cmd_mm = np.asarray(step_command(step_idx), dtype=float)
                target_node_pos = target_node_pos + cmd_mm
                env.drive_step(cmd_mm / MM)

                if env.is_exploded():
                    exploded = True
                    break

                positions = env.get_positions(mm=True)
                velocities = env.get_velocities(mm=True)
                ang_vel_z = env.get_node_angular_velocity_z()
                traj_rows.append(
                    build_row(traj_idx, action_idx, step_idx,
                              positions, velocities, ang_vel_z,
                              cmd_mm, target_node_pos))
            if exploded:
                break
            radii.append(float(np.linalg.norm(target_node_pos - FIXED_NODE_POS)))

        if exploded or not traj_rows:
            continue

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(traj_rows)
        pbar.update(1)
        traj_idx += 1

    pbar.close()

    # --- disc coverage report: the whole point of this design ---------------
    if radii:
        r = np.array(radii) / ROPE_REACH
        hist, _ = np.histogram(r, bins=np.linspace(0.0, 1.0, 11))
        print(f"    node-0 radius / L:  min {r.min():.2f}  mean {r.mean():.2f}  max {r.max():.2f}")
        print(f"    decile counts 0.0 -> 1.0: {hist.tolist()}")
        if (hist[2:9] == 0).any():
            print("    <-- EMPTY BANDS in the mid-disc; widen radius_frac or raise n_actions")

    elapsed = time.time() - case_start
    print(f"    case {case_id} done in {elapsed / 60:.2f} min")
    return elapsed


start_time = time.time()
timings = {c: collect_case(c) for c in SELECTED_CASES}
total = time.time() - start_time
print("\n─── Summary ───")
for c, el in timings.items():
    print(f"  case {c:>2} {CASES[c]['name']:<16} {el / 60:6.2f} min")
print(f"Data collection completed in {total / 60:.2f} minutes.")