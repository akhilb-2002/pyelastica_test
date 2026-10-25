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

Usage:
    python koopman_data_gen.py all --num-trajectories 100 --seed 0
    python koopman_data_gen.py 5 6 7 --num-trajectories 50
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
# Motion profiles  (identical semantics to the SOFA script)
#
# A profile turns one action (magnitude + direction + hold length T) into a
# per-step displacement increment of Node 0. Magnitudes are in millimetres.
# ─────────────────────────────────────────────
def _ramp(mag, angle, T):
    """Spread the displacement evenly across the hold: smooth, MPC-like motion."""
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)]) / T
    return lambda k: step

def _teleport(mag, angle, T):
    """All displacement on the first step, then hold (impulse response)."""
    jump = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: jump if k == 0 else np.zeros(2)


# ─────────────────────────────────────────────
# Per-step profiles (cases 10-16)
#
# For these, `mag` is the PER-STEP displacement (mm/step), not the action total.
# Decoupling speed from hold length lets T vary independently, so a short T with
# many actions gives frequent direction changes at a fixed speed -- which is what
# a shape-tracing MPC actually emits.
#
# u_max is the per-step ceiling used when collecting; constrain the solver to
# ~0.75 * U_MAX_MM so it never extrapolates off the edge of the training set.
# At control_dt = 0.01 s: 2.0 mm/step = 200 mm/s, and the rest segment is
# base_length / (n_nodes - 1) ~ 10.3 mm, so 2 mm/step ~ 0.2 segment/step.
# ─────────────────────────────────────────────
U_MAX_MM = 2.0

def _ramp_step(mag, angle, T):
    """Constant per-step displacement `mag` (mm/step), held for T steps."""
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: step

def _drive_hold(mag, angle, T, T_drive=20):
    """Drive at `mag` mm/step for T_drive steps, then hold (u = 0) for the rest.

    Case 8 settles BEFORE recording, so its transitions are ~identity and push the
    spectrum toward the unit circle. This records relaxation from ENERGETIC states
    instead -- the decay data A needs to be learned as contractive.
    """
    step = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    return lambda k: step if k < T_drive else np.zeros(2)

WARMUP_RATE_MM = 1.0      # mm/step while driving the warm-up (well under U_MAX_MM)
WARMUP_SETTLE = 60        # held steps (u = 0) after the drive, to kill transients
WARMUP_SETTLE_TOL = 5.0   # mm/s; max |v| considered "settled"

def warmup_cmds(bend_mm, angle, n_legs=1, turn=0.0, rate=WARMUP_RATE_MM):
    """Per-step mm increments for a multi-leg warm-up drive.

    Total path length is `bend_mm`, split evenly over `n_legs`; each leg rotates
    the drive direction by `turn` radians from the last. n_legs=1 gives a straight
    pull (simple arc); n_legs=3-5 with turn ~0.4 rad gives a curled shape.
    """
    per_leg = bend_mm / n_legs
    steps_per_leg = max(1, int(round(per_leg / rate)))
    step_mm = per_leg / steps_per_leg          # exact, absorbs the rounding
    cmds = []
    for leg in range(n_legs):
        a = angle + leg * turn
        cmds.extend([step_mm * np.array([np.cos(a), np.sin(a)])] * steps_per_leg)
    return cmds


def sample_warmup(spec, start_mm):
    """Sample a warm-up whose endpoint stays inside the rope's reach.

    Returns (cmds, bend_mm, angle) or (None, ...) if no valid sample was found.
    """
    for _ in range(MAX_TARGET_RESAMPLES):
        bend_mm = float(np.random.uniform(*spec["bend_mm"]))
        angle = np.random.uniform(0.0, 2 * np.pi)
        n_legs = spec.get("n_legs", 1)
        turn = float(np.random.uniform(*spec["turn"])) if "turn" in spec else 0.0
        cmds = warmup_cmds(bend_mm, angle, n_legs, turn, spec.get("rate", WARMUP_RATE_MM))
        endpoint = start_mm + np.sum(cmds, axis=0)
        if np.linalg.norm(endpoint - FIXED_NODE_POS) <= MAX_REACH_FRAC * ROPE_REACH:
            return cmds, bend_mm, angle
    return None, None, None


def run_warmup(env, spec):
    """Drive the rope into a deformed configuration. Nothing is recorded.

    Returns (ok, bend_mm, angle, max_v). ok=False means exploded or unreachable;
    the caller should retry the trajectory.
    """
    start_mm = env.driven_position(mm=True)[:2].copy()
    cmds, bend_mm, angle = sample_warmup(spec, start_mm)
    if cmds is None:
        return False, None, None, None

    for c in cmds:
        env.drive_step(c / MM)
        if env.is_exploded():
            return False, None, None, None

    for _ in range(spec.get("settle", WARMUP_SETTLE)):
        env.drive_step(np.zeros(2))
        if env.is_exploded():
            return False, None, None, None

    max_v = float(np.abs(env.get_velocities(mm=True)[:, :2]).max())
    return True, bend_mm, angle, max_v


def _pname(profile):
    """Profile name for logging, unwrapping functools.partial."""
    p = profile
    while isinstance(p, functools.partial):
        p = p.func
    return p.__name__.lstrip("_")

CASES = {
    # ---- impulse response, wide magnitude coverage --------------------------
    1: dict(name="fixed_10mm",      magnitude=lambda: 10.0,
            profile=_teleport, T=20, n_actions=5),
    2: dict(name="fixed_50mm",      magnitude=lambda: 50.0,
            profile=_teleport, T=20, n_actions=5),
    3: dict(name="random_1_10mm",   magnitude=lambda: float(np.random.uniform(1, 10)),
            profile=_teleport, T=20, n_actions=5),
    4: dict(name="random_10_80mm",  magnitude=lambda: float(np.random.uniform(10, 80)),
            profile=_teleport, T=20, n_actions=5),
    # ---- smooth motion: matches how MPC will actually drive the rope --------
    5: dict(name="ramp_1_10mm",     magnitude=lambda: float(np.random.uniform(1, 10)),
            profile=_ramp, T=20, n_actions=5),
    6: dict(name="ramp_10_80mm",    magnitude=lambda: float(np.random.uniform(10, 80)),
            profile=_ramp, T=20, n_actions=5),
    # ---- small-signal: the regime a linear model should nail ---------------
    7: dict(name="micro_0.1_1mm",   magnitude=lambda: float(np.random.uniform(0.1, 1.0)),
            profile=_ramp, T=10, n_actions=10),
    # ---- autonomous dynamics: u = 0, isolates A from B ---------------------
    8: dict(name="free_relaxation", magnitude=lambda: 0.0,
            profile=_teleport, T=100, n_actions=2),
    # ---- long-horizon held-out set for rollout evaluation ------------------
    9: dict(name="long_mixed",      magnitude=lambda: float(np.random.uniform(1, 60)),
            profile=_ramp, T=20, n_actions=40),

    # ========== per-step cases: `magnitude` is mm/STEP, not an action total ====
    # Total action displacement = magnitude * (T_drive if present else T).
    # ---- speed sweep at the nominal hold length (T = 20) --------------------
    10: dict(name="micro_step",      magnitude=lambda: float(np.random.uniform(0.02, 0.2)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    11: dict(name="slow_step",       magnitude=lambda: float(np.random.uniform(0.2, 0.8)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    12: dict(name="fast_step",       magnitude=lambda: float(np.random.uniform(0.8, U_MAX_MM)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True),
    # ---- frequent direction changes: closest match to a tracing controller --
    13: dict(name="turn_short",      magnitude=lambda: float(np.random.uniform(0.2, 1.5)),
             profile=_ramp_step, T=5, n_actions=40, per_step=True),
    14: dict(name="turn_v_short",    magnitude=lambda: float(np.random.uniform(0.2, 1.5)),
             profile=_ramp_step, T=2, n_actions=100, per_step=True),
    # ---- relaxation from MOVING states: the decay data A needs -------------
    15: dict(name="drive_hold",      magnitude=lambda: float(np.random.uniform(0.5, 1.5)),
             profile=functools.partial(_drive_hold, T_drive=20),
             T=60, n_actions=5, per_step=True, T_drive=20),
    # ---- long-horizon held-out set for rollout evaluation ------------------
    16: dict(name="long_mixed_step", magnitude=lambda: float(np.random.uniform(0.1, 1.5)),
             profile=_ramp_step, T=10, n_actions=80, per_step=True),

    # ========== bent-start cases: recording begins from a DEFORMED rope ========
    # Same per-step profiles as 10-16, but each trajectory is warmed up into a
    # curved configuration before logging starts. These are the cases that give
    # the encoder state-space coverage away from the straight rope.
    # ---- mild bend, fine motion: the small-signal regime, off the straight axis
    17: dict(name="bent_micro_step",  magnitude=lambda: float(np.random.uniform(0.05, 0.4)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True,
             warmup=dict(bend_mm=(10.0, 40.0), n_legs=1)),
    # ---- moderate bend, nominal speed --------------------------------------
    18: dict(name="bent_slow_step",   magnitude=lambda: float(np.random.uniform(0.2, 0.8)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True,
             warmup=dict(bend_mm=(40.0, 90.0), n_legs=1)),
    # ---- deep bend: pushes the coverage boundary as far as the rope allows --
    19: dict(name="deep_bend_step",   magnitude=lambda: float(np.random.uniform(0.2, 1.0)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True,
             warmup=dict(bend_mm=(90.0, 150.0), n_legs=1)),
    # ---- CURLED starts: multi-leg warm-up sweeps the drive direction, giving
    #      shapes with real curvature variation rather than single arcs. This is
    #      the case most directly aimed at letter targets.
    20: dict(name="curl_step",        magnitude=lambda: float(np.random.uniform(0.2, 1.0)),
             profile=_ramp_step, T=20, n_actions=10, per_step=True,
             warmup=dict(bend_mm=(60.0, 140.0), n_legs=4, turn=(0.20, 0.55))),
    # ---- curled start + frequent direction changes: closest match to tracing -
    21: dict(name="curl_turn_short",  magnitude=lambda: float(np.random.uniform(0.2, 1.5)),
             profile=_ramp_step, T=5, n_actions=40, per_step=True,
             warmup=dict(bend_mm=(60.0, 140.0), n_legs=4, turn=(0.20, 0.55))),
    # ---- relaxation from a bent, ENERGETIC state: decay data away from straight
    22: dict(name="bent_drive_hold",  magnitude=lambda: float(np.random.uniform(0.5, 1.5)),
             profile=functools.partial(_drive_hold, T_drive=20),
             T=60, n_actions=5, per_step=True, T_drive=20,
             warmup=dict(bend_mm=(40.0, 120.0), n_legs=3, turn=(0.15, 0.45))),
    # ---- long-horizon held-out set from curled starts (rollout evaluation) ---
    23: dict(name="curl_long_mixed",  magnitude=lambda: float(np.random.uniform(0.1, 1.5)),
             profile=_ramp_step, T=10, n_actions=80, per_step=True,
             warmup=dict(bend_mm=(50.0, 130.0), n_legs=4, turn=(0.20, 0.55))),

    # ===== deep-curl coverage: the states letter targets actually occupy =====
    # The gap these close is STATES, not command magnitude -- the C was reached
    # at 0.893 mm/step, well inside the existing 0.02-1.5 range. What was missing
    # is NET travel in a COHERENT direction: `sample_action` redraws the angle
    # every action, so many short actions random-walk (~sqrt(n) growth) instead
    # of accumulating curl. Hence n_actions=1 with a long T -- one sustained
    # sweep of 180-360 mm, the same drive that produced the C.
    # ---- one coherent sweep into a deep curl -------------------------------
    24: dict(name="curl_sweep",       magnitude=lambda: float(np.random.uniform(0.6, 1.2)),
             profile=_ramp_step, T=300, n_actions=1, per_step=True),
    # ---- same sweep, then HOLD: relaxation from a deep curl -----------------
    25: dict(name="curl_sweep_hold",  magnitude=lambda: float(np.random.uniform(0.6, 1.2)),
             profile=functools.partial(_drive_hold, T_drive=300),
             T=800, n_actions=1, per_step=True, T_drive=300),
    # ---- UNCURL: drive back toward straight from a curled start ------------
    #      every earlier case drives outward only; MPC needs both directions.
    #      `drive_toward="straight"` is honoured in collect_case below.
    26: dict(name="uncurl_step",      magnitude=lambda: float(np.random.uniform(0.4, 1.2)),
             profile=_ramp_step, T=10, n_actions=30, per_step=True,
             warmup=dict(bend_mm=(200.0, 320.0), n_legs=1),
             drive_toward="straight"),
    # ---- small motions FROM deep curl: the local regime MPC operates in ----
    27: dict(name="deep_micro_step",  magnitude=lambda: float(np.random.uniform(0.05, 0.5)),
             profile=_ramp_step, T=20, n_actions=15, per_step=True,
             warmup=dict(bend_mm=(200.0, 320.0), n_legs=1)),
    # ---- held-out: long rollouts from deep curl ----------------------------
    28: dict(name="deep_long_mixed",  magnitude=lambda: float(np.random.uniform(0.1, 1.2)),
             profile=_ramp_step, T=10, n_actions=80, per_step=True,
             warmup=dict(bend_mm=(200.0, 320.0), n_legs=1)),
    29: dict(name="fast_sweep",     magnitude=lambda: float(np.random.uniform(1.5, 3.0)),
             profile=_ramp_step, T=100, n_actions=1, per_step=True),
    30: dict(name="fast_turn",      magnitude=lambda: float(np.random.uniform(1.5, 3.0)),
             profile=_ramp_step, T=5, n_actions=40, per_step=True),
}

# ---------------------------------------------------------------------------
# Cases 1-9: EXCLUDED. Teleport-profile pollution (impulsive inputs give ~50x
# higher MSE than ramps at matched amplitude), legacy magnitude convention
# (per-action total, not per-step), and near-identity transitions.
# ---------------------------------------------------------------------------

# Cases 10-15: per-step ramp profiles at nominal speeds, straight starts.
BASE_CASES = [10, 11, 12, 13, 14, 15]
# Cases 17-22: bent/curled starts -- coverage away from the straight rope.
BENT_CASES = [17, 18, 19, 20, 21, 22]
# Cases 24-27: deep-curl coverage. Coherent single-bearing sweeps reaching the
# ~300 mm net travel that letter targets require, plus uncurling and small
# motions from deep curl.
DEEP_CASES = [24, 25, 26, 27, 29, 30]

TRAIN_CASES = BASE_CASES + BENT_CASES + DEEP_CASES
# -> [10,11,12,13,14,15, 17,18,19,20,21,22, 24,25,26,27, 29,30]

# Held out for long-horizon rollout evaluation -- never in the training glob.
# NOTE: a `case2*` glob would sweep up 20-28 including these. Match on the full
# directory name (case_16_*, case_23_*, case_28_*) when building training sets.
HELDOUT_CASES = [16, 23, 28]


_case_help = ", ".join(f"{k}={v['name']}" for k, v in sorted(CASES.items()))
parser = argparse.ArgumentParser(description="Collect Koopman time-series rope data (PyElastica).")
parser.add_argument("cases", nargs="+",
                    help=f"Case number(s), 'all', 'new' (={TRAIN_CASES}), "
                         f"'bent' (={BENT_CASES}), 'deep' (={DEEP_CASES}), "
                         f"or 'trace' (= the full training set). "
                         f"Available: {_case_help}")
parser.add_argument("--num-trajectories", type=int, default=100,
                    help="Number of independent multi-action rollouts per case.")
parser.add_argument("--seed", type=int, default=None, help="RNG seed (recorded in description.txt).")
args = parser.parse_args()

_sel = args.cases[0].lower() if len(args.cases) == 1 else None
if _sel == "all":
    SELECTED_CASES = sorted(CASES.keys())
elif _sel in ("new", "trace"):
    # TRAIN_CASES already contains BENT_CASES -- don't concatenate, or 17-22
    # get collected twice.
    SELECTED_CASES = list(TRAIN_CASES)
elif _sel == "bent":
    SELECTED_CASES = list(BENT_CASES)
elif _sel == "deep":
    SELECTED_CASES = list(DEEP_CASES)
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
UNCURL_JITTER = 0.4        # rad; spread around the straight-home bearing (case 26)

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
    seed=args.seed,)

env.reset()
num_points = env.n_nodes
DT = env.control_dt                      # one recorded timestep, in seconds
print(f"Rope has {num_points} nodes. Recorded timestep dt = {DT:.4f} s.")

# Rope reach and the fixed-node position, measured on the settled rest config.
FIXED_NODE_POS = env.fixed_position(mm=True)[:2].copy()
ROPE_REACH = env.rope_reach(mm=True)
# Node 0's position on the straight rest rope -- the "home" the uncurl case aims
# for. Taken from the settled reset config rather than assumed, so it stays
# correct if the rope's rest orientation ever changes.
HOME_NODE0_POS = env.driven_position(mm=True)[:2].copy()
print(f"Rope reach (rest arc length): {ROPE_REACH:.2f} mm | "
      f"target limit: {MAX_REACH_FRAC * ROPE_REACH:.2f} mm from fixed node")
print(f"Fixed node: {FIXED_NODE_POS} mm | straight-rope node 0 (uncurl home): {HOME_NODE0_POS} mm")

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

    Legacy cases (1-9): `magnitude` IS the total. Per-step cases (10-16): it is
    mm/step, so the total is magnitude * the number of DRIVEN steps -- T_drive for
    drive-hold profiles, T otherwise.
    """
    if not case.get("per_step", False):
        return magnitude
    return magnitude * case.get("T_drive", case["T"])


def sample_action(case, current_target_mm):
    """Sample (magnitude, angle) in mm whose endpoint stays within the rope's reach."""
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
    with open(path, "w") as f:
        f.write(f"Case {case_id}: {case['name']}\n")
        f.write(f"Motion profile: {_pname(case['profile'])}\n")
        f.write("Magnitude convention: "
                f"{'mm/step' if case.get('per_step') else 'mm (total per action)'}\n\n")
        f.write("Collection parameters:\n")
        f.write(f"  num_trajectories       : {NUM_TRAJECTORIES}\n")
        f.write(f"  actions_per_trajectory : {case['n_actions']}\n")
        f.write(f"  timesteps_per_action   : {case['T']}\n")

        if case.get("per_step"):
            f.write(f"  driven_steps_per_action: {case.get('T_drive', case['T'])}\n")
            f.write(f"  u_max_mm_per_step      : {U_MAX_MM}\n")

        if case.get("drive_toward"):
            f.write(f"  drive_toward           : {case['drive_toward']} "
                    f"(bearing aimed at {HOME_NODE0_POS}, jitter "
                    f"+/-{UNCURL_JITTER} rad)\n")

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

        w = case.get("warmup")
        if w:
            f.write("  warmup_bend_mm         : "
                    f"{w['bend_mm'][0]:.1f} - {w['bend_mm'][1]:.1f}\n")
            f.write(f"  warmup_n_legs          : {w.get('n_legs', 1)}\n")
            if "turn" in w:
                f.write(f"  warmup_turn_rad        : {w['turn'][0]:.2f} - {w['turn'][1]:.2f}\n")
            f.write(f"  warmup_rate_mm_step    : {w.get('rate', WARMUP_RATE_MM)}\n")
            f.write(f"  warmup_settle_steps    : {w.get('settle', WARMUP_SETTLE)}\n")


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

    warm_stats = []

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

        if case.get("warmup"):
            ok, bend_mm, angle, max_v = run_warmup(env, case["warmup"])
            if not ok:
                continue  # retry this rollout
            warm_stats.append((bend_mm, angle, max_v))

        target_node_pos = env.driven_position(mm=True)[:2].copy()

        # Uncurl cases aim each action back at the straight-rope node-0 position
        # instead of a fresh random bearing, so the rope actually straightens.
        # Without this the case is just a duplicate of 27 with bigger steps.
        aim_home = case.get("drive_toward") == "straight"

        for action_idx in range(n_actions):
            magnitude, angle = sample_action(case, target_node_pos)
            if aim_home:
                d = HOME_NODE0_POS - target_node_pos
                if np.linalg.norm(d) > 1e-6:
                    angle = (float(np.arctan2(d[1], d[0]))
                             + float(np.random.uniform(-UNCURL_JITTER, UNCURL_JITTER)))
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

    if warm_stats:
        b = np.array([w[0] for w in warm_stats])
        v = np.array([w[2] for w in warm_stats])      # index 2 = max_v; 1 is the angle
        print(f"    warm-up bend mm: mean {b.mean():.1f}  range [{b.min():.1f}, {b.max():.1f}]")
        print(f"    settled max|v| mm/s: mean {v.mean():.2f}  max {v.max():.2f}"
              + ("  <-- NOT SETTLED, raise WARMUP_SETTLE" if v.max() > WARMUP_SETTLE_TOL else ""))

    elapsed = time.time() - case_start
    print(f"    case {case_id} done in {elapsed / 60:.2f} min "
          f"({NUM_TRAJECTORIES * n_actions * T} rows)")
    return elapsed


start_time = time.time()
timings = {c: collect_case(c) for c in SELECTED_CASES}
total = time.time() - start_time
print("\n─── Summary ───")
for c, el in timings.items():
    print(f"  case {c:>2} {CASES[c]['name']:<16} {el / 60:6.2f} min")
print(f"Data collection completed in {total / 60:.2f} minutes.")