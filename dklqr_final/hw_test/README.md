# Rope shaping — hardware replay package (letters C, L, I, U)

Four open-loop trajectories that deform a rope on a table into four target
shapes. Everything needed to replay them on hardware is here.

Units are **millimetres** and **seconds** throughout.

## Physical setup

Identical for all four runs.

- Rope: length 300 mm, radius 3 mm, lying flat on a table. Planar (2D) motion.
- Discretised as 30 nodes, 10.345 mm apart, straight along +x at t = 0.
- **Node 29 is the fixed end**, clamped at (300, 0). It never moves in any run.
- **Node 0 is the actuated end**, starting at the origin (0, 0). This is the
  point the robot end-effector grips and moves.
- Frame: table plane, origin at the actuated end's start pose, +x pointing
  toward the clamp. Any rigid transform into the robot base frame is fine —
  only the relative motion matters.
- Control period 10 ms (100 Hz) in every run.

Note the actuated end is at x = 0 and the clamp at x = 300, not the other way
round.

## The trajectory files

One CSV per letter: `rope_C_trajectory.csv`, `rope_L_trajectory.csv`,
`rope_I_trajectory.csv`, `rope_U_trajectory.csv`. Same columns in each.

| column | meaning |
|---|---|
| `step` | index, 0-based |
| `t_s` | time since start, seconds (`step * 0.01`) |
| `x_mm`, `y_mm` | **absolute target position of the actuated end** |
| `dx_mm`, `dy_mm` | incremental move from the previous row |
| `speed_mm_s` | `hypot(dx, dy) / 0.01`, for limit checking |

`x_mm`/`y_mm` are the running sum of the increments, pre-computed so no
integration is needed on your side. Use whichever pair suits the controller:
absolute Cartesian waypoints, or per-cycle deltas. Row 0 is the start pose with
zero increment.

Runs are independent — the rope is reset to straight before each one.

## Per-letter summary

| | C | L | I | U |
|---|---|---|---|---|
| rows in CSV | 275 | 350| 305 | 231 |
| duration | 2.74 s | 3.49 s | 3.04 s | 2.30 s |
| moving steps | 258 | 322 | 247 | 220 |
| trailing hold | 128 steps | 600 steps | 57 steps | 10 steps |
| end pose (mm) | (318, 113) | (483, 145) | (336, −295) | (289, 72) |
| path length | 360 mm | 553 mm | 456 mm | 342 mm |
| steps at speed limit | 31 | 103 | 146 | 56 |
| x range (mm) | 0 … 318 | 0 … 483 | 0 … 336 | 0 … 289 |
| y range (mm) | 0 … 123 | 0 … 172 | −295 … 0 | 0 … 113 |
| clamp separation | 300 → 114 | 300 → 234 | 300 → 297 | 300 → 73 |
| best shape RMSE | 7.87 mm | 9.14 mm | 9.88 mm | 27.62 mm |
| at step | 239 | 321 (last) | 229 | 229 (last) |
| final shape RMSE | 11.89 mm | 11.39 mm | 13.53 mm | 27.62 mm |



## Limits and clearances

- Peak commanded speed is **212 mm/s** in every run — a diagonal step with both
  axes at the 1.5 mm-per-cycle cap (150 mm/s per axis). Check this against the
  arm's limits before running.
- The rope never goes taut. Separation between the actuated end and the clamp
  starts at 300 mm and decreases; the largest value seen mid-run is 298.5 mm
  (in L), still under the 300 mm rope length. So the rope buckles and goes slack
  rather than stretching — no tension risk.
- Workspace needed, as a single envelope covering all four: roughly
  **x ∈ [0, 490] mm, y ∈ [−300, 175] mm** relative to the actuated end's start
  pose. Note that **I sweeps below the clamp line into negative y**, and **L
  reaches 483 mm in x, well past the clamp at x = 300**. Both need the table
  cleared accordingly.
- If the controller can't hold a 100 Hz cycle, resample the absolute columns.
  Linear interpolation is adequate — the paths are smooth.