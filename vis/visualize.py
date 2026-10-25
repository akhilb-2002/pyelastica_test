"""Drive the RopeOnTable case study and export a trajectory for rendering.

Follows the pattern of examples/flexible_swinging_pendulum/visualize.py: the
C++ case study is stepped from Python through `run_for`, and `communicate()`
hands back the rod so its state can be read each iteration.

The dump is the same pickled dict that the PyElastica POVray example consumes
(`time`, `position`, `radius`, `director`), so the existing rope_render.py can
render it unchanged.

    python visualize.py --drive lift --final-time 3.0 --out rope_on_table.dat
    python rope_render.py --data rope_on_table.dat --still 1.0
"""

import argparse
import pickle

import numpy as np
from elasticapp.Systems import CosseratRods  # noqa: F401  (registers bindings)
from elasticapp.CaseStudies.PyRopeOnTableCaseStudy import RopeOnTableCaseStudy


def run(drive="lift", final_time=3.0, n_frames=300, n_elem=40,
        length=1.0, radius=0.020, out="rope_on_table.dat", up="y"):
    dt = 0.002 * (length / n_elem)

    params = RopeOnTableCaseStudy.params(
        n_elements=n_elem,
        length=length,
        radius=radius,
        youngs_modulus=5e5,
        shear_modulus=5e5 / 3.0,
        damping_rate=1e-4,
        gravitational_acc=-9.80665,
        static_friction=0.6,
        kinetic_friction=0.4,
        drive=drive,
        drive_time=0.85 * final_time,
        dt=dt,
        final_sim_time=final_time,
    )
    cs = RopeOnTableCaseStudy(params)

    times, positions, radii, directors = [], [], [], []

    def collect():
        rod = cs.communicate()
        times.append(cs.current_time())
        positions.append(np.asarray(rod.get_position()).copy())
        radii.append(np.asarray(rod.get_radius()).copy())
        directors.append(np.asarray(rod.get_director()).copy())

    collect()
    interval = final_time / n_frames
    for _ in range(n_frames):
        cs.run_for(additional_time=interval)
        collect()

    pos = np.asarray(positions)              # (T, 3, n_node)
    if up == "y":
        # the case study is z-up; POVray is y-up
        pos = pos[:, [0, 2, 1], :]

    with open(out, "wb") as fh:
        pickle.dump(
            {
                "time": list(np.asarray(times)),
                "position": list(pos),
                "radius": list(np.asarray(radii)),
                "director": list(np.asarray(directors)),
            },
            fh,
        )
    print(f"wrote {out}  ({len(times)} frames, {pos.shape[2]} nodes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", choices=["lift", "sweep", "settle"],
                    default="lift")
    ap.add_argument("--final-time", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--n-elem", type=int, default=40)
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=0.020)
    ap.add_argument("--up", choices=["y", "z"], default="y",
                    help="'y' rewrites axes for POVray; 'z' keeps sim frame")
    ap.add_argument("--out", default="rope_on_table.dat")
    a = ap.parse_args()
    run(a.drive, a.final_time, a.frames, a.n_elem, a.length, a.radius,
        a.out, a.up)
