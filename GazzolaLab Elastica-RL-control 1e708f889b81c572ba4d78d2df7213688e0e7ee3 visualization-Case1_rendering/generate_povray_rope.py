"""Generate POV-Ray frames for rope_manip_env.py rollouts.

Mirrors generate_POVray_files.py's output layout (images/moving_arm%02d.inc/.pov)
so render_frames.sh and make_vid.sh work unchanged, but draws the rope from
RopeManipEnv instead of the arm-reaching npz data.

Coordinate mapping (must match camera_position.inc, case 1):
    POV <X,Y,Z> = SCALE * <sim_x, sim_z, sim_y>
i.e. sim's vertical z (table = 0) becomes POV's up axis y.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rope_manip_env import RopeManipEnv  # noqa: E402

SCALE = 100.0  # metres -> POV units; keep in sync with camera_position.inc

DRIVEN_COLOR = "rgb<1,0.5,0.4>"   # highlight the driven (controlled) end
FIXED_COLOR = "rgb<0.75,0.75,0.75>"  # highlight the pinned end
ROPE_COLOR = "rgb<0.45,0.39,1>"
POLE_COLOR = "rgb<0.3,0.3,0.32>"
MARKER_RADIUS = 0.5  # POV units

POLE_HEIGHT = 4.0    # POV units, vertical post marking the fixed (pinned) end
AXIS_LENGTH = 4.0    # POV units, per-arm length of the driven-end axes gizmo
AXIS_RADIUS = 0.12


def to_pov(p):
    """sim (x, y, z) -> POV (x, z, y), scaled."""
    return SCALE * p[0], SCALE * p[2], SCALE * p[1]


def fixed_pole(origin):
    """Vertical post + knob planted at the table, marking the pinned end."""
    x, y, z = origin
    top = y + POLE_HEIGHT
    knob_r = 1.8 * AXIS_RADIUS
    s = "cylinder{<%f,%f,%f>,<%f,%f,%f>,%f\n" % (x, 0.0, z, x, top, z, 0.9 * AXIS_RADIUS)
    s += "texture{ pigment{ color %s } finish{ phong 0.3 } }\n}\n" % POLE_COLOR
    s += "sphere{<%f,%f,%f>,%f\n" % (x, top, z, knob_r)
    s += "texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % POLE_COLOR
    return s


def axis_gizmo(origin):
    """RGB (x, y, z) arrow triad at the driven end, marking the movable frame."""
    x, y, z = origin
    axes = [
        ((AXIS_LENGTH, 0.0, 0.0), "rgb<0.9,0.15,0.15>"),   # +X, red
        ((0.0, AXIS_LENGTH, 0.0), "rgb<0.15,0.75,0.15>"),  # +Y (up), green
        ((0.0, 0.0, AXIS_LENGTH), "rgb<0.15,0.35,0.95>"),  # +Z, blue
    ]
    parts = []
    for (dx, dy, dz), color in axes:
        tip_x, tip_y, tip_z = x + dx, y + dy, z + dz
        parts.append("cylinder{<%f,%f,%f>,<%f,%f,%f>,%f\n" % (x, y, z, tip_x, tip_y, tip_z, AXIS_RADIUS))
        parts.append("texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % color)
        cone_tip = (tip_x + 0.3 * dx / AXIS_LENGTH, tip_y + 0.3 * dy / AXIS_LENGTH, tip_z + 0.3 * dz / AXIS_LENGTH)
        parts.append("cone{<%f,%f,%f>,%f,<%f,%f,%f>,0\n" % (tip_x, tip_y, tip_z, 2.5 * AXIS_RADIUS, *cone_tip))
        parts.append("texture{ pigment{ color %s } finish{ phong 1 } }\n}\n" % color)
    return "".join(parts)


def write_frame(k, positions, radius):
    inc_path = os.path.join("images", "moving_arm%02d.inc" % k)
    with open(inc_path, "w") as f:
        # Rope
        f.write("sphere_sweep\n{b_spline %d" % positions.shape[0])
        for x, y, z in positions:
            px, py, pz = to_pov((x, y, z))
            f.write(",\n<%f,%f,%f>,%f" % (px, py, pz, SCALE * radius))
        f.write("\ntexture{")
        f.write("pigment{ color %s transmit 0.1 }" % ROPE_COLOR)
        f.write("finish{ phong 1 } }\n}\n")

        # Driven end marker
        dx, dy, dz = to_pov(positions[0])
        f.write("sphere\n{<%f,%f,%f>,%f\n" % (dx, dy, dz, 1.5 * SCALE * radius))
        f.write("texture{ pigment{ color %s transmit 0.1 } finish{ phong 1 } }\n}\n"
                % DRIVEN_COLOR)

        # Fixed end marker + vertical pole signifying the pinned point
        fx, fy, fz = to_pov(positions[-1])
        f.write("sphere\n{<%f,%f,%f>,%f\n" % (fx, fy, fz, MARKER_RADIUS))
        f.write("texture{ pigment{ color %s transmit 0.1 } finish{ phong 1 } }\n}\n"
                % FIXED_COLOR)
        f.write(fixed_pole((fx, fy, fz)))

        # Axes gizmo at the driven end, showing the movable frame
        f.write(axis_gizmo((dx, dy, dz)))

    pov_path = os.path.join("images", "moving_arm%02d.pov" % k)
    with open(pov_path, "w") as f:
        f.write("#include \"../camera_position.inc\"\n")
        f.write("#include \"moving_arm%02d.inc\"\n" % k)


def main():
    os.makedirs("images", exist_ok=True)

    env = RopeManipEnv(seed=0)
    env.reset()
    env.randomize_shape()

    frames = [(env.get_positions().copy(), env.base_radius)]
    n_steps = 20
    for _ in range(n_steps):
        env.drive_step(np.array([0.0, 0.002]))  # 2 mm/step ramp in +y
        frames.append((env.get_positions().copy(), env.base_radius))

    for k, (positions, radius) in enumerate(frames):
        write_frame(k, positions, radius)

    print("wrote %d frames to images/" % len(frames))


if __name__ == "__main__":
    main()
