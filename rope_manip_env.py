"""
Kinematic rope-manipulation environment in PyElastica for Koopman/EDMD data.

This is the PyElastica analogue of the SOFA `rope_manipulation_koopman_timeseries`
scene used for Koopman data collection. The control paradigm matches that scene:

  * Node 0 is the DRIVEN end - its position is imposed each step (kinematic
    "gripper"). The control input u_k is Node 0's per-step displacement (dx, dy).
  * The LAST node (index n_nodes-1) is FIXED (pinned to the table).
  * All interior nodes are free: gravity presses them onto a flat frictional
    table, and they respond to the driven-end motion through the rod's elasticity.

Unlike the SOFA pipeline, PyElastica integrates the rod dynamics directly, so no
PBD inextensibility projection or velocity back-correction is needed - node
velocities come straight from the integrator. A driven-end displacement is spread
smoothly across the physics substeps of each control step, which keeps a large
"teleport" command (up to ~80 mm in one step) inextensible and stable
(<~4% element stretch) without an explicit length constraint.

Units: the simulation runs in SI (metres). The collection script works in
millimetres to match the SOFA CSVs; use `mm=True` on the readout helpers, or
convert with the MM = 1000.0 factor.

Coordinate convention: table is the z = 0 plane, gravity is -z. The rope lies
along +x with the driven node (0) at the origin and the fixed node at (L, 0).
"""

from __future__ import annotations

import numpy as np
import elastica as ea
from elastica.boundary_conditions import ConstraintBase

MM = 1000.0  # metres -> millimetres


# --------------------------------------------------------------------------- #
# Kinematic driving
# --------------------------------------------------------------------------- #
class _DriveHolder:
    """Mutable target/velocity for the driven node, shared with the constraint."""

    def __init__(self, start):
        self.target = np.asarray(start, dtype=float).copy()
        self.velocity = np.zeros(3)


class _EndDriver(ConstraintBase):
    """Impose the driven node's position/velocity; pin the fixed node in place."""

    def __init__(self, *args, holder=None, driven_idx=0, fixed_idx=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.holder = holder
        self.driven_idx = driven_idx
        self.fixed_idx = fixed_idx
        self._fixed_pos = None

    def constrain_values(self, system, time):
        if self._fixed_pos is None:  # capture rest position of the fixed node once
            self._fixed_pos = system.position_collection[:, self.fixed_idx].copy()
        system.position_collection[:, self.driven_idx] = self.holder.target
        system.position_collection[:, self.fixed_idx] = self._fixed_pos

    def constrain_rates(self, system, time):
        system.velocity_collection[:, self.driven_idx] = self.holder.velocity
        system.velocity_collection[:, self.fixed_idx] = 0.0


class _ForceHolder:
    def __init__(self, n_nodes):
        self.force = np.zeros((3, n_nodes))


class _TransientForces(ea.NoForces):
    """Adds the current held force each step (used to randomize the rope shape)."""

    def __init__(self, holder):
        self.holder = holder

    def apply_forces(self, system, time=0.0):
        system.external_forces += self.holder.force


class _RopeSim(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class RopeManipEnv:
    def __init__(
        self,
        n_nodes: int = 30,
        base_length: float = 0.30,     # metres 
        base_radius: float = 0.003,    # metres  
        density: float = 1000.0,
        youngs_modulus: float = 1e6,
        poisson_ratio: float = 0.5,
        dt: float = 5e-5,
        control_dt: float = 0.01,      # one recorded step (SOFA: frame_skip*time_step)
        gravity: float = 9.81,
        damping: float = 2.0,
        static_mu: float = 0.3,
        kinetic_mu: float = 0.2,
        plane_k: float = 1e2,
        plane_nu: float = 1e1,
        seed: int | None = None,
    ):
        self.n_nodes = int(n_nodes)
        self.driven_idx = 0
        self.fixed_idx = self.n_nodes - 1
        self.base_length = base_length
        self.base_radius = base_radius
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        self.dt = dt
        self.control_dt = control_dt
        self.substeps = max(1, int(round(control_dt / dt)))
        self.gravity = gravity
        self.damping = damping
        self.static_mu = static_mu
        self.kinetic_mu = kinetic_mu
        self.plane_k = plane_k
        self.plane_nu = plane_nu
        self._rng = np.random.default_rng(seed)
        self.time = 0.0
        self._build()

    # ---- construction -------------------------------------------------- #
    def _build(self):
        self.sim = _RopeSim()
        self.rod = ea.CosseratRod.straight_rod(
            self.n_nodes - 1,
            start=np.array([0.0, 0.0, self.base_radius]),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=self.base_length,
            base_radius=self.base_radius,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )
        self.sim.append(self.rod)

        # Driven node 0 (imposed) + fixed last node (pinned).
        self.driver = _DriveHolder(start=np.array([0.0, 0.0, self.base_radius]))
        self.sim.constrain(self.rod).using(
            _EndDriver,
            holder=self.driver,
            driven_idx=self.driven_idx,
            fixed_idx=self.fixed_idx,
            constrained_position_idx=(self.driven_idx, self.fixed_idx),
            constrained_director_idx=(),
        )

        # Transient forces for shape randomization.
        self.force_holder = _ForceHolder(self.n_nodes)
        self.sim.add_forcing_to(self.rod).using(_TransientForces, holder=self.force_holder)

        self.sim.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -self.gravity])
        )
        self.sim.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            uniform_damping_constant=self.damping,
            time_step=self.dt,
        )
        self.plane = ea.Plane(
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        self.sim.append(self.plane)
        self.sim.detect_contact_between(self.rod, self.plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=self.plane_k,
            nu=self.plane_nu,
            slip_velocity_tol=1e-4,
            static_mu_array=np.array([self.static_mu] * 3),
            kinetic_mu_array=np.array([self.kinetic_mu] * 3),
        )
        self.sim.finalize()
        self.stepper = ea.PositionVerlet()
        self.time = 0.0

    # ---- integration --------------------------------------------------- #
    def _integrate(self, n):
        for _ in range(n):
            self.time = self.stepper.step(self.sim, self.time, self.dt)

    def reset(self, settle_threshold: float = 1e-5, settle_max_time: float = 0.5):
        """Rebuild a straight rope and settle it onto the table."""
        self._build()
        self.driver.velocity[:] = 0.0
        self.force_holder.force[:] = 0.0
        self.settle(settle_threshold, settle_max_time)
        return self.get_positions()

    def settle(self, threshold: float = 1e-5, max_time: float = 0.5):
        """Integrate (holding both ends) until per-step motion falls below threshold (m)."""
        self.force_holder.force[:] = 0.0
        self.driver.velocity[:] = 0.0
        prev = self.rod.position_collection.copy()
        n_chunk = self.substeps
        for _ in range(int(round(max_time / self.control_dt))):
            self._integrate(n_chunk)
            curr = self.rod.position_collection
            if np.max(np.linalg.norm(curr - prev, axis=0)) < threshold:
                break
            prev = curr.copy()
        return self.get_positions()

    # ---- shape randomization (state-space coverage) -------------------- #
    def randomize_shape(self, force_scale: float = 0.1, duration: float = 0.1):
        """Kick the free nodes with random in-plane forces, then settle.

        Mirrors the SOFA step that applies random transient forces to the free
        nodes so each rollout starts from a different rope shape.
        """
        free = np.arange(1, self.n_nodes - 1)  # exclude driven (0) and fixed (last)
        f = np.zeros((3, self.n_nodes))
        f[0, free] = self._rng.normal(0.0, force_scale, size=free.size)
        f[1, free] = self._rng.normal(0.0, force_scale, size=free.size)
        self.force_holder.force = f
        self._integrate(int(round(duration / self.dt)))
        self.force_holder.force = np.zeros((3, self.n_nodes))
        return self.settle()

    # ---- driving ------------------------------------------------------- #
    def drive_step(self, cmd_xy):
        """Move the driven node by cmd (metres, in-plane) over one control step.

        The displacement is spread smoothly across the physics substeps so large
        commands stay stable. Returns the state at the end of the control step.
        """
        cmd = np.asarray(cmd_xy, dtype=float)
        p0 = self.driver.target.copy()
        pN = p0 + np.array([cmd[0], cmd[1], 0.0])
        self.driver.velocity = np.array([cmd[0], cmd[1], 0.0]) / self.control_dt
        for s in range(self.substeps):
            self.driver.target = p0 + ((s + 1) / self.substeps) * (pN - p0)
            self.time = self.stepper.step(self.sim, self.time, self.dt)
        self.driver.target = pN
        return self.get_positions()

    def set_driven_target(self, xy):
        """Teleport the driven-node target (used to seed a rollout's start)."""
        self.driver.target = np.array([xy[0], xy[1], self.base_radius], dtype=float)

    # ---- readouts ------------------------------------------------------ #
    def get_positions(self, mm: bool = False):
        p = self.rod.position_collection.T.copy()
        return p * MM if mm else p

    def get_velocities(self, mm: bool = False):
        v = self.rod.velocity_collection.T.copy()
        return v * MM if mm else v

    def get_node_angular_velocity_z(self):
        """Per-node angular velocity about z, mapped from element omega.

        PyElastica stores angular velocity per element (director frame). We average
        adjacent elements onto nodes. For a rope lying flat and bending in-plane the
        material-frame z ~ lab z, so omega_z is the planar spin rate; treat as an
        approximate logging channel (as in the SOFA export).
        """
        omega_z = self.rod.omega_collection[2]        # (n_elements,)
        node = np.empty(self.n_nodes)
        node[0] = omega_z[0]
        node[-1] = omega_z[-1]
        node[1:-1] = 0.5 * (omega_z[:-1] + omega_z[1:])
        return node

    def driven_position(self, mm: bool = False):
        p = self.rod.position_collection[:, self.driven_idx].copy()
        return p * MM if mm else p

    def fixed_position(self, mm: bool = False):
        p = self.rod.position_collection[:, self.fixed_idx].copy()
        return p * MM if mm else p

    def rope_reach(self, mm: bool = False):
        """Rest arc length (sum of segment lengths) in the current configuration."""
        seg = np.linalg.norm(np.diff(self.rod.position_collection[:2], axis=1), axis=0)
        reach = float(seg.sum())
        return reach * MM if mm else reach

    def is_exploded(self, threshold: float = 1e3):
        p = self.rod.position_collection
        return bool(np.any(~np.isfinite(p)) or np.any(np.abs(p) > threshold))


if __name__ == "__main__":
    env = RopeManipEnv(seed=0)
    env.reset()
    print("nodes:", env.n_nodes, "| reach:", round(env.rope_reach(mm=True), 1), "mm")
    print("driven node:", np.round(env.driven_position(mm=True), 1),
          "| fixed node:", np.round(env.fixed_position(mm=True), 1))
    env.randomize_shape()
    # ramp the driven node 40 mm in +y over 20 steps
    for _ in range(20):
        env.drive_step(np.array([0.0, 0.002]))  # 2 mm/step
    print("after ramp, driven node:", np.round(env.driven_position(mm=True), 1),
          "| exploded:", env.is_exploded())
    seg = np.linalg.norm(np.diff(env.get_positions(mm=True)[:, :2], axis=0), axis=1)
    print("max segment:", round(seg.max(), 2), "mm  (rest",
          round(env.base_length * MM / (env.n_nodes - 1), 2), "mm)")