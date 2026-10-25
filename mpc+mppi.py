"""
letter_mpc.py -- Koopman MPC for tracing letter shapes with the rope.

Two controllers, both on the SAME linear Koopman model z+ = A z + B u:

  --controller qp    Latent-space QP. The tracking cost is ||z - z_ref||^2
                     weighted by J^T J (J = decoder Jacobian at the goal), which
                     makes it a first-order proxy for mm position error. Convex,
                     fast, exact box constraints -- but the proxy is only valid
                     near z_ref.

  --controller mppi  Sampling MPC. Rolls K command sequences through the same
                     LINEAR dynamics, decodes each rollout, and scores it in
                     actual millimetres. No Jacobian approximation, no nullspace
                     drift. The rollouts are matmuls rather than simulator calls,
                     which is exactly what makes 1k+ samples per step affordable
                     -- the Koopman lift paying off inside the controller.

The two differ ONLY in how they handle the decoder, so comparing them isolates
the cost of the linearised cost. Everything else -- model, horizon, u_max, rate
limit, reference -- is shared.

Usage:
    python letter_mpc.py --shape rope_all_shapes/rope_C.json
    python letter_mpc.py --shape rope_all_shapes/rope_C.json --controller mppi
"""

### QP PATH WORKS FOR ROPE_C.JSON -- do not retune

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from mpc import load_exp_args
from rope_manip_env import RopeManipEnv, MM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="rope_all_shapes/rope_C.json")
    ap.add_argument("--controller", choices=["qp", "mppi"], default="qp")
    ap.add_argument("--results-dir", default="training_results")
    ap.add_argument("--exp-name", default="exp_simple_1788069999")
    ap.add_argument("--H", type=int, default=40, help="prediction horizon")
    ap.add_argument("--K", type=int, default=40, help="control horizon (K <= H)")
    ap.add_argument("--u-max", type=float, default=1.5,
                    help="mm/step box on the command; keep <= 0.75 * U_MAX_MM")
    ap.add_argument("--du-max", type=float, default=0.3,      # rate limit
                    help="max change in command between steps, mm")
    ap.add_argument("--r-weight", type=float, default=1e-1, help="input cost")
    ap.add_argument("--terminal-weight", type=float, default=50)
    ap.add_argument("--w-vel", type=float, default=0.05,
                    help="weight on velocity error in the latent cost")
    ap.add_argument("--steps", type=int, default=400, help="closed-loop MPC steps")
    ap.add_argument("--stall", type=int, default=40,
                    help="stop once RMSE has not improved for this many steps")
    ap.add_argument("--ref-rate", type=float, default=0,
                    help="mm/step the interpolated reference advances toward the goal. "
                         "0 disables interpolation and tracks the goal directly.")
    ap.add_argument("--project-rho", type=float, default=0.99,
                    help="rescale A to this spectral radius; 0 disables")

    # ---- MPPI only ---------------------------------------------------------
    ap.add_argument("--mppi-samples", type=int, default=1024,
                    help="K: command sequences sampled per step")
    ap.add_argument("--mppi-sigma", type=float, default=0.4,
                    help="exploration std on each command component, mm")
    ap.add_argument("--mppi-beta", type=float, default=0.02,
                    help="softmax temperature, in the units of the cost (mm^2). "
                         "Small = greedy (take the best sample), large = average "
                         "broadly. Tune by watching the printed effective sample size.")
    ap.add_argument("--mppi-gamma", type=float, default=1.0,
                    help="per-step discount on the stage cost")
    ap.add_argument("--mppi-decode-every", type=int, default=2,
                    help="decode only every n-th horizon step (the terminal step is "
                         "always decoded). 1 is exact but n times slower.")
    args = ap.parse_args()
    assert args.K <= args.H, "control horizon must not exceed prediction horizon"

    device = torch.device("cpu")
    ckpt_dir = Path(args.results_dir) / "checkpoints" / args.exp_name
    ta = load_exp_args(args.results_dir, args.exp_name)

    # ---- dataset: only for state_mean / state_std / cmd_scale ---------------
    dataset = RopeKoopmanDataset(
        ta["csv"], features=tuple(ta["features"]), horizon=1,
        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None,
    )
    mean = torch.as_tensor(dataset.state_mean[0])
    std = torch.as_tensor(dataset.state_std[0])
    pos_slice = dataset.pos_slice
    n_nodes = (pos_slice.stop - pos_slice.start) // 2
    cmd_scale = dataset.cmd_scale

    # ---- model -------------------------------------------------------------
    model = KoopmanModel(
        state_dim=dataset.state_dim,
        control_dim=dataset.control_dim,
        koopman_dim=ta["koopman_dim"],
        encoded_control_dim=ta.get("encoded_control_dim", 16),
        hidden_dim=ta["hidden_dim"],
        num_hidden=ta["num_hidden"],
        control_hidden_dim=ta.get("control_hidden_dim", 64),
        control_num_hidden=ta.get("control_num_hidden", 2),
        operator_bias=not ta.get("no_operator_bias", False),
    ).to(device)

    ld = lambda f: torch.load(ckpt_dir / f, map_location=device)
    model.encoder.load_state_dict(ld("koopman_enc.pt"))
    model.decoder.load_state_dict(ld("koopman_dec.pt"))
    model.operator.load_state_dict(ld("koopman_op.pt"))
    #model.input_encoder.load_state_dict(ld("koopman_input_enc.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A_np = model.operator.A.detach().cpu().numpy().astype(np.float64)
    B_np = model.operator.B.detach().cpu().numpy().astype(np.float64)

    B_np /= cmd_scale  # undo the training-time scaling of the command

    print(f"A {A_np.shape}  B {B_np.shape}  controller = {args.controller}")
    z_dim = A_np.shape[0]

    # ---- spectral projection ----------------------------------------------
    rho = np.abs(np.linalg.eigvals(A_np)).max()
    print(f"rho(A) = {rho:.6f}")
    if args.project_rho > 0 and rho > args.project_rho:
        A_np = A_np * (args.project_rho / rho)
        print(f"rho(A) after projection = {np.abs(np.linalg.eigvals(A_np)).max():.6f}")



    # ---- goal --------------------------------------------------------------
    J = json.load(open(args.shape))
    goal_pos = np.array(J["nodes"], dtype=np.float32)
    assert goal_pos.shape == (n_nodes, 2)

    x_goal = np.zeros(dataset.state_dim, dtype=np.float32)
    x_goal[dataset.feature_slices["pos"]] = goal_pos.reshape(-1)

    xg = (torch.as_tensor(x_goal).unsqueeze(0) - mean) / std

    with torch.no_grad():
        rec = model.decoder(model.encoder(xg)) * std + mean
    rec_pos = rec[0, pos_slice].reshape(n_nodes, 2).numpy()
    floor = np.sqrt(((rec_pos - goal_pos) ** 2).sum(1).mean())

    print(f"goal |z| max = {xg.abs().max().item():.3f}")
    print(f"goal recon RMSE = {floor:.3f} mm  <-- HARD FLOOR on achievable tracking")
    # if floor > 10.0:
    #     print("  !! goal is not representable; MPC cannot beat this. Fix coverage first.")

    # ---- env ---------------------------------------------------------------
    env = RopeManipEnv(n_nodes=n_nodes, base_length=0.30, base_radius=0.003,
                       youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01,
                       seed=0)
    env.reset()

    def measure():
        # Return the current state in mm, with the same feature order as the dataset.
        # [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]
        s = np.zeros(dataset.state_dim, dtype=np.float32)
        s[dataset.feature_slices["pos"]] = env.get_positions(mm=True)[:, :2].reshape(-1)
        if "vel" in dataset.feature_slices:
            s[dataset.feature_slices["vel"]] = env.get_velocities(mm=True)[:, :2].reshape(-1)
        return s

    def encode(state_mm):
        xn = (torch.as_tensor(state_mm).unsqueeze(0) - mean) / std
        with torch.no_grad():
            return model.encoder(xn).numpy().ravel().astype(np.float64)

    # ---- Jacobian-weighted cost -------------------------------------------
    # ||z - z_ref||^2 is not position error: the decoder is not an isometry, so the
    # solver can drive latent error down while mm error rises. Weighting by J^T J
    # makes the latent cost a first-order proxy for squared mm position error.
    z_ref_fixed = encode(x_goal)
    zg_t = torch.tensor(z_ref_fixed, dtype=torch.float32)

    vel_slice = dataset.feature_slices["vel"]
    w_vel = args.w_vel    # relative weight; velocities are mm/s, positions mm

    def dec_pv(z):
        out = model.decoder(z.unsqueeze(0))[0]
        return torch.cat([out[pos_slice], w_vel * out[vel_slice]])

    Jt = torch.autograd.functional.jacobian(dec_pv, zg_t)          # (120, z_dim)
    scale = np.concatenate([dataset.state_std[0, pos_slice],
                            dataset.state_std[0, vel_slice]])
    Jd = (Jt.numpy() * scale[:, None]).astype(np.float64)
    Q = Jd.T @ Jd
    Q += 1e-4 * (np.trace(Q) / z_dim) * np.eye(z_dim)
    Q /= (np.trace(Q) / z_dim)  

    
    print(f"Q normalized, cond {np.linalg.cond(Q):.2e}")
    print(f"Q: trace {np.trace(Q):.3e}, rank {np.linalg.matrix_rank(Jd)}")

    # ======================================================================= #
    # CONTROLLER: qp                                                          #
    # ======================================================================= #
    if args.controller == "qp":
        # ---- QP, built once and re-solved with new parameters each step ----
        opti = ca.Opti()

        # Variables
        Z = opti.variable(z_dim, args.H + 1)
        U = opti.variable(2, args.K)

        # Initial state and reference parameters
        z0_p = opti.parameter(z_dim)
        zref_p = opti.parameter(z_dim)

        # Constraints and cost
        opti.subject_to(Z[:, 0] == z0_p)
        cost = 0

        # Till control Horizon, K, we have a control input for each step.
        # After that, we hold the last input constant.

        for k in range(args.H):
            u_k = U[:, min(k, args.K - 1)]
              # zero-order hold past the control horizon
            opti.subject_to(Z[:, k + 1] == ca.DM(A_np) @ Z[:, k] +
                            ca.DM(B_np) @ u_k)

            e = Z[:, k + 1] - zref_p

            w = args.terminal_weight if k == args.H - 1 else 1.0

            cost += w * ca.bilin(ca.DM(Q), e, e)

        for k in range(args.K):
            cost += args.r_weight * ca.dot(U[:, k], U[:, k])
            opti.subject_to(opti.bounded(-args.u_max, U[:, k], args.u_max))

        for k in range(1, args.K):
            opti.subject_to(opti.bounded(-args.du_max, U[:, k] - U[:, k-1], args.du_max))

        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 200})

        def control(t, z0, z_ref, ref_pos):
            opti.set_value(z0_p, z0)
            opti.set_value(zref_p, z_ref)
            try:
                sol = opti.solve()
                u = np.array(sol.value(U))[:, 0]
                opti.set_initial(Z, sol.value(Z))
                opti.set_initial(U, sol.value(U))
            except RuntimeError:
                u = np.array(opti.debug.value(U))[:, 0]
                print(f"  step {t}: solver failed, using debug iterate")
            return np.clip(u, -args.u_max, args.u_max), {}

    # ======================================================================= #
    # CONTROLLER: mppi                                                        #
    # ======================================================================= #
    else:
        At = torch.tensor(A_np, dtype=torch.float32)
        Bt = torch.tensor(B_np, dtype=torch.float32)
        pstd = torch.as_tensor(dataset.state_std[0, pos_slice])
        pmean = torch.as_tensor(dataset.state_mean[0, pos_slice])
        vstd = torch.as_tensor(dataset.state_std[0, vel_slice])
        vmean = torch.as_tensor(dataset.state_mean[0, vel_slice])
        disc = args.mppi_gamma ** torch.arange(args.H, dtype=torch.float32)

        Kn = args.mppi_samples
        # Which horizon steps get decoded. Decoding is the only expensive part of
        # a rollout (the dynamics are a matmul), and intermediate shapes matter
        # far less than the terminal one, so subsample -- but never skip the last.
        dec_steps = sorted(set(list(range(0, args.H, args.mppi_decode_every))
                               + [args.H - 1]))
        print(f"MPPI: K={Kn}  sigma={args.mppi_sigma}  beta={args.mppi_beta}  "
              f"decoding {len(dec_steps)}/{args.H} horizon steps")

        U_nom = torch.zeros(args.H, 2)

        @torch.no_grad()
        def mppi_plan(z0, goal_t):
            nonlocal U_nom
            z = torch.tensor(z0, dtype=torch.float32).expand(Kn, -1).contiguous()

            # sample, clip to the box, then project onto the rate limit so the
            # samples respect the same constraints the QP path enforces
            eps = args.mppi_sigma * torch.randn(Kn, args.H, 2)
            Us = (U_nom.unsqueeze(0) + eps).clamp(-args.u_max, args.u_max)
            dU = torch.diff(Us, dim=1).clamp(-args.du_max, args.du_max)
            Us = torch.cat([Us[:, :1], Us[:, :1] + dU.cumsum(1)], dim=1)
            Us = Us.clamp(-args.u_max, args.u_max)

            S = torch.zeros(Kn)
            for j in range(args.H):
                z = z @ At.T + Us[:, j] @ Bt.T
                if j not in dec_steps:
                    continue
                out = model.decoder(z)
                pos = out[:, pos_slice] * pstd + pmean               # (K, 60) mm
                err = pos.reshape(Kn, -1, 2) - goal_t                # (K, N, 2)
                c = err.pow(2).sum(-1).mean(-1)                      # mm^2 per sample
                if w_vel > 0:
                    vel = out[:, vel_slice] * vstd + vmean           # (K, 60) mm/s
                    # goal velocity is zero: the target is a shape held at rest
                    c = c + w_vel * vel.pow(2).reshape(Kn, -1, 2).sum(-1).mean(-1)
                w = args.terminal_weight if j == args.H - 1 else 1.0
                S = S + w * disc[j] * c

            S = S + args.r_weight * Us.pow(2).sum(dim=(1, 2))
            beta = args.mppi_beta * float(S.std() + 1e-9)
            wts = torch.softmax(-(S - S.min()) / beta, dim=0)
            U_nom = (wts[:, None, None] * Us).sum(0)                 # (H, 2)

            # effective sample size: ~1 means the average collapsed onto a single
            # rollout (beta too small); ~K means it is averaging noise (too large)
            ess = float(1.0 / wts.pow(2).sum())
            return U_nom[0].numpy().astype(np.float64), dict(
                ess=ess, cost_min=float(S.min()), cost_mean=float(S.mean()))

        def control(t, z0, z_ref, ref_pos):
            goal_t = torch.as_tensor(ref_pos, dtype=torch.float32)
            u, info = mppi_plan(z0, goal_t)
            u = np.clip(u, -args.u_max, args.u_max)
            # shift the nominal sequence for the next step (receding horizon)
            U_shift = torch.roll(U_nom, -1, dims=0)
            U_shift[-1] = 0.0
            U_nom.copy_(U_shift)
            return u, info

    # ---- reachability diagnostic -------------------------------------------
    # B spans only 2 of z_dim latent directions. Over the horizon the reachable
    # set is span{Bu, A Bu, ..., A^(H-1) Bu} -- at most 2H dims out of 256. If the
    # goal error lies mostly outside that subspace, the controller is CORRECT to
    # emit tiny commands: large ones would not reduce the cost. That is a horizon
    # or reference problem, not a u_max problem.
    z0_d = encode(measure())
    zref_d = encode(x_goal)
    e_d = zref_d - z0_d
    print(f"\n||z_ref - z0|| = {np.linalg.norm(e_d):.4f}")
    print(f"||Bu||_2 * u_max = {np.linalg.norm(B_np, 2) * args.u_max:.4f}")

    Kr = np.concatenate([np.linalg.matrix_power(A_np, j) @ B_np for j in range(args.H)], axis=1)
    Qr, _ = np.linalg.qr(Kr)
    frac_H = np.linalg.norm(Qr.T @ e_d) / np.linalg.norm(e_d)
    Q1, _ = np.linalg.qr(B_np)
    frac_1 = np.linalg.norm(Q1.T @ e_d) / np.linalg.norm(e_d)
    print(f"frac of error in span(Bu)        : {frac_1:.4f}")
    print(f"frac of error in {args.H}-step reachable: {frac_H:.4f}")
    if frac_H < 0.10:
        print("  -> goal is largely UNREACHABLE within the horizon; "
              "use --ref-rate 1.2 rather than a larger --u-max")

    # ---- closed loop -------------------------------------------------------
    start_pos = env.get_positions(mm=True)[:, :2].copy()
    total_travel = np.linalg.norm(goal_pos[0] - start_pos[0])
    print(f"\nnode-0 must travel {total_travel:.1f} mm; at {args.u_max} mm/step that is "
          f">= {int(total_travel / args.u_max)} steps")

    hist = []
    best_rmse, best_shape, best_t, stall = np.inf, None, 0, 0
    for t in range(args.steps):
        state = measure()
        z0 = encode(state)

        # Interpolated reference: a distant goal makes every solve saturate at the
        # box, so walk the setpoint toward it at ref-rate mm/step instead.
        if args.ref_rate > 0:
            alpha = min(1.0, (t * args.ref_rate) / max(total_travel, 1e-6))
            ref_pos = (1 - alpha) * start_pos + alpha * goal_pos
            x_ref = np.zeros(dataset.state_dim, dtype=np.float32)
            x_ref[dataset.feature_slices["pos"]] = ref_pos.reshape(-1)
            z_ref = encode(x_ref)
        else:
            ref_pos = goal_pos
            z_ref = encode(x_goal)

        u, info = control(t, z0, z_ref, ref_pos)

        env.drive_step(u / MM)
        if env.is_exploded():
            print(f"  step {t}: sim exploded"); break

        cur = env.get_positions(mm=True)[:, :2]
        rmse = np.sqrt(((cur - goal_pos) ** 2).sum(1).mean())
        hist.append(rmse)
        if t % 20 == 0:
            extra = (f"  ess {info['ess']:6.1f}  Smin {info['cost_min']:.1f}"
                     if "ess" in info else "")
            print(f"  step {t:4d}  |u| {np.linalg.norm(u):.3f} mm  "
                  f"RMSE-to-goal {rmse:7.2f} mm{extra}")

        if rmse < best_rmse - 1e-6:
            best_rmse, best_shape, best_t, stall = rmse, cur.copy(), t, 0
        else:
            stall += 1
            if stall > args.stall:
                print(f"  no improvement for {args.stall} steps -- stopping at t={t}; "
                      f"best {best_rmse:.2f} mm at t={best_t}")
                break

    final = env.get_positions(mm=True)[:, :2]
    vmax = np.abs(env.get_velocities(mm=True)[:, :2]).max()
    print(f"  step {t:4d}  |u| {np.linalg.norm(u):.3f}  "
              f"RMSE {rmse:7.2f} mm  max|v| {vmax:7.2f} mm/s")
 
 
    print(f"\n[{args.controller}] final RMSE {hist[-1]:.2f} mm   "
          f"best {best_rmse:.2f} mm at t={best_t}   "
          f"(reconstruction floor {floor:.2f} mm)")
 
    # ---- plots -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], 'o-', color='g', label='goal')
    ax[0].plot(final[:, 0], final[:, 1], 's-', color='r', label='MPC final')
    if best_shape is not None:
        ax[0].plot(best_shape[:, 0], best_shape[:, 1], '^--', color='orange',
                   alpha=0.7, label=f'best ({best_rmse:.1f} mm)')
    ax[0].plot(start_pos[:, 0], start_pos[:, 1], ':', color='gray', label='start')
    ax[0].axis('equal'); ax[0].grid(True); ax[0].legend()
    ax[0].set_xlabel('X (mm)'); ax[0].set_ylabel('Y (mm)')
    ax[0].set_title(f"{Path(args.shape).stem} -- {args.controller}")
    ax[1].plot(hist)
    ax[1].axhline(floor, ls='--', color='r', label='recon floor')
    ax[1].set_xlabel('MPC step'); ax[1].set_ylabel('RMSE to goal (mm)')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout(); plt.show()
 
 
if __name__ == "__main__":
    main()