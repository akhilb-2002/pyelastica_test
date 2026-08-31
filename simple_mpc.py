"""
letter_mpc.py -- Koopman MPC for tracing letter shapes with the rope.

The optimisation runs entirely in latent space. The encoder is an MLP, so it
cannot go inside the solver; instead the measured state is encoded ONCE per MPC
step (outside CasADi) and becomes a parameter. Inside the solver the dynamics are

    z_{k+1} = A z_k + Bu u_k + c0

which is linear in the raw mm command, so the whole OCP is a QP and the command
box |u| <= U_MAX sits directly on the decision variable.

That linearity depends on the input encoder having been trained with 0 hidden
layers, i.e. a single Linear(2, encoded_dim). Then

    B eta(u/s) = B (W u/s + b) = (B W / s) u + B b  =:  Bu u + c0

and the lift disappears into a constant matrix. The script asserts this.

Tracking is on the LATENT error z - z_goal, not on positions, because putting the
decoder in the solver would destroy the QP. This is the main approximation here:
latent distance weights directions by whatever the encoder learned, so the cost
falling does not by itself mean the shape converged. mm RMSE is reported
separately every step -- trust that number, not the cost.

Usage:
    python letter_mpc.py --shape rope_all_shapes/rope_C_reachable.json
"""

### THIS WORK FOR ROPE_C.JSON!!!

import argparse
import json
from pathlib import Path
from xml.parsers.expat import model

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from mpc import load_exp_args
from rope_manip_env import RopeManipEnv, MM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="rope_all_shapes/rope_I.json")
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

    print(f"A {A_np.shape}  B {B_np.shape}")
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

    # ---- QP, built once and re-solved with new parameters each step --------
    
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


    # ---- reachability diagnostic -------------------------------------------
    # B spans only 2 of z_dim latent directions. Over the horizon the reachable
    # set is span{Bu, A Bu, ..., A^(H-1) Bu} -- at most 2H dims out of 256. If the
    # goal error lies mostly outside that subspace, the QP is CORRECT to emit tiny
    # commands: large ones would not reduce the cost. That is a horizon/reference
    # problem, not a u_max problem, and raising the box will not fix it.
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
            z_ref = encode(x_goal)

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

        u = np.clip(u, -args.u_max, args.u_max)

        env.drive_step(u / MM)
        if env.is_exploded():
            print(f"  step {t}: sim exploded"); break

        cur = env.get_positions(mm=True)[:, :2]
        rmse = np.sqrt(((cur - goal_pos) ** 2).sum(1).mean())
        hist.append(rmse)
        if t % 20 == 0:
            print(f"  step {t:4d}  |u| {np.linalg.norm(u):.3f} mm  "
                  f"RMSE-to-goal {rmse:7.2f} mm")

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
 
 
    print(f"\nfinal RMSE {hist[-1]:.2f} mm   best {best_rmse:.2f} mm at t={best_t}   "
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
    ax[1].plot(hist)
    ax[1].axhline(floor, ls='--', color='r', label='recon floor')
    ax[1].set_xlabel('MPC step'); ax[1].set_ylabel('RMSE to goal (mm)')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout(); plt.show()
 
 
if __name__ == "__main__":
    main()
 