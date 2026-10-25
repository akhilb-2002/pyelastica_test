"""
letter_qp.py -- single-shot QP trajectory planning for rope letter shapes.

Unlike the receding-horizon MPC, this solves ONCE for the whole T-step command
sequence and then executes it open-loop:

    min_{u_0..u_{T-1}}  sum_t (z_t - z_ref)^T Q (z_t - z_ref) + sum_t u_t^T R u_t
    s.t.                z_0     = phi(x_0)
                        z_{t+1} = A z_t + B u_t
                        u_min <= u_t <= u_max

The point of the long horizon is GLOBAL ROUTING. A hooked letter (J, S) needs the
rope to move AWAY from the goal before it can reach it, and a short-horizon
controller cannot see past the detour. A T-step plan can.

CONDENSED FORM
--------------
Eliminating the latent states via  z_t = A^t z_0 + sum_{i<t} A^{t-1-i} B u_i
leaves only the 2T command variables. Writing y = J z (the decoder Jacobian, so y
is in millimetres) the cost becomes a plain least-squares problem in u:

    min_u ||Gamma u - b||^2 + lambda ||u||^2 ,   |u| <= u_max

    where Gamma is a tall matrix of size (len(checks)*2N, 2T) and b is a vector of
    size (len(checks)*2N). The checks are a subsample of timesteps plus the terminal step, and N is the number of rope nodes. The
    Hessian H = 2 (Gamma^T Gamma + lambda I) is positive definite, so this is a

which is a dense QP of size 2T -- 800 variables at T = 400, solved in well under
a second, with a Hessian that does not change between shapes of the same length.

The running cost is evaluated on a SUBSAMPLE of timesteps (--cost-every) plus the
terminal step. Two reasons: it keeps Gamma small, and a full running cost charges
the optimiser for the very detour a hooked letter requires. Keep the terminal
weight dominant unless the planned path visibly wanders.

Usage:
    python letter_qp.py --shape rope_all_shapes/rope_J.json --T 400
"""

import argparse
import ast
import json
from pathlib import Path

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="rope_all_shapes/rope_J.json")
    ap.add_argument("--results-dir", default="training_results")
    ap.add_argument("--exp-name", default="exp_simple_1788269928")
    ap.add_argument("--out-dir", default="koopman_qp_outputs")
    ap.add_argument("--T", type=int, default=400, help="plan length in steps")
    ap.add_argument("--u-max", type=float, default=1.5, help="mm/step command box")
    ap.add_argument("--du-max", type=float, default=0.3,
                    help="max change between consecutive commands, mm (0 disables)")
    ap.add_argument("--r-weight", type=float, default=1e-2, help="input cost lambda")
    ap.add_argument("--terminal-weight", type=float, default=100.0)
    ap.add_argument("--cost-every", type=int, default=20,
                    help="evaluate the running cost every n-th step (terminal always)")
    ap.add_argument("--settle", type=int, default=600)
    ap.add_argument("--project-rho", type=float, default=0.99)
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt_dir = Path(args.results_dir) / "checkpoints" / args.exp_name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_exp_args(results_dir, exp_name):
        p = Path(results_dir) / f"{exp_name}.txt"
        return ast.literal_eval(p.read_text().splitlines()[0].split("Args: ", 1)[1])

    ta = load_exp_args(args.results_dir, args.exp_name)

    dataset = RopeKoopmanDataset(
        ta["csv"], features=tuple(ta["features"]), horizon=1,
        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None,
    )
    mean = torch.as_tensor(dataset.state_mean[0])
    std = torch.as_tensor(dataset.state_std[0])
    n_nodes = dataset.state_dim // 2
    cmd_scale = dataset.cmd_scale

    model = KoopmanModel(
        state_dim=dataset.state_dim, control_dim=dataset.control_dim,
        koopman_dim=ta["koopman_dim"], hidden_dim=ta["hidden_dim"],
        num_hidden=ta["num_hidden"],
        operator_bias=not ta.get("no_operator_bias", False),
    ).to(device)
    ld = lambda f: torch.load(ckpt_dir / f, map_location=device)
    model.encoder.load_state_dict(ld("koopman_enc.pt"))
    model.decoder.load_state_dict(ld("koopman_dec.pt"))
    model.operator.load_state_dict(ld("koopman_op.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A = model.operator.A.detach().cpu().numpy().astype(np.float64)
    B = model.operator.B.detach().cpu().numpy().astype(np.float64) / cmd_scale
    nz = A.shape[0]
    rho = np.abs(np.linalg.eigvals(A)).max()
    if args.project_rho > 0 and rho > args.project_rho:
        A = A * (args.project_rho / rho)
    print(f"nz {nz} | rho(A) {rho:.6f} -> {np.abs(np.linalg.eigvals(A)).max():.6f}")

    # How long the initial condition survives. With rho < 1 the influence of z_0
    # decays as rho^t, so beyond this horizon the terminal state depends only on
    # recent commands -- planning further ahead than this buys nothing for the
    # terminal cost, and the effective horizon is much shorter than T.
    if rho < 1:
        t_mem = int(np.ceil(np.log(1e-3) / np.log(min(rho, 0.999))))
        print(f"initial-condition memory: rho^t < 1e-3 after t = {t_mem} steps")

    def encode(pos_mm):
        x = torch.as_tensor(np.asarray(pos_mm, np.float32).reshape(-1)).unsqueeze(0)
        with torch.no_grad():
            return model.encoder((x - mean) / std).numpy().ravel().astype(np.float64)

    def decode(z):
        with torch.no_grad():
            x = model.decoder(torch.tensor(z, dtype=torch.float32)[None]) * std + mean
        return x[0].reshape(n_nodes, 2).numpy()

    # ---- goal --------------------------------------------------------------
    J = json.load(open(args.shape))
    goal_pos = np.array(J["nodes"], dtype=np.float32)
    name = Path(args.shape).stem
    rest_len = J["rope_length"] / (n_nodes - 1)

    z_ref = encode(goal_pos)
    floor = float(np.sqrt(((decode(z_ref) - goal_pos) ** 2).sum(1).mean()))
    print(f"{name}: goal recon RMSE {floor:.3f} mm")

    # ---- decoder Jacobian: latent error -> mm error -------------------------
    zg_t = torch.tensor(z_ref, dtype=torch.float32)
    Jt = torch.autograd.functional.jacobian(
        lambda z: model.decoder(z.unsqueeze(0))[0], zg_t).numpy()
    Jd = (Jt * dataset.state_std[0][:, None]).astype(np.float64)      # (2N, nz), mm

    # ---- env ---------------------------------------------------------------
    env = RopeManipEnv(n_nodes=n_nodes, base_length=0.30, base_radius=0.003,
                       youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01,
                       seed=0)
    env.reset()
    start_pos = env.get_positions(mm=True)[:, :2].copy()
    z0 = encode(start_pos)

    T = args.T
    travel = float(np.linalg.norm(goal_pos[0] - start_pos[0]))
    print(f"node-0 travel {travel:.1f} mm; at {args.u_max} mm/step needs "
          f">= {int(travel / args.u_max)} steps (planning {T})")
    if T < travel / args.u_max:
        print("  !! T is below the travel-limited minimum; the plan cannot arrive")

    # ---- condensed prediction matrices -------------------------------------
    # Cost checkpoints: a subsample plus the terminal step. A full running cost
    # would also charge the optimiser for the detour a hooked letter needs.
    checks = sorted(set(list(range(args.cost_every, T, args.cost_every)) + [T]))
    print(f"cost evaluated at {len(checks)} checkpoints (terminal weight "
          f"{args.terminal_weight})")

    # impulse response blocks M_k = A^k B, k = 0 .. T-1
    print("building condensed matrices ...")
    Mk = np.empty((T, nz, 2))
    P = B.copy()
    for k in range(T):
        Mk[k] = P
        P = A @ P

    Apow = {0: np.eye(nz)}
    Pz = np.eye(nz)
    for t in range(1, T + 1):
        Pz = A @ Pz
        if t in checks:
            Apow[t] = Pz.copy()

    # y_t = Jd z_t (mm). Row block for checkpoint t: Gamma[t] u + Jd A^t z0
    rows_G, rows_b, wts = [], [], []
    for t in checks:
        G = np.zeros((Jd.shape[0], 2 * T))
        for i in range(t):
            G[:, 2 * i:2 * i + 2] = Jd @ Mk[t - 1 - i]
        w = args.terminal_weight if t == T else 1.0
        rows_G.append(np.sqrt(w) * G)
        rows_b.append(np.sqrt(w) * (Jd @ (z_ref - Apow[t] @ z0)))
        wts.append(w)

    Gam = np.vstack(rows_G)                      # (len(checks)*2N, 2T)
    bvec = np.concatenate(rows_b)
    print(f"Gamma {Gam.shape}")

    # ---- QP ----------------------------------------------------------------
    # min ||Gam u - b||^2 + lambda ||u||^2, |u| <= u_max, |du| <= du_max
    H = 2.0 * (Gam.T @ Gam + args.r_weight * np.eye(2 * T))
    g = -2.0 * (Gam.T @ bvec)

    opti = ca.Opti()
    U = opti.variable(2 * T)
    opti.minimize(0.5 * ca.bilin(ca.DM(H), U, U) + ca.dot(ca.DM(g), U))
    opti.subject_to(opti.bounded(-args.u_max, U, args.u_max))
    if args.du_max > 0:
        for t in range(1, T):
            du = U[2 * t:2 * t + 2] - U[2 * (t - 1):2 * (t - 1) + 2]
            opti.subject_to(opti.bounded(-args.du_max, du, args.du_max))
    opti.set_initial(U, np.zeros(2 * T))
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "sb": "yes", "max_iter": 500})

    print("solving ...")
    sol = opti.solve()
    u_plan = np.array(sol.value(U)).reshape(T, 2)
    print(f"|u| mean {np.linalg.norm(u_plan, axis=1).mean():.3f}  "
          f"max {np.linalg.norm(u_plan, axis=1).max():.3f} mm/step")

    # ---- what the MODEL thinks the plan achieves ---------------------------
    z = z0.copy()
    for t in range(T):
        z = A @ z + B @ u_plan[t]
    model_pos = decode(z)
    model_rmse = float(np.sqrt(((model_pos - goal_pos) ** 2).sum(1).mean()))
    print(f"model-predicted final RMSE: {model_rmse:.2f} mm")

    # ---- execute open-loop, then settle ------------------------------------
    hist, shapes = [], []
    for t in range(T):
        env.drive_step(u_plan[t] / MM)
        if env.is_exploded():
            print(f"  exploded at step {t}"); break
        cur = env.get_positions(mm=True)[:, :2]
        shapes.append(cur.copy())
        hist.append(float(np.sqrt(((cur - goal_pos) ** 2).sum(1).mean())))
        if t % 50 == 0:
            print(f"  step {t:4d}  RMSE {hist[-1]:7.2f} mm")

    n_driven = len(hist)
    print(f"executed: RMSE {hist[-1]:.2f} mm")

    for _ in range(args.settle):
        env.drive_step(np.zeros(2))
        if env.is_exploded():
            break
        cur = env.get_positions(mm=True)[:, :2]
        shapes.append(cur.copy())
        hist.append(float(np.sqrt(((cur - goal_pos) ** 2).sum(1).mean())))

    final = env.get_positions(mm=True)[:, :2]
    vfin = float(np.abs(env.get_velocities(mm=True)[:, :2]).max())
    fseg = np.linalg.norm(np.diff(final, axis=0), axis=1)
    print(f"settled:  RMSE {hist[-1]:.2f} mm  max|v| {vfin:.4f} mm/s  "
          f"seg strain {100 * (fseg.max() - rest_len) / rest_len:+.2f}%")

    # The gap between what the model predicted and what the sim did IS the
    # open-loop drift -- the quantity the closed-loop re-solve would remove.
    print(f"\nopen-loop drift: model said {model_rmse:.2f} mm, "
          f"sim gave {hist[n_driven - 1]:.2f} mm  "
          f"(delta {abs(hist[n_driven - 1] - model_rmse):.2f} mm)")

    tip_path = np.stack(shapes)[:n_driven, 0, :]
    np.savez_compressed(out_dir / f"{name}_qp.npz",
                        commands_mm=u_plan[:n_driven], tip_path_mm=tip_path,
                        shapes_mm=np.stack(shapes), rmse_mm=np.array(hist),
                        goal_nodes_mm=goal_pos, settled_nodes_mm=final,
                        model_pred_nodes_mm=model_pos, n_driven=n_driven,
                        recon_floor_mm=floor, model_rmse_mm=model_rmse)

    # ---- plots -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], 'o-', color='g', label='goal')
    ax[0].plot(final[:, 0], final[:, 1], 's-', color='r',
               label=f'settled ({hist[-1]:.1f} mm)')
    ax[0].plot(model_pos[:, 0], model_pos[:, 1], 'x--', color='b', alpha=0.6,
               label=f'model predicted ({model_rmse:.1f} mm)')
    ax[0].plot(tip_path[:, 0], tip_path[:, 1], '-', color='tab:blue', lw=1,
               alpha=0.5, label='tip path')
    ax[0].plot(start_pos[:, 0], start_pos[:, 1], ':', color='gray', label='start')
    ax[0].axis('equal'); ax[0].grid(True); ax[0].legend(); ax[0].set_title(name)
    ax[0].set_xlabel('X (mm)'); ax[0].set_ylabel('Y (mm)')

    ax[1].plot(hist)
    ax[1].axvline(n_driven, ls=':', color='k', label='settle')
    ax[1].axhline(floor, ls='--', color='r', label='recon floor')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('RMSE to goal (mm)')
    ax[1].grid(True); ax[1].legend()

    ax[2].plot(u_plan[:, 0], label='u_x')
    ax[2].plot(u_plan[:, 1], label='u_y')
    ax[2].axhline(args.u_max, ls='--', color='r')
    ax[2].axhline(-args.u_max, ls='--', color='r')
    ax[2].set_xlabel('step'); ax[2].set_ylabel('command (mm/step)')
    ax[2].grid(True); ax[2].legend()

    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_qp.png", dpi=140)
    plt.show()


if __name__ == "__main__":
    main()