"""
letter_mpc_bilinear.py -- receding-horizon MPC on the BILINEAR Koopman model.

    z_{k+1} = A z_k + B u_k + (sum_j u_k^(j) N_j) z_k

The bilinear term is non-convex in (z, u) jointly, but it is bilinear: fixing one
side makes it linear in the other. At every MPC step the nominal latent trajectory
z^nom is obtained by rolling the TRUE model forward under the shifted previous
solution, and then

    (sum_j u^(j) N_j) z_k^nom  =  M_k u_k ,   M_k = [N_0 z_k^nom , N_1 z_k^nom]

so the prediction becomes z_{k+1} = A z_k + (B + M_k) u_k -- linear, with a
time-varying input matrix. That is an ordinary QP, solved with the usual box and
rate constraints. Optionally the linearisation is repeated once or twice within
the step (--inner) so the QP is solved about its own solution.

This is the receding-horizon counterpart of the one-shot OCP: re-linearising every
step keeps the approximation local, which is exactly where a bilinear expansion is
accurate, and the feedback absorbs the model error that made the open-loop plan
drift.

Tracking is on the latent error weighted by J^T J (J = decoder Jacobian at the
goal), so the latent cost is a first-order proxy for millimetres -- an
identity-weighted latent cost lets the solver drive latent error down while the
physical error rises.

Usage:
    python letter_mpc_bilinear.py --exp-name exp_XXXX --shape rope_all_shapes/rope_C.json
    python letter_mpc_bilinear.py --exp-name exp_XXXX --all
"""

import argparse
import ast
import json
from pathlib import Path

import casadi as ca
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM


def run_shape(args, shape_path, ctx):
    dataset, model, env = ctx["dataset"], ctx["model"], ctx["env"]
    mean, std = ctx["mean"], ctx["std"]
    pos_slice, has_vel = ctx["pos_slice"], ctx["has_vel"]
    n_nodes, nz = ctx["n_nodes"], ctx["nz"]
    A, B, N = ctx["A"], ctx["B"], ctx["N"]

    name = Path(shape_path).stem
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}")

    def to_state(pos, vel=None):
        s = np.zeros(dataset.state_dim, dtype=np.float32)
        s[dataset.feature_slices["pos"]] = np.asarray(pos)[:, :2].reshape(-1)
        if has_vel:
            v = np.zeros_like(np.asarray(pos)) if vel is None else np.asarray(vel)
            s[dataset.feature_slices["vel"]] = v[:, :2].reshape(-1)
        return s

    def encode(pos, vel=None):
        with torch.no_grad():
            x = (torch.as_tensor(to_state(pos, vel))[None] - mean) / std
            return model.encoder(x).numpy().ravel().astype(np.float64)

    def decode(z):
        with torch.no_grad():
            x = model.decoder(torch.tensor(z, dtype=torch.float32)[None]) * std + mean
        return x[0, pos_slice].reshape(n_nodes, 2).numpy()

    def step_model(z, u):
        """One TRUE bilinear step. Used for the nominal rollout, never in the QP."""
        zn = A @ z + B @ u
        if N is not None:
            zn = zn + np.einsum("j,jmn,n->m", u, N, z)
        return zn

    def measure():
        return (env.get_positions(mm=True)[:, :2],
                env.get_velocities(mm=True)[:, :2] if has_vel else None)

    def rmse_now():
        return float(np.sqrt(((env.get_positions(mm=True)[:, :2] - goal_pos) ** 2)
                             .sum(1).mean()))

    def vmax_now():
        return float(np.abs(env.get_velocities(mm=True)[:, :2]).max())

    # ---- goal --------------------------------------------------------------
    J = json.load(open(shape_path))
    goal_pos = np.array(J["nodes"], dtype=np.float32)
    rest_len = J["rope_length"] / (n_nodes - 1)
    z_ref = encode(goal_pos)                       # target held at rest
    rec = decode(z_ref)
    floor = float(np.sqrt(((rec - goal_pos) ** 2).sum(1).mean()))
    xg = (torch.as_tensor(to_state(goal_pos))[None] - mean) / std
    print(f"goal |z| max {xg.abs().max().item():.3f} | recon RMSE {floor:.3f} mm")

    env.reset()

    # ---- Jacobian-weighted cost -------------------------------------------
    Jt = torch.autograd.functional.jacobian(
        lambda z: model.decoder(z.unsqueeze(0))[0, pos_slice],
        torch.tensor(z_ref, dtype=torch.float32)).numpy()
    Jd = (Jt * dataset.state_std[0, pos_slice][:, None]).astype(np.float64)
    Q = Jd.T @ Jd
    Q = Q / (np.trace(Q) / nz)                     # unit mean eigenvalue
    Q = Q + 1e-2 * np.eye(nz)                      # rank(Q) <= 2N << nz
    print(f"Q cond {np.linalg.cond(Q):.2e}")

    # ---- QP, built ONCE. B_k enters as a parameter because the linearisation
    #      point changes every step, so B + M_k is not a constant. -----------
    H = args.H
    opti = ca.Opti()
    Z = opti.variable(nz, H + 1)
    U = opti.variable(2, H)
    z0_p = opti.parameter(nz)
    zref_p = opti.parameter(nz)
    uprev_p = opti.parameter(2)
    Bt_p = [opti.parameter(nz, 2) for _ in range(H)]

    opti.subject_to(Z[:, 0] == z0_p)
    cost = 0
    for k in range(H):
        opti.subject_to(Z[:, k + 1] == ca.DM(A) @ Z[:, k] + Bt_p[k] @ U[:, k])
        e = Z[:, k + 1] - zref_p
        w = args.terminal_weight if k == H - 1 else 1.0
        cost += w * ca.bilin(ca.DM(Q), e, e)
        cost += args.r_weight * ca.dot(U[:, k], U[:, k])
        opti.subject_to(opti.bounded(-args.u_max, U[:, k], args.u_max))

    # rate limit; the k=0 constraint ties the plan to the command actually applied
    opti.subject_to(opti.bounded(-args.du_max, U[:, 0] - uprev_p, args.du_max))
    for k in range(1, H):
        opti.subject_to(opti.bounded(-args.du_max, U[:, k] - U[:, k - 1], args.du_max))

    opti.minimize(cost)
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "sb": "yes", "max_iter": 200})

    # ---- closed loop -------------------------------------------------------
    start = env.get_positions(mm=True)[:, :2].copy()
    travel = float(np.linalg.norm(goal_pos[0] - start[0]))
    min_steps = int(travel / args.u_max)
    # Hooked shapes (J, S) must move AWAY from the goal before they can reach it,
    # so a monotonic-improvement test applied early kills the run before the rope
    # has curled at all. Scale the grace period to the travel the shape needs.
    stall_after = max(args.stall_after, int(1.5 * min_steps))
    print(f"travel {travel:.1f} mm -> >= {min_steps} steps | "
          f"stall test after t={stall_after}")

    U_nom = np.zeros((H, 2))
    u_prev = np.zeros(2)
    hist, shapes, vhist, u_list = [], [], [], []
    best_rmse, best_shape, best_t, stall = np.inf, None, 0, 0
    exploded = False

    for t in range(args.steps):
        pos, vel = measure()
        z0 = encode(pos, vel)

        u_applied = None
        for _ in range(max(1, args.inner)):
            # nominal latent trajectory under the current guess, through the TRUE
            # bilinear model -- this is the point the QP linearises about
            zc, Bts = z0.copy(), []
            for k in range(H):
                Bk = B.copy()
                if N is not None:
                    Bk = Bk + np.stack([N[j] @ zc for j in range(2)], axis=1)
                Bts.append(Bk)
                zc = step_model(zc, U_nom[k])

            opti.set_value(z0_p, z0)
            opti.set_value(zref_p, z_ref)
            opti.set_value(uprev_p, u_prev)
            for k in range(H):
                opti.set_value(Bt_p[k], Bts[k])
            try:
                sol = opti.solve()
                U_sol = np.array(sol.value(U)).T
                opti.set_initial(Z, sol.value(Z))
                opti.set_initial(U, sol.value(U))
            except RuntimeError:
                U_sol = np.array(opti.debug.value(U)).T
            U_nom = U_sol
            u_applied = U_sol[0]

        u = np.clip(u_applied, -args.u_max, args.u_max)
        u_prev = u.copy()
        u_list.append(u.copy())

        env.drive_step(u / MM)
        if env.is_exploded():
            print(f"  step {t}: exploded"); exploded = True; break

        cur = env.get_positions(mm=True)[:, :2]
        r = rmse_now()
        hist.append(r); shapes.append(cur.copy()); vhist.append(vmax_now())

        if t % 40 == 0:
            print(f"  step {t:4d}  |u| {np.linalg.norm(u):.3f} mm  RMSE {r:7.2f} mm")

        if r < best_rmse - 1e-6:
            best_rmse, best_shape, best_t, stall = r, cur.copy(), t, 0
        else:
            stall += 1
            if t > stall_after and stall > args.stall:
                print(f"  no improvement for {args.stall} steps -- stop at t={t}; "
                      f"best {best_rmse:.2f} mm at t={best_t}")
                break

        # shift the nominal sequence for the next step (receding horizon)
        U_nom = np.vstack([U_nom[1:], np.zeros((1, 2))])

    print(f"driven: last {hist[-1]:.2f} | best {best_rmse:.2f} mm at t={best_t}")

    # ---- rewind to the best step, then settle ------------------------------
    # The stall counter fires N steps AFTER the minimum and the controller keeps
    # driving in between, so settling from the last state locks in the overshoot.
    # PyElastica has no rewind; replay is deterministic, so re-drive to best_t.
    if not args.no_rewind and best_t < len(hist) - 1:
        print(f"rewinding to t={best_t} ...")
        env.reset()
        hist, shapes, vhist = [], [], []
        for k in range(best_t + 1):
            env.drive_step(u_list[k] / MM)
            shapes.append(env.get_positions(mm=True)[:, :2].copy())
            hist.append(rmse_now()); vhist.append(vmax_now())
        print(f"  replayed to {hist[-1]:.2f} mm (expected {best_rmse:.2f})")
    n_driven = len(hist)

    for _ in range(args.settle):
        env.drive_step(np.zeros(2))
        if env.is_exploded():
            break
        shapes.append(env.get_positions(mm=True)[:, :2].copy())
        hist.append(rmse_now()); vhist.append(vmax_now())

    final = env.get_positions(mm=True)[:, :2]
    fseg = np.linalg.norm(np.diff(final, axis=0), axis=1)
    print(f"settled: {hist[-1]:.2f} mm | max|v| {vhist[-1]:.4f} mm/s | "
          f"seg strain {100 * (fseg.max() - rest_len) / rest_len:+.2f}% "
          f"| floor {floor:.2f} mm")

    shapes_arr = np.stack(shapes)
    np.savez_compressed(
        out_dir / f"{name}_commands.npz",
        commands_mm=np.array(u_list[:n_driven]),
        tip_path_mm=shapes_arr[:n_driven, 0, :], shapes_mm=shapes_arr,
        rmse_mm=np.array(hist), vmax_mm_s=np.array(vhist),
        goal_nodes_mm=goal_pos, settled_nodes_mm=final,
        best_nodes_mm=best_shape if best_shape is not None else np.zeros((0, 2)),
        n_driven=n_driven, best_step=best_t, best_rmse_mm=best_rmse,
        settled_rmse_mm=hist[-1], recon_floor_mm=floor)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], 'o-', color='g', label='goal')
    ax[0].plot(final[:, 0], final[:, 1], 's-', color='r',
               label=f'settled ({hist[-1]:.1f} mm)')
    if best_shape is not None:
        ax[0].plot(best_shape[:, 0], best_shape[:, 1], '^--', color='orange',
                   alpha=0.7, label=f'best driven ({best_rmse:.1f} mm)')
    ax[0].plot(shapes_arr[:n_driven, 0, 0], shapes_arr[:n_driven, 0, 1], '-',
               color='tab:blue', lw=1, alpha=0.5, label='tip path')
    ax[0].plot(start[:, 0], start[:, 1], ':', color='gray', label='start')
    ax[0].axis('equal'); ax[0].grid(True); ax[0].legend(); ax[0].set_title(name)
    ax[1].plot(hist)
    ax[1].axvline(n_driven, ls=':', color='k', label='settle')
    ax[1].axhline(floor, ls='--', color='r', label='recon floor')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('RMSE (mm)')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return dict(shape=name, best_rmse_mm=best_rmse, settled_rmse_mm=hist[-1],
                recon_floor_mm=floor, best_step=best_t, steps=len(u_list),
                exploded=exploded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="rope_all_shapes/rope_C.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shape-dir", default="rope_all_shapes")
    ap.add_argument("--out-dir", default="koopman_bmpc_outputs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--results-dir", default="training_results")
    ap.add_argument("--exp-name", default = "exp_simple_1788437703")

    ap.add_argument("--H", type=int, default=15, help="horizon")
    ap.add_argument("--inner", type=int, default=2,
                    help="linearisation passes per MPC step (1 = linearise about "
                         "the shifted previous solution only)")
    ap.add_argument("--u-max", type=float, default=1.5)
    ap.add_argument("--du-max", type=float, default=0.3)
    ap.add_argument("--r-weight", type=float, default=1e-1)
    ap.add_argument("--terminal-weight", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--stall", type=int, default=40)
    ap.add_argument("--stall-after", type=int, default=150)
    ap.add_argument("--settle", type=int, default=600)
    ap.add_argument("--no-rewind", action="store_true")
    ap.add_argument("--project-rho", type=float, default=0.0,
                    help="clip eigenvalues of A above this onto the circle "
                         "(0 disables; a uniform rescale would kill the "
                         "integrator modes the tip needs)")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    ckpt_dir = Path(args.results_dir) / "checkpoints" / args.exp_name
    ta = ast.literal_eval((Path(args.results_dir) / f"{args.exp_name}.txt")
                          .read_text().splitlines()[0].split("Args: ", 1)[1])

    dataset = RopeKoopmanDataset(
        ta["csv"], features=tuple(ta["features"]), horizon=1,
        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None)
    mean = torch.as_tensor(dataset.state_mean[0])
    std = torch.as_tensor(dataset.state_std[0])
    pos_slice = dataset.pos_slice
    has_vel = "vel" in dataset.feature_slices
    n_nodes = (pos_slice.stop - pos_slice.start) // 2

    model = KoopmanModel(
        state_dim=dataset.state_dim, control_dim=dataset.control_dim,
        koopman_dim=ta["koopman_dim"], hidden_dim=ta["hidden_dim"],
        num_hidden=ta["num_hidden"],
        operator_bias=not ta.get("no_operator_bias", False),
        control_lag=ta.get("control_lag", 0),
        bilinear=ta.get("bilinear", 0)).to("cpu")
    ld = lambda f: torch.load(ckpt_dir / f, map_location="cpu")
    model.encoder.load_state_dict(ld("koopman_enc.pt"))
    model.decoder.load_state_dict(ld("koopman_dec.pt"))
    model.operator.load_state_dict(ld("koopman_op.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A = model.operator.A.detach().numpy().astype(np.float64)
    B = model.operator.B.detach().numpy().astype(np.float64) / dataset.cmd_scale
    N = (model.operator.N.detach().numpy().astype(np.float64) / dataset.cmd_scale
         if getattr(model.operator, "N", None) is not None else None)
    nz = A.shape[0]

    ev = np.abs(np.linalg.eigvals(A))
    print(f"nz {nz} | rho(A) {ev.max():.6f} | integrator modes {int((ev>0.98).sum())} "
          f"| bilinear {N is not None}")
    if N is not None:
        print(f"||N||/||A|| = {np.linalg.norm(N)/np.linalg.norm(A):.4f}")
    if args.project_rho > 0:
        w, V = np.linalg.eig(A)
        A = np.real(V @ np.diag(np.where(np.abs(w) > args.project_rho,
                                         w / np.abs(w) * args.project_rho, w))
                    @ np.linalg.inv(V))
        print(f"projected rho(A) {np.abs(np.linalg.eigvals(A)).max():.6f}")

    env = RopeManipEnv(n_nodes=n_nodes, base_length=0.30, base_radius=0.003,
                       youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01,
                       seed=0)
    env.reset()

    ctx = dict(dataset=dataset, model=model, env=env, mean=mean, std=std,
               pos_slice=pos_slice, has_vel=has_vel, n_nodes=n_nodes, nz=nz,
               A=A, B=B, N=N)

    shapes_in = (sorted(Path(args.shape_dir).glob("*.json")) if args.all
                 else [Path(args.shape)])
    results = []
    for sp in shapes_in:
        try:
            results.append(run_shape(args, sp, ctx))
        except Exception as exc:
            print(f"  !! {sp.stem} failed: {type(exc).__name__}: {exc}")
            results.append(dict(shape=sp.stem, error=str(exc)))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["shape,best_rmse_mm,settled_rmse_mm,recon_floor_mm,best_step,steps,exploded"]
    for r in results:
        if "error" in r:
            lines.append(f"{r['shape']},,,,,,ERROR"); continue
        lines.append(",".join(str(x) for x in [
            r["shape"], f"{r['best_rmse_mm']:.3f}", f"{r['settled_rmse_mm']:.3f}",
            f"{r['recon_floor_mm']:.3f}", r["best_step"], r["steps"], r["exploded"]]))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")

    print(f"\n{'=' * 70}\nSUMMARY")
    print(f"{'shape':<22} {'best':>8} {'settled':>9} {'floor':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['shape']:<22} {'ERROR':>8}"); continue
        print(f"{r['shape']:<22} {r['best_rmse_mm']:8.2f} "
              f"{r['settled_rmse_mm']:9.2f} {r['recon_floor_mm']:8.2f}")


if __name__ == "__main__":
    main()