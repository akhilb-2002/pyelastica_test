"""
letter_mpc.py -- Koopman MPC for tracing letter shapes with the rope.

The optimisation runs entirely in latent space. The encoder is an MLP, so it
cannot go inside the solver; instead the measured state is encoded ONCE per MPC
step (outside CasADi) and becomes a parameter. Inside the solver the dynamics are

    z_{k+1} = A z_k + B u_k

which is linear in the raw mm command (B is divided by cmd_scale because the
operator was trained on u / cmd_scale), so the whole OCP is a QP and the command
box |u| <= U_MAX sits directly on the decision variable.

Tracking is on the LATENT error z - z_goal, weighted by J^T J with J the decoder
Jacobian at the goal, which makes it a first-order proxy for mm position error.
The decoder itself cannot go inside the solver without destroying the QP.

Each run ends with a SETTLE phase at the BEST step, not the last one. The stall
counter only fires N steps after the minimum and the controller keeps driving in
the meantime, so settling from the last step locks in that overshoot. The command
history is replayed from a fresh reset up to the best step instead (replay is
deterministic: same reset, same commands, same trajectory), and the damper then
brings the rope to rest. The settled shape is the honest result -- a letter that
only exists while the rope is moving is not a letter.

Per-shape outputs (in --out-dir):
    <name>.gif            drive + settle animation
    <name>.png            static goal / settled / best figure
    <name>_commands.npz   commands, tip path, every shape frame, RMSE and |v|
                          traces, the drive/settle split, and the settled nodes
    summary.csv           one row per shape

Usage:
    python letter_mpc.py --shape rope_all_shapes/rope_I.json
    python letter_mpc.py --all
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
import matplotlib.animation as animation

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM


def run_shape(args, shape_path, ctx):
    """MPC + settle for one target shape. `ctx` holds everything shape-independent
    (dataset stats, model, A, B, env) so the model loads once across a batch."""
    dataset, model, env = ctx["dataset"], ctx["model"], ctx["env"]
    mean, std = ctx["mean"], ctx["std"]
    pos_slice, vel_slice = ctx["pos_slice"], ctx["vel_slice"]
    n_nodes, z_dim = ctx["n_nodes"], ctx["z_dim"]
    A_np, B_np = ctx["A_np"], ctx["B_np"]

    name = Path(shape_path).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n=== {name}\n{'=' * 72}")

    # ---- goal --------------------------------------------------------------
    J = json.load(open(shape_path))
    goal_pos = np.array(J["nodes"], dtype=np.float32)
    assert goal_pos.shape == (n_nodes, 2)

    x_goal = np.zeros(dataset.state_dim, dtype=np.float32)
    x_goal[dataset.feature_slices["pos"]] = goal_pos.reshape(-1)
    xg = (torch.as_tensor(x_goal).unsqueeze(0) - mean) / std

    with torch.no_grad():
        rec = model.decoder(model.encoder(xg)) * std + mean
    rec_pos = rec[0, pos_slice].reshape(n_nodes, 2).numpy()
    floor = float(np.sqrt(((rec_pos - goal_pos) ** 2).sum(1).mean()))
    print(f"goal |z| max = {xg.abs().max().item():.3f}")
    print(f"goal recon RMSE = {floor:.3f} mm")

    env.reset()          # fresh straight rope for every shape

    def measure():
        s = np.zeros(dataset.state_dim, dtype=np.float32)
        s[dataset.feature_slices["pos"]] = env.get_positions(mm=True)[:, :2].reshape(-1)
        if "vel" in dataset.feature_slices:
            s[dataset.feature_slices["vel"]] = env.get_velocities(mm=True)[:, :2].reshape(-1)
        return s

    def encode(state_mm):
        xn = (torch.as_tensor(state_mm).unsqueeze(0) - mean) / std
        with torch.no_grad():
            return model.encoder(xn).numpy().ravel().astype(np.float64)

    def rmse_now():
        cur = env.get_positions(mm=True)[:, :2]
        return float(np.sqrt(((cur - goal_pos) ** 2).sum(1).mean()))

    def vmax_now():
        return float(np.abs(env.get_velocities(mm=True)[:, :2]).max())

    # ---- Jacobian-weighted cost -------------------------------------------
    zg_t = torch.tensor(encode(x_goal), dtype=torch.float32)
    w_vel = args.w_vel

    def dec_pv(z):
        out = model.decoder(z.unsqueeze(0))[0]
        return torch.cat([out[pos_slice], w_vel * out[vel_slice]])

    Jt = torch.autograd.functional.jacobian(dec_pv, zg_t)
    scale = np.concatenate([dataset.state_std[0, pos_slice],
                            dataset.state_std[0, vel_slice]])
    Jd = (Jt.numpy() * scale[:, None]).astype(np.float64)
    Q = Jd.T @ Jd
    Q += 1e-4 * (np.trace(Q) / z_dim) * np.eye(z_dim)
    Q /= (np.trace(Q) / z_dim)
    print(f"Q cond {np.linalg.cond(Q):.2e}  rank(Jd) {np.linalg.matrix_rank(Jd)}")

    # ---- QP, built once and re-solved with new parameters each step --------
    opti = ca.Opti()
    Z = opti.variable(z_dim, args.H + 1)
    U = opti.variable(2, args.K)
    z0_p = opti.parameter(z_dim)
    zref_p = opti.parameter(z_dim)

    opti.subject_to(Z[:, 0] == z0_p)
    cost = 0
    for k in range(args.H):
        u_k = U[:, min(k, args.K - 1)]          # zero-order hold past K
        opti.subject_to(Z[:, k + 1] == ca.DM(A_np) @ Z[:, k] + ca.DM(B_np) @ u_k)
        e = Z[:, k + 1] - zref_p
        w = args.terminal_weight if k == args.H - 1 else 1.0
        cost += w * ca.bilin(ca.DM(Q), e, e)

    for k in range(args.K):
        cost += args.r_weight * ca.dot(U[:, k], U[:, k])
        opti.subject_to(opti.bounded(-args.u_max, U[:, k], args.u_max))
    for k in range(1, args.K):
        opti.subject_to(opti.bounded(-args.du_max, U[:, k] - U[:, k - 1], args.du_max))

    opti.minimize(cost)
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "sb": "yes", "max_iter": 200})

    # ---- closed loop -------------------------------------------------------
    start_pos = env.get_positions(mm=True)[:, :2].copy()
    total_travel = float(np.linalg.norm(goal_pos[0] - start_pos[0]))
    print(f"node-0 must travel {total_travel:.1f} mm; at {args.u_max} mm/step "
          f">= {int(total_travel / args.u_max)} steps")

    hist, shapes, vhist, u_list = [], [], [], []
    best_rmse, best_shape, best_t, stall = np.inf, None, 0, 0
    exploded = False

    for t in range(args.steps):
        z0 = encode(measure())

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
        u_list.append(u.copy())

        env.drive_step(u / MM)
        if env.is_exploded():
            print(f"  step {t}: sim exploded")
            exploded = True
            break

        cur = env.get_positions(mm=True)[:, :2]
        rmse = rmse_now()
        hist.append(rmse); shapes.append(cur.copy()); vhist.append(vmax_now())

        if t % 40 == 0:
            print(f"  step {t:4d}  |u| {np.linalg.norm(u):.3f} mm  RMSE {rmse:7.2f} mm")

        if rmse < best_rmse - 1e-6:
            best_rmse, best_shape, best_t, stall = rmse, cur.copy(), t, 0
        else:
            stall += 1
            if stall > args.stall and t > 200:
                print(f"  no improvement for {args.stall} steps -- stop at t={t}; "
                      f"best {best_rmse:.2f} mm at t={best_t}")
                break

    n_solved = len(u_list)
    print(f"driven:  last {hist[-1]:.2f} mm   best {best_rmse:.2f} mm at t={best_t}")

    # ---- rewind to the best step -------------------------------------------
    # PyElastica has no rewind, so replay the recorded commands from a fresh
    # reset. Deterministic, so this reproduces the best state exactly.
    if not args.no_rewind and best_t < len(hist) - 1:
        print(f"rewinding to t={best_t} ...")
        env.reset()
        hist, shapes, vhist = [], [], []
        for k in range(best_t + 1):
            env.drive_step(u_list[k] / MM)
            shapes.append(env.get_positions(mm=True)[:, :2].copy())
            hist.append(rmse_now()); vhist.append(vmax_now())
        print(f"  replayed to {hist[-1]:.2f} mm (expected {best_rmse:.2f} mm)")

    u_applied = np.array(u_list[:len(hist)], dtype=np.float64)     # (n_driven, 2)
    n_driven = len(hist)

    # ---- settle ------------------------------------------------------------
    print("settling (u = 0) ...")
    for k in range(args.settle):
        env.drive_step(np.zeros(2))
        if env.is_exploded():
            print("  exploded during settle"); break
        shapes.append(env.get_positions(mm=True)[:, :2].copy())
        hist.append(rmse_now()); vhist.append(vmax_now())

    final = env.get_positions(mm=True)[:, :2]
    settled_rmse, settled_v = hist[-1], vhist[-1]
    print(f"settled: RMSE {settled_rmse:.2f} mm   max|v| {settled_v:.4f} mm/s   "
          f"(best driven {best_rmse:.2f}, floor {floor:.2f})")

    shapes_arr = np.stack(shapes)                                  # (F, N, 2)
    tip_path = shapes_arr[:n_driven, 0, :]                         # (n_driven, 2) mm

    # ---- save the command / settle record ----------------------------------
    # tip_path is the absolute node-0 trajectory: it is what a robot would follow
    # as a position reference, independent of how fast it is executed.
    npz_path = out_dir / f"{name}_commands.npz"
    np.savez_compressed(
        npz_path,
        commands_mm=u_applied,           # per-step node-0 displacement (mm)
        tip_path_mm=tip_path,            # absolute node-0 positions (mm)
        shapes_mm=shapes_arr,            # every node, every frame (mm)
        rmse_mm=np.array(hist),
        vmax_mm_s=np.array(vhist),
        goal_nodes_mm=goal_pos,
        settled_nodes_mm=final,
        best_nodes_mm=best_shape if best_shape is not None else np.zeros((0, 2)),
        n_driven=n_driven,               # frames before the settle begins
        n_settle=len(hist) - n_driven,
        best_step=best_t, best_rmse_mm=best_rmse,
        settled_rmse_mm=settled_rmse, settled_vmax_mm_s=settled_v,
        recon_floor_mm=floor, control_dt=env.control_dt,
    )
    print(f"saved -> {npz_path}")

    # ---- static figure -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(goal_pos[:, 0], goal_pos[:, 1], 'o-', color='g', label='goal')
    ax[0].plot(final[:, 0], final[:, 1], 's-', color='r',
               label=f'settled ({settled_rmse:.1f} mm)')
    if best_shape is not None:
        ax[0].plot(best_shape[:, 0], best_shape[:, 1], '^--', color='orange',
                   alpha=0.7, label=f'best driven ({best_rmse:.1f} mm)')
    ax[0].plot(start_pos[:, 0], start_pos[:, 1], ':', color='gray', label='start')
    ax[0].axis('equal'); ax[0].grid(True); ax[0].legend()
    ax[0].set_xlabel('X (mm)'); ax[0].set_ylabel('Y (mm)'); ax[0].set_title(name)
    ax[1].plot(hist)
    ax[1].axvline(n_driven, ls=':', color='k', label='settle starts')
    ax[1].axhline(floor, ls='--', color='r', label='recon floor')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('RMSE to goal (mm)')
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    if not args.show:
        plt.close(fig)

    # ---- animation ---------------------------------------------------------
    if not args.no_anim:
        idx = list(range(0, len(shapes), args.anim_every))
        if idx[-1] != len(shapes) - 1:
            idx.append(len(shapes) - 1)              # always end on the settled shape

        allpts = np.concatenate([shapes_arr.reshape(-1, 2), goal_pos])
        pad = 20.0
        xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
        ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

        afig, aax = plt.subplots(1, 2, figsize=(14, 6))
        aax[0].plot(goal_pos[:, 0], goal_pos[:, 1], 'o-', color='g',
                    alpha=0.5, label='goal', zorder=1)
        aax[0].plot(start_pos[:, 0], start_pos[:, 1], ':', color='gray',
                    label='start', zorder=1)
        aax[0].plot(tip_path[:, 0], tip_path[:, 1], '-', color='tab:blue',
                    lw=1, alpha=0.6, label='tip path', zorder=2)
        (rope_ln,) = aax[0].plot([], [], 's-', color='r', ms=4, label='rope', zorder=3)
        (tip_ln,) = aax[0].plot([], [], 'o', color='k', ms=9, zorder=4)
        aax[0].set_xlim(*xlim); aax[0].set_ylim(*ylim)
        aax[0].set_aspect('equal'); aax[0].grid(True); aax[0].legend(loc='best')
        aax[0].set_xlabel('X (mm)'); aax[0].set_ylabel('Y (mm)')
        title = aax[0].set_title("")

        aax[1].plot(hist, color='C0', lw=1)
        aax[1].axvline(n_driven, ls=':', color='k', label='settle starts')
        aax[1].axhline(floor, ls='--', color='r', label='recon floor')
        (cursor,) = aax[1].plot([], [], 'o', color='r', ms=7)
        aax[1].set_xlabel('step'); aax[1].set_ylabel('RMSE to goal (mm)')
        aax[1].grid(True); aax[1].legend()
        plt.tight_layout()

        def draw(f):
            i = idx[f]
            s = shapes_arr[i]
            rope_ln.set_data(s[:, 0], s[:, 1])
            tip_ln.set_data([s[0, 0]], [s[0, 1]])
            cursor.set_data([i], [hist[i]])
            phase = "DRIVE" if i < n_driven else "SETTLE (u = 0)"
            rope_ln.set_color('r' if i < n_driven else 'tab:purple')
            title.set_text(f"{name}  |  {phase}  |  step {i}  |  "
                           f"RMSE {hist[i]:.2f} mm  |  max|v| {vhist[i]:.1f} mm/s")
            return rope_ln, tip_ln, cursor, title

        anim = animation.FuncAnimation(afig, draw, frames=len(idx),
                                       interval=1000 / args.anim_fps, blit=False)
        gif = out_dir / f"{name}.gif"
        print(f"writing {len(idx)} frames -> {gif} ...")
        anim.save(gif, writer=animation.PillowWriter(fps=args.anim_fps))
        if not args.show:
            plt.close(afig)

    if args.show:
        plt.show()

    return dict(shape=name, best_rmse_mm=best_rmse, settled_rmse_mm=settled_rmse,
                settled_vmax_mm_s=settled_v, recon_floor_mm=floor,
                best_step=best_t, steps_solved=n_solved,
                goal_z_max=float(xg.abs().max().item()),
                node0_travel_mm=total_travel, exploded=exploded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="rope_all_shapes/rope_I.json")
    ap.add_argument("--all", action="store_true",
                    help="run every *.json in --shape-dir")
    ap.add_argument("--shape-dir", default="rope_all_shapes")
    ap.add_argument("--out-dir", default="koopman_mpc_outputs")
    ap.add_argument("--show", action="store_true",
                    help="display figures (off by default so --all runs unattended)")
    ap.add_argument("--results-dir", default="training_results")
    ap.add_argument("--exp-name", default="exp_simple_1788069999")
    ap.add_argument("--H", type=int, default=10, help="prediction horizon")
    ap.add_argument("--K", type=int, default=10, help="control horizon (K <= H)")
    ap.add_argument("--u-max", type=float, default=1.5,
                    help="mm/step box on the command; keep <= 0.75 * U_MAX_MM")
    ap.add_argument("--du-max", type=float, default=0.3)
    ap.add_argument("--r-weight", type=float, default=1e-1)
    ap.add_argument("--terminal-weight", type=float, default=50)
    ap.add_argument("--w-vel", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--stall", type=int, default=40)
    ap.add_argument("--settle", type=int, default=600)
    ap.add_argument("--no-rewind", action="store_true",
                    help="settle from the LAST step instead of the best one")
    ap.add_argument("--ref-rate", type=float, default=0)
    ap.add_argument("--project-rho", type=float, default=0.99)
    ap.add_argument("--no-anim", action="store_true", help="skip GIF generation")
    ap.add_argument("--anim-every", type=int, default=4)
    ap.add_argument("--anim-fps", type=int, default=30)
    args = ap.parse_args()
    assert args.K <= args.H, "control horizon must not exceed prediction horizon"

    if not args.show:
        matplotlib.use("Agg")

    device = torch.device("cpu")
    ckpt_dir = Path(args.results_dir) / "checkpoints" / args.exp_name

    def load_exp_args(results_dir, exp_name):
        log_path = Path(results_dir) / f"{exp_name}.txt"
        if not log_path.exists():
            raise FileNotFoundError(f"{log_path} not found; cannot recover the training args.")
        return ast.literal_eval(log_path.read_text().splitlines()[0].split("Args: ", 1)[1])

    ta = load_exp_args(args.results_dir, args.exp_name)

    dataset = RopeKoopmanDataset(
        ta["csv"], features=tuple(ta["features"]), horizon=1,
        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None,
    )
    mean = torch.as_tensor(dataset.state_mean[0])
    std = torch.as_tensor(dataset.state_std[0])
    pos_slice = dataset.pos_slice
    vel_slice = dataset.feature_slices["vel"]
    n_nodes = (pos_slice.stop - pos_slice.start) // 2
    cmd_scale = dataset.cmd_scale

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
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A_np = model.operator.A.detach().cpu().numpy().astype(np.float64)
    B_np = model.operator.B.detach().cpu().numpy().astype(np.float64) / cmd_scale
    z_dim = A_np.shape[0]

    rho = np.abs(np.linalg.eigvals(A_np)).max()
    if args.project_rho > 0 and rho > args.project_rho:
        A_np = A_np * (args.project_rho / rho)
    print(f"A {A_np.shape}  B {B_np.shape}  rho(A) {rho:.6f} -> "
          f"{np.abs(np.linalg.eigvals(A_np)).max():.6f}")

    env = RopeManipEnv(n_nodes=n_nodes, base_length=0.30, base_radius=0.003,
                       youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01,
                       seed=0)
    env.reset()

    ctx = dict(dataset=dataset, model=model, env=env, mean=mean, std=std,
               pos_slice=pos_slice, vel_slice=vel_slice, n_nodes=n_nodes,
               z_dim=z_dim, A_np=A_np, B_np=B_np)

    if args.all:
        shapes_in = sorted(Path(args.shape_dir).glob("*.json"))
        if not shapes_in:
            raise SystemExit(f"no *.json found in {args.shape_dir}")
        print(f"\n--all: {len(shapes_in)} shapes from {args.shape_dir}")
    else:
        shapes_in = [Path(args.shape)]

    results = []
    for sp in shapes_in:
        try:
            results.append(run_shape(args, sp, ctx))
        except Exception as exc:          # one bad shape must not kill the batch
            print(f"  !! {sp.stem} failed: {type(exc).__name__}: {exc}")
            results.append(dict(shape=sp.stem, error=str(exc)))

    # ---- summary -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = ("shape,best_rmse_mm,settled_rmse_mm,settled_vmax_mm_s,recon_floor_mm,"
           "best_step,steps_solved,goal_z_max,node0_travel_mm,exploded")
    lines = [hdr]
    for r in results:
        if "error" in r:
            lines.append(f"{r['shape']},,,,,,,,,ERROR")
            continue
        lines.append(",".join(str(x) for x in [
            r["shape"], f"{r['best_rmse_mm']:.3f}", f"{r['settled_rmse_mm']:.3f}",
            f"{r['settled_vmax_mm_s']:.4f}", f"{r['recon_floor_mm']:.3f}",
            r["best_step"], r["steps_solved"], f"{r['goal_z_max']:.3f}",
            f"{r['node0_travel_mm']:.1f}", r["exploded"]]))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")

    print(f"\n{'=' * 72}\nSUMMARY  ({out_dir}/summary.csv)")
    print(f"{'shape':<22} {'best':>8} {'settled':>9} {'floor':>8} {'max|v|':>9}")
    for r in results:
        if "error" in r:
            print(f"{r['shape']:<22} {'ERROR':>8}")
            continue
        print(f"{r['shape']:<22} {r['best_rmse_mm']:8.2f} {r['settled_rmse_mm']:9.2f} "
              f"{r['recon_floor_mm']:8.2f} {r['settled_vmax_mm_s']:9.4f}")


if __name__ == "__main__":
    main()