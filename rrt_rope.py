"""
rrt_rope.py -- kinodynamic RRT for rope letter shapes, research scratch code.

Planning happens in TIP space (node 0, 2-d) rather than latent space. RRT degrades
badly above ~10 dimensions, and worse, uniform samples in a 256-d latent would
almost surely land off the ~60-d manifold of physically realisable ropes. The tip
is 2-d and, quasi-statically, the rope shape is a function of the tip path -- so
sampling tips and propagating the Koopman model gives a well-conditioned search.

Each tree node stores (z, decoded positions, parent, command). The decode happens
ONCE at insertion, so the nearest-neighbour loop is pure numpy / KD-tree lookups
with no network calls.

Nearest-neighbour is on tip position (what the sample specifies); the goal test is
on the full decoded shape (what we actually care about). Those are deliberately
different metrics.
"""

import ast, json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from rope_manip_env import RopeManipEnv, MM

# ---------------------------------------------------------------- setup ----
RESULTS, EXP = "training_results", "exp_simple_1788269928"
SHAPE = "rope_all_shapes/rope_J.json"

N_ITER      = 4000     # tree expansions
STEPS_EXT   = 15       # model steps per extension
U_MAX       = 1.5      # mm/step
GOAL_BIAS   = 0.15     # fraction of samples aimed at the goal tip
REBUILD     = 50       # rebuild the KD-tree every this many insertions
RHO_TARGET  = 0.99

ta = ast.literal_eval(
    (Path(RESULTS) / f"{EXP}.txt").read_text().splitlines()[0].split("Args: ", 1)[1])

ds = RopeKoopmanDataset(ta["csv"], features=tuple(ta["features"]), horizon=1,
                        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None)
mean = torch.as_tensor(ds.state_mean[0])
std = torch.as_tensor(ds.state_std[0])
N_NODES = ds.state_dim // 2

model = KoopmanModel(state_dim=ds.state_dim, control_dim=ds.control_dim,
                     koopman_dim=ta["koopman_dim"], hidden_dim=ta["hidden_dim"],
                     num_hidden=ta["num_hidden"],
                     operator_bias=not ta.get("no_operator_bias", False))
ck = Path(RESULTS) / "checkpoints" / EXP
model.encoder.load_state_dict(torch.load(ck / "koopman_enc.pt", map_location="cpu"))
model.decoder.load_state_dict(torch.load(ck / "koopman_dec.pt", map_location="cpu"))
model.operator.load_state_dict(torch.load(ck / "koopman_op.pt", map_location="cpu"))
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

A = model.operator.A.detach().numpy().astype(np.float64)
B = model.operator.B.detach().numpy().astype(np.float64) / ds.cmd_scale
rho = np.abs(np.linalg.eigvals(A)).max()
if rho > RHO_TARGET:
    A = A * (RHO_TARGET / rho)
print(f"rho(A) {rho:.4f} -> {np.abs(np.linalg.eigvals(A)).max():.4f}")


def encode(pos):
    x = torch.as_tensor(np.asarray(pos, np.float32).reshape(-1))[None]
    with torch.no_grad():
        return model.encoder((x - mean) / std).numpy().ravel().astype(np.float64)


def decode(z):
    with torch.no_grad():
        x = model.decoder(torch.tensor(z, dtype=torch.float32)[None]) * std + mean
    return x[0].reshape(N_NODES, 2).numpy()


# ---------------------------------------------------------------- problem --
J = json.load(open(SHAPE))
goal = np.array(J["nodes"], dtype=np.float32)
name = Path(SHAPE).stem

env = RopeManipEnv(n_nodes=N_NODES, base_length=0.30, base_radius=0.003,
                   youngs_modulus=1e6, damping=2.0, dt=5e-5, control_dt=0.01, seed=0)
env.reset()
start = env.get_positions(mm=True)[:, :2].copy()
anchor = start[-1]
REACH = 300.0

floor = np.sqrt(((decode(encode(goal)) - goal) ** 2).sum(1).mean())
print(f"{name}: recon floor {floor:.2f} mm | tip travel {np.linalg.norm(goal[0]-start[0]):.1f} mm")


def shape_rmse(P):
    return float(np.sqrt(((P - goal) ** 2).sum(1).mean()))


# ------------------------------------------------------------------- RRT --
Z = [encode(start)]                 # latent per node
P = [start.copy()]                  # decoded (or true) positions per node
TIP = [start[0].copy()]             # node-0 position, the search space
PARENT = [-1]
CMD = [np.zeros(2)]                 # command that produced this node from its parent
COST = [shape_rmse(start)]

tree = cKDTree(np.array(TIP))
best_i, best_r = 0, COST[0]
history = [best_r]

for it in range(N_ITER):
    # --- sample a tip target, uniform by area in the reachable disc --------
    if np.random.rand() < GOAL_BIAS:
        q = goal[0]
    else:
        r = REACH * 0.95 * np.sqrt(np.random.rand())
        th = np.random.uniform(0, 2 * np.pi)
        q = anchor + r * np.array([np.cos(th), np.sin(th)])

    # --- nearest tree node by TIP position --------------------------------
    _, i = tree.query(q)
    i = int(i)

    # --- extend: drive toward q for STEPS_EXT model steps ------------------
    d = q - TIP[i]
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        continue
    u = (d / nd) * min(U_MAX, nd / STEPS_EXT)
    u = u + np.random.normal(0, 0.15, 2)                 # keeps branches distinct
    u = np.clip(u, -U_MAX, U_MAX)

    z = Z[i].copy()
    for _ in range(STEPS_EXT):
        z = A @ z + B @ u
    Pn = decode(z)                                       # ONE decode per new node

    Z.append(z); P.append(Pn); TIP.append(Pn[0].copy())
    PARENT.append(i); CMD.append(u)
    r_new = shape_rmse(Pn)
    COST.append(r_new)

    if r_new < best_r:
        best_r, best_i = r_new, len(Z) - 1
    history.append(best_r)

    if len(Z) % REBUILD == 0:
        tree = cKDTree(np.array(TIP))

print(f"tree size {len(Z)} | best shape RMSE {best_r:.2f} mm (node {best_i})")

# --- extract the path -------------------------------------------------------
path = []
j = best_i
while j != -1:
    path.append(j)
    j = PARENT[j]
path = path[::-1]
print(f"path length {len(path)} nodes -> {(len(path)-1)*STEPS_EXT} sim steps")

u_seq = np.repeat(np.array([CMD[j] for j in path[1:]]), STEPS_EXT, axis=0)

# --- execute the plan in the SIM -------------------------------------------
env.reset()
sim_hist, sim_shapes = [], []
for u in u_seq:
    env.drive_step(u / MM)
    if env.is_exploded():
        print("exploded"); break
    cur = env.get_positions(mm=True)[:, :2]
    sim_shapes.append(cur.copy()); sim_hist.append(shape_rmse(cur))

n_driven = len(sim_hist)
for _ in range(600):                                     # settle
    env.drive_step(np.zeros(2))
    cur = env.get_positions(mm=True)[:, :2]
    sim_shapes.append(cur.copy()); sim_hist.append(shape_rmse(cur))

final = env.get_positions(mm=True)[:, :2]
print(f"model said {best_r:.2f} mm | sim executed {sim_hist[n_driven-1]:.2f} mm | "
      f"settled {sim_hist[-1]:.2f} mm")

# ------------------------------------------------------------------ plots --
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

# tree in tip space
TIParr = np.array(TIP)
for k in range(1, len(TIP)):
    a, b = TIP[PARENT[k]], TIP[k]
    ax[0].plot([a[0], b[0]], [a[1], b[1]], '-', color='lightsteelblue', lw=0.4, zorder=1)
pth = np.array([TIP[j] for j in path])
ax[0].plot(pth[:, 0], pth[:, 1], '-o', color='tab:red', ms=3, lw=1.6,
           label='chosen path', zorder=3)
ax[0].plot(goal[0, 0], goal[0, 1], '*', color='g', ms=18, label='goal tip', zorder=4)
ax[0].plot(start[0, 0], start[0, 1], 's', color='k', ms=8, label='start tip', zorder=4)
ax[0].set_title(f'RRT tree in tip space ({len(TIP)} nodes)')
ax[0].axis('equal'); ax[0].grid(True); ax[0].legend()
ax[0].set_xlabel('X (mm)'); ax[0].set_ylabel('Y (mm)')

# shapes
ax[1].plot(goal[:, 0], goal[:, 1], 'o-', color='g', label='goal')
ax[1].plot(P[best_i][:, 0], P[best_i][:, 1], 'x--', color='b', alpha=0.7,
           label=f'model best ({best_r:.1f} mm)')
ax[1].plot(final[:, 0], final[:, 1], 's-', color='r',
           label=f'sim settled ({sim_hist[-1]:.1f} mm)')
ax[1].plot(start[:, 0], start[:, 1], ':', color='gray', label='start')
ax[1].axis('equal'); ax[1].grid(True); ax[1].legend(); ax[1].set_title(name)
ax[1].set_xlabel('X (mm)'); ax[1].set_ylabel('Y (mm)')

# convergence + sim execution
ax[2].plot(history, color='C0', label='best RMSE vs tree size')
ax[2].plot(np.linspace(0, len(history) - 1, len(sim_hist)), sim_hist,
           color='C3', alpha=0.7, label='sim execution (rescaled)')
ax[2].axhline(floor, ls='--', color='r', label='recon floor')
ax[2].set_xlabel('tree node'); ax[2].set_ylabel('RMSE to goal (mm)')
ax[2].grid(True); ax[2].legend()

plt.tight_layout()
plt.savefig(f"rrt_{name}.png", dpi=140)
plt.show()