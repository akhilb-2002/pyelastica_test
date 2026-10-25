"""
Kinematic-consistency check: does  p_{k+1} - p_k = dt * v_k  hold?

Two separate questions, and it matters which one fails:

  1. Does it hold in the DATA? If not, the relation as written is wrong (wrong dt,
     wrong velocity convention, or v_k is the velocity BEFORE the step rather than
     after) and adding it as a loss would pull the model toward a false constraint.

  2. Does it hold in the MODEL's rollout? Only worth asking once (1) passes. If the
     data satisfies it and the model does not, the position and velocity blocks of
     the decoder have drifted apart -- the model reports velocities its own
     positions contradict -- and w_kin has something to fix.

Run this BEFORE turning on --w-kin.
"""

import numpy as np
import torch
from pathlib import Path

from ngk_simple import RopeKoopmanDataset, KoopmanModel
from mpc import load_exp_args

results_dir = "training_results"
exp_name = "exp_simple_1788370256"          # <-- a pos+vel checkpoint
DT = 0.01                              # env.control_dt

ta = load_exp_args(results_dir, exp_name)
ds = RopeKoopmanDataset(ta["csv"], features=tuple(ta["features"]), horizon=10,
                        normalize=not ta.get("no_normalize", False), clamp_tol_mm=None)
assert "vel" in ds.feature_slices, "this check needs a pos+vel model"

ps, vs = ds.pos_slice, ds.feature_slices["vel"]
pm = torch.as_tensor(ds.state_mean[0, ps]); pstd = torch.as_tensor(ds.state_std[0, ps])
vm = torch.as_tensor(ds.state_mean[0, vs]); vstd = torch.as_tensor(ds.state_std[0, vs])


def denorm(x, m, s):
    return x * s + m


# ---- 1. does the relation hold in the DATA? --------------------------------
# Take raw consecutive states straight from the dataset, no model involved.
idx = np.random.RandomState(0).choice(len(ds), 512, replace=False)
res_after, res_before, disp = [], [], []
for i in idx:
    x0, us, xs, u_prev, _ = ds[int(i)]
    P = denorm(torch.cat([x0[ps].unsqueeze(0), xs[:, ps]], 0), pm, pstd)   # [H+1,60]
    V = denorm(torch.cat([x0[vs].unsqueeze(0), xs[:, vs]], 0), vm, vstd)   # [H+1,60]
    d = P[1:] - P[:-1]                                                     # [H,60] mm
    res_after.append((d - DT * V[1:]).abs().mean().item())    # v AFTER the step
    res_before.append((d - DT * V[:-1]).abs().mean().item())  # v BEFORE the step
    disp.append(d.abs().mean().item())

disp_m = float(np.mean(disp))
print("--- 1. in the DATA ---")
print(f"mean |p_k+1 - p_k|                 : {disp_m:.5f} mm")
print(f"residual using v_AFTER  the step   : {np.mean(res_after):.5f} mm "
      f"({100*np.mean(res_after)/disp_m:.1f}% of the displacement)")
print(f"residual using v_BEFORE the step   : {np.mean(res_before):.5f} mm "
      f"({100*np.mean(res_before)/disp_m:.1f}% of the displacement)")

best = "AFTER" if np.mean(res_after) < np.mean(res_before) else "BEFORE"
frac = min(np.mean(res_after), np.mean(res_before)) / disp_m
print(f"-> v_{best} is the right convention; residual is {100*frac:.1f}% of motion")
if frac > 0.5:
    print("   !! the relation does NOT hold in the data at either convention. "
          "Check dt and the velocity units before adding this loss -- a wrong "
          "constraint is worse than none.")

# ---- what dt would make it exact? ------------------------------------------
# Least squares over the whole sample: dt* = <d, v> / <v, v>
num = den = 0.0
for i in idx[:128]:
    x0, us, xs, u_prev, _ = ds[int(i)]
    P = denorm(torch.cat([x0[ps].unsqueeze(0), xs[:, ps]], 0), pm, pstd)
    V = denorm(torch.cat([x0[vs].unsqueeze(0), xs[:, vs]], 0), vm, vstd)
    d, v = P[1:] - P[:-1], (V[1:] if best == "AFTER" else V[:-1])
    num += float((d * v).sum()); den += float((v * v).sum())
print(f"best-fit dt = {num/den:.6f} s   (env.control_dt = {DT})")

# ---- 2. does the MODEL's rollout satisfy it? -------------------------------
device = torch.device("cpu")
ck = Path(results_dir) / "checkpoints" / exp_name
model = KoopmanModel(
    state_dim=ds.state_dim, control_dim=ds.control_dim,
    koopman_dim=ta["koopman_dim"], hidden_dim=ta["hidden_dim"],
    num_hidden=ta["num_hidden"],
    operator_bias=not ta.get("no_operator_bias", False),
    control_lag=ta.get("control_lag", 0),
    bilinear=ta.get("bilinear", 0)).to(device)
ld = lambda f: torch.load(ck / f, map_location=device)
model.encoder.load_state_dict(ld("koopman_enc.pt"))
model.decoder.load_state_dict(ld("koopman_dec.pt"))
model.operator.load_state_dict(ld("koopman_op.pt"))
model.eval()
for p_ in model.parameters():
    p_.requires_grad_(False)

X0 = torch.stack([ds[int(i)][0] for i in idx[:256]])
US = torch.stack([ds[int(i)][1] for i in idx[:256]])
UP = torch.stack([ds[int(i)][3] for i in idx[:256]])
XS = torch.stack([ds[int(i)][2] for i in idx[:256]])
with torch.no_grad():
    x_hat, _, _ = model.rollout(X0, US, UP)

P = denorm(torch.cat([X0[:, ps].unsqueeze(1), x_hat[:, :, ps]], 1), pm, pstd)
V = denorm(torch.cat([X0[:, vs].unsqueeze(1), x_hat[:, :, vs]], 1), vm, vstd)
d = P[:, 1:] - P[:, :-1]
v = V[:, 1:] if best == "AFTER" else V[:, :-1]
res_model = (d - DT * v).abs().mean().item()
disp_model = d.abs().mean().item()

# how far off are the predicted velocities themselves?
V_true = denorm(XS[:, :, vs], vm, vstd)
v_err = (denorm(x_hat[:, :, vs], vm, vstd) - V_true).abs().mean().item()

print("\n--- 2. in the MODEL rollout ---")
print(f"mean |p_k+1 - p_k|      : {disp_model:.5f} mm")
print(f"kinematic residual      : {res_model:.5f} mm "
      f"({100*res_model/disp_model:.1f}% of the displacement)")
print(f"velocity prediction err : {v_err:.2f} mm/s")
print(f"kin loss value (mm^2)   : {((d - DT*v)**2).mean().item():.6f}")

if res_model / disp_model < 1.2 * frac:
    print("-> the model already satisfies it about as well as the data does; "
          "w_kin will have little to fix")
else:
    print("-> the model violates it MORE than the data does; the position and "
          "velocity blocks have drifted apart, and w_kin has something to correct")