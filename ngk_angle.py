"""
neural_koopman_angle.py

Koopman-with-control model for rope manipulation, on a TANGENT-ANGLE state with
BILINEAR control.

State
-----
The rope is inextensible: the sim measures segment lengths within 0.06% of the
10.345 mm rest length over 300 mm of travel. With the anchor pinned and the length
fixed, the configuration is therefore fully determined by the 29 tangent angles,
not by 60 Cartesian coordinates. Working in angles:

  * inextensibility is EXACT, not penalised -- the decoder emits (cos, sin) pairs
    that are renormalised to unit length, so every reconstructed edge is exactly
    ell long. The `seg` loss of the Cartesian version is unnecessary here.
  * the target shapes get easier to represent. A constant-curvature C is a linear
    ramp in theta; in Cartesian it is 60 coordinates each far from anything in the
    training set. The |z| = 1.35 extrapolation on the C target should shrink.
  * rotation about the anchor is a constant offset in theta, so the encoder does
    not have to learn it 60 times over.

    theta_i = atan2(p_{i+1} - p_i),   i = 0 .. N-2
    p_i     = p_anchor - ell * sum_{j>=i} [cos theta_j, sin theta_j]

The rope is second order, so angular rates are carried too. These are much better
conditioned than Cartesian node velocities (29 dims instead of 60, and bounded by
the bending timescale rather than by tip speed), which is what made the earlier
pos+vel model 4x worse: its loss averaged over 60 hard-to-predict velocity
coordinates and starved the position block of gradient.

    thetadot_i = (e_x * edot_y - e_y * edot_x) / |e|^2 ,  e = p_{i+1} - p_i

State layout: [cos theta (N-1), sin theta (N-1), thetadot (N-1)] = 87 for N = 30.

Operator
--------
    z_{k+1} = A z_k + B u_k + sum_j u_k^(j) N_j z_k

The bilinear term is the physics: the rope's response to a tip displacement
depends on its CURRENT SHAPE -- pulling a straight rope sideways does something
quite different from pulling a curled one. A constant B asserts the response is
shape-independent, which is false, and is a nonlinearity the lift evidently was
not absorbing.

NOTE: the bilinear term makes the dynamics non-convex in (z, u) jointly, so an MPC
built on this is no longer a plain QP. The standard treatment is to fix z over the
horizon from the previous solution, which makes each iteration linear in u, and to
iterate two or three times.

Losses
------
  pred  : multi-step rollout error, measured in MILLIMETRES after reconstructing
          positions from angles -- so the training objective is the same quantity
          that gets reported. A plain MSE over theta mis-weights: an angle error
          near the anchor swings the far end much further than one near the tip.
  lin   : latent linearity, the actual Koopman assumption
  recon : autoencoder consistency
  node0 : node 0 is kinematically driven, so its trajectory is known in closed
          form: p0_{k+h} = p0_k + sum u. Supervising it directly fixes the
          integration-rate error that a shape-averaged loss does not penalise.
  diss  : with u = 0 the rod can only lose bending energy. Physically grounded
          contractivity, replacing the sigma_max hinge.

Usage
-----
  python neural_koopman_angle.py --csv case_1/*.csv --horizon 10 --epochs 50
"""

import argparse
import re
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

ROPE_LENGTH_MM = 300.0


# --------------------------------------------------------------------------- #
# Angle <-> position
# --------------------------------------------------------------------------- #
def angles_to_positions(cos_t, sin_t, anchor, ell):
    """Reconstruct node positions from tangent angles. Differentiable.

    cos_t, sin_t : [..., N-1]
    anchor       : [2]      position of the fixed node (index N-1)
    returns      : [..., N, 2]

    The (cos, sin) pair is renormalised to unit length first, which is what makes
    every edge EXACTLY ell long regardless of what the decoder emitted. This is the
    whole reason for working in angles: inextensibility by construction rather than
    by penalty.
    """
    norm = torch.sqrt(cos_t ** 2 + sin_t ** 2).clamp_min(1e-8)
    e = ell * torch.stack([cos_t / norm, sin_t / norm], dim=-1)      # [..., N-1, 2]
    # p_i = p_anchor - sum_{j>=i} e_j   (since e_j = p_{j+1} - p_j)
    suffix = torch.flip(torch.cumsum(torch.flip(e, [-2]), dim=-2), [-2])
    anchor = anchor.to(e.device).view(*([1] * (e.dim() - 2)), 1, 2)
    p_head = anchor - suffix                                        # nodes 0 .. N-2
    p_tail = anchor.expand(*p_head.shape[:-2], 1, 2)                # node N-1
    return torch.cat([p_head, p_tail], dim=-2)


def positions_to_angle_state(pos, vel, ell):
    """[R, N, 2] positions (+ velocities) -> [R, 3*(N-1)] angle state (numpy)."""
    e = pos[:, 1:] - pos[:, :-1]                                    # [R, N-1, 2]
    L = np.linalg.norm(e, axis=-1, keepdims=True).clip(1e-8)
    cos_t, sin_t = e[..., 0] / L[..., 0], e[..., 1] / L[..., 0]
    if vel is None:
        dth = np.zeros_like(cos_t)
    else:
        ed = vel[:, 1:] - vel[:, :-1]
        # thetadot = (e x edot) / |e|^2
        dth = (e[..., 0] * ed[..., 1] - e[..., 1] * ed[..., 0]) / (L[..., 0] ** 2)
    return np.concatenate([cos_t, sin_t, dth], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class RopeAngleDataset(Dataset):
    """Rollout windows in tangent-angle coordinates.

    __getitem__ -> x0 [D], us [H,2], xs [H,D], pos_xs [H,N,2] (mm), case []

    pos_xs is carried alongside so the prediction loss can be taken in millimetres
    without re-deriving the ground truth from angles.
    """

    REQUIRED = ("trajectory_id", "action_idx", "step_idx", "cmd_dx", "cmd_dy")

    def __init__(self, csv_paths, horizon=1, normalize=True, use_rates=True):
        super().__init__()
        self.horizon, self.use_rates = horizon, use_rates

        df, self.case_names = self._load_merged(csv_paths)
        self.num_nodes = self._count(df, "pos_x_")
        self.n_edges = self.num_nodes - 1
        self.ell = ROPE_LENGTH_MM / self.n_edges
        missing = set(self.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        df = df.sort_values(["case_id", "trajectory_id", "action_idx", "step_idx"]
                            ).reset_index(drop=True)

        pos = np.stack([df[[f"pos_x_{i}", f"pos_y_{i}"]].to_numpy(np.float32)
                        for i in range(self.num_nodes)], axis=1)     # [R, N, 2]
        vel = None
        if use_rates and f"vel_x_0" in df.columns:
            vel = np.stack([df[[f"vel_x_{i}", f"vel_y_{i}"]].to_numpy(np.float32)
                            for i in range(self.num_nodes)], axis=1)

        states = positions_to_angle_state(pos, vel, self.ell)
        if not use_rates:
            states = states[:, :2 * self.n_edges]
        self.state_dim = states.shape[1]
        self.control_dim = 2
        self.anchor = torch.tensor(pos[0, -1], dtype=torch.float32)  # fixed node
        self.positions = torch.from_numpy(pos)

        seg = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)
        print(f"segment length in data: {seg.min():.4f} / {seg.max():.4f} mm "
              f"(rest {self.ell:.4f}, max strain {100*(seg.max()-self.ell)/self.ell:+.3f}%)")

        cmd = df[["cmd_dx", "cmd_dy"]].to_numpy(np.float32)
        traj = df["trajectory_id"].to_numpy(np.int64)
        case = df["case_id"].to_numpy(np.int64)

        # cos/sin are already bounded in [-1,1]; only the rate block needs scaling.
        # Normalising the trig block would destroy the unit-norm structure the
        # decoder relies on.
        self.normalize = normalize
        self.state_mean = np.zeros((1, self.state_dim), dtype=np.float32)
        self.state_std = np.ones((1, self.state_dim), dtype=np.float32)
        if normalize:
            if use_rates:
                sl = slice(2 * self.n_edges, 3 * self.n_edges)
                s = states[:, sl].std(axis=0, keepdims=True)
                self.state_std[0, sl] = np.where(s < 1e-6, 1.0, s)
                states[:, sl] = states[:, sl] / self.state_std[0, sl]
            self.cmd_scale = float(np.abs(cmd).max()) + 1e-8
            cmd = cmd / self.cmd_scale
        else:
            self.cmd_scale = 1.0

        self.states = torch.from_numpy(states)
        self.cmds = torch.from_numpy(cmd)

        key = case.astype(np.int64) * (traj.max() + 1) + traj
        starts, bnd = [], np.flatnonzero(np.diff(key)) + 1
        for lo, hi in zip(np.r_[0, bnd], np.r_[bnd, len(key)]):
            if hi - lo > horizon:
                starts.append(np.arange(lo, hi - horizon))
        if not starts:
            raise ValueError(f"No trajectory longer than horizon={horizon}.")
        self.window_starts = torch.from_numpy(np.concatenate(starts))
        self.traj_ids = torch.from_numpy(traj[self.window_starts.numpy()])
        self.case_ids = torch.from_numpy(case[self.window_starts.numpy()])

    @staticmethod
    def _load_merged(csv_paths):
        paths = [csv_paths] if isinstance(csv_paths, str) else list(csv_paths)
        frames, names = [], []
        for i, p in enumerate(paths):
            d = pd.read_csv(p)
            d["trajectory_id"] = d["trajectory_id"].astype(np.int64) + i * 1_000_000
            d["case_id"] = i
            frames.append(d)
            names.append(Path(p).parent.name or Path(p).stem)
        return pd.concat(frames, axis=0, ignore_index=True), names

    @staticmethod
    def _count(df, prefix):
        pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        idxs = sorted(int(m.group(1)) for m in map(pat.match, df.columns) if m)
        if not idxs:
            raise ValueError(f"No columns matching '{prefix}'")
        return len(idxs)

    def __len__(self):
        return self.window_starts.numel()

    def __getitem__(self, idx):
        s = int(self.window_starts[idx])
        H = self.horizon
        return (self.states[s], self.cmds[s + 1: s + 1 + H],
                self.states[s + 1: s + 1 + H], self.positions[s + 1: s + 1 + H],
                self.positions[s], self.case_ids[idx])

    def decode_positions(self, x, device=None):
        """Angle state (normalised) -> node positions in mm. [..., D] -> [..., N, 2]"""
        ne = self.n_edges
        return angles_to_positions(x[..., :ne], x[..., ne:2 * ne],
                                   self.anchor, self.ell)


def split_by_trajectory(dataset, val_frac=0.2, test_frac=0.1, seed=0):
    traj, case = dataset.traj_ids.numpy(), dataset.case_ids.numpy()
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(case):
        uniq = np.unique(traj[case == c]); rng.shuffle(uniq)
        nv = max(1, int(round(len(uniq) * val_frac))) if val_frac > 0 else 0
        nt = max(1, int(round(len(uniq) * test_frac))) if test_frac > 0 else 0
        if nv + nt >= len(uniq):
            raise ValueError(f"Case {c}: too few trajectories for these fractions.")
        va.append(uniq[:nv]); te.append(uniq[nv:nv + nt]); tr.append(uniq[nv + nt:])
    sub = lambda parts: Subset(
        dataset, np.flatnonzero(np.isin(traj, np.concatenate(parts))).tolist())
    return sub(tr), sub(va), sub(te)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BilinearOperator(nn.Module):
    """z_{k+1} = A z_k + B u_k + sum_j u^(j) N_j z_k.

    A starts near identity: one sim step barely changes the state, so an untrained
    operator close to the identity keeps early rollouts stable, and it leaves the
    unit-circle modes the tip integrator needs available from the start.

    N is initialised small: the bilinear correction should start as a perturbation
    on the linear response, not compete with it.
    """

    def __init__(self, koopman_dim, control_dim, bilinear=True):
        super().__init__()
        self.A = nn.Parameter(torch.eye(koopman_dim)
                              + 0.01 * torch.randn(koopman_dim, koopman_dim))
        self.B = nn.Parameter(0.01 * torch.randn(koopman_dim, control_dim))
        self.N = (nn.Parameter(1e-3 * torch.randn(control_dim, koopman_dim, koopman_dim))
                  if bilinear else None)
        self.register_buffer("_u_iter", torch.randn(koopman_dim))

    def forward(self, z, u):
        out = z @ self.A.T + u @ self.B.T
        if self.N is not None:
            # state-dependent control: the response to a tip command depends on the
            # rope's current shape, which a constant B cannot express
            # As normal matrix multiplication, this is sum_j u^(j) N_j z, with N_j the j-th slice of N.
            out = out  + torch.einsum("bi,ijk,bk->bj", u, self.N, z)  # [B,K] = [B,K] + [B,J] @ [J,K,K] @ [B,K]
        return out

    def spectral_norm(self, n_iter=2):
        u = self._u_iter
        with torch.no_grad():
            for _ in range(n_iter):
                v = torch.mv(self.A.T, u); v = v / (v.norm() + 1e-12)
                u = torch.mv(self.A, v);   u = u / (u.norm() + 1e-12)
            self._u_iter.copy_(u)
        return torch.mv(self.A.T, u).norm()

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.A.detach().cpu()).abs().max().item()


class KoopmanModel(nn.Module):
    def __init__(self, state_dim, control_dim, koopman_dim,
                 hidden_dim=512, num_hidden=2, bilinear=True):
        super().__init__()
        self.encoder = MLP([state_dim] + [hidden_dim] * num_hidden + [koopman_dim])
        self.decoder = MLP([koopman_dim] + [hidden_dim] * num_hidden + [state_dim])
        self.operator = BilinearOperator(koopman_dim, control_dim, bilinear)

    def rollout(self, x0, us):
        z = self.encoder(x0)
        x_hat, z_hat = [], []
        for h in range(us.size(1)):
            z = self.operator(z, us[:, h])
            z_hat.append(z)
            x_hat.append(self.decoder(z))
        return torch.stack(x_hat, 1), torch.stack(z_hat, 1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class Loss:
    def __init__(self, dataset, w_pred=1.0, w_recon=0.5, w_lin=0.1,
                 w_node0=0.0, w_diss=0.0, w_stab=0.0, gamma=1.0, stab_margin=1.0):
        self.ds = dataset
        self.w_pred, self.w_recon, self.w_lin = w_pred, w_recon, w_lin
        self.w_node0, self.w_diss, self.w_stab = w_node0, w_diss, w_stab
        self.gamma, self.stab_margin = gamma, stab_margin
        self.ne = dataset.n_edges

    def _disc(self, H, device):
        w = self.gamma ** torch.arange(H, device=device, dtype=torch.float32)
        return w / w.sum()

    def _pos(self, x):
        """Angle state -> positions in mm, unit-normalised so edges are exactly ell."""
        return angles_to_positions(x[..., :self.ne], x[..., self.ne:2 * self.ne],
                                   self.ds.anchor, self.ds.ell)

    def prediction_mm(self, x_hat, pos_true):
        """Rollout error in MILLIMETRES. Training on the reported quantity avoids
        the mis-weighting a plain MSE over theta would introduce: an angle error
        near the anchor moves the far end much more than one near the tip."""
        p = self._pos(x_hat)                                  # [B,H,N,2]
        per_step = ((p - pos_true) ** 2).sum(-1).mean(dim=(0, 2))   # [H], mm^2
        return (per_step * self._disc(p.size(1), p.device)).sum(), per_step.detach()

    def reconstruction(self, model, x0, xs):
        flat = torch.cat([x0.unsqueeze(1), xs], 1).reshape(-1, x0.size(-1))
        return ((model.decoder(model.encoder(flat)) - flat) ** 2).mean()

    def linearity(self, model, z_hat, xs):
        z_true = model.encoder(xs.reshape(-1, xs.size(-1))).view_as(z_hat)
        per_step = ((z_hat - z_true) ** 2).mean(dim=(0, 2))
        return (per_step * self._disc(z_hat.size(1), z_hat.device)).sum()

    def node0(self, x_hat, pos0, us):
        """Node 0 is kinematically driven, so p0_{k+h} = p0_k + sum_{j<h} u_j
        EXACTLY. A shape-averaged loss barely penalises getting this wrong -- it is
        2 of 60 coordinates -- yet a few percent of gain error compounds into tens
        of mm over a 300-step trace."""
        if self.w_node0 == 0.0:
            return torch.zeros((), device=x_hat.device)
        u_mm = us * self.ds.cmd_scale
        target = pos0[:, 0:1, :] + torch.cumsum(u_mm, dim=1)        # [B,H,2]
        return ((self._pos(x_hat)[:, :, 0, :] - target) ** 2).sum(-1).mean()

    def dissipation(self, x_hat, us):
        """With u = 0 the rod can only shed bending energy. U = sum kappa^2, and
        kappa is the turning angle between consecutive edges -- cheap in this
        representation. Physically grounded contractivity, replacing the sigma_max
        hinge that flattened the integrator modes in earlier runs."""
        if self.w_diss == 0.0:
            return torch.zeros((), device=x_hat.device)
        c, s = x_hat[..., :self.ne], x_hat[..., self.ne:2 * self.ne]
        n = torch.sqrt(c ** 2 + s ** 2).clamp_min(1e-8)
        c, s = c / n, s / n
        # turning angle between edge i and i+1 via the 2-d cross/dot
        cross = c[..., :-1] * s[..., 1:] - s[..., :-1] * c[..., 1:]
        dot = c[..., :-1] * c[..., 1:] + s[..., :-1] * s[..., 1:]
        U = (torch.atan2(cross, dot) ** 2).sum(-1)                  # [B,H]
        free = (us.abs().sum(-1) < 1e-8).float()                    # autonomous steps
        rise = torch.relu(U[:, 1:] - U[:, :-1]) * free[:, 1:]
        return rise.sum() / free[:, 1:].sum().clamp_min(1.0)

    def stability(self, model):
        if self.w_stab == 0.0:
            return torch.zeros((), device=model.operator.A.device)
        return torch.relu(model.operator.spectral_norm() - self.stab_margin) ** 2

    def __call__(self, model, x0, us, xs, pos_xs, pos0):
        x_hat, z_hat = model.rollout(x0, us)
        pred, per_step = self.prediction_mm(x_hat, pos_xs)
        recon = self.reconstruction(model, x0, xs)
        lin = self.linearity(model, z_hat, xs)
        n0 = self.node0(x_hat, pos0, us)
        dis = self.dissipation(x_hat, us)
        stab = self.stability(model)
        total = (self.w_pred * pred + self.w_recon * recon + self.w_lin * lin
                 + self.w_node0 * n0 + self.w_diss * dis + self.w_stab * stab)
        parts = {"loss": total, "pred": pred, "recon": recon, "lin": lin,
                 "node0": n0, "diss": dis, "stab": stab}
        return total, parts, x_hat, per_step


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, loss_fn, dataset, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    keys = ("loss", "pred", "recon", "lin", "node0", "diss", "stab")
    sums = {k: 0.0 for k in keys}
    n, per_step_sum = 0, None
    rmse_1 = rmse_H = n0_err = 0.0
    per_case = {}

    for x0, us, xs, pos_xs, pos0, case in loader:
        x0, us, xs = x0.to(device), us.to(device), xs.to(device)
        pos_xs, pos0 = pos_xs.to(device), pos0.to(device)
        bs = x0.size(0)

        with torch.set_grad_enabled(train):
            total, parts, x_hat, per_step = loss_fn(model, x0, us, xs, pos_xs, pos0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k in keys:
            sums[k] += parts[k].item() * bs
        per_step_sum = per_step * bs if per_step_sum is None else per_step_sum + per_step * bs

        with torch.no_grad():
            p = loss_fn._pos(x_hat)
            e = ((p - pos_xs) ** 2).sum(-1)                          # [B,H,N] mm^2
            rmse_1 += torch.sqrt(e[:, 0].mean()).item() * bs
            rmse_H += torch.sqrt(e[:, -1].mean()).item() * bs
            n0_err += torch.sqrt(e[:, -1, 0].mean()).item() * bs
            for c in case.unique():
                m = case == c
                acc = per_case.setdefault(int(c), [0.0, 0])
                acc[0] += torch.sqrt(e[m].mean()).item() * int(m.sum())
                acc[1] += int(m.sum())
        n += bs

    out = {k: v / n for k, v in sums.items()}
    out["rmse_1_mm"] = rmse_1 / n
    out["rmse_H_mm"] = rmse_H / n
    out["node0_H_mm"] = n0_err / n
    out["per_step"] = (per_step_sum / n).cpu().numpy()
    out["per_case"] = {c: s / cnt for c, (s, cnt) in per_case.items()}
    return out


def main():
    p = argparse.ArgumentParser(description="Angle-state bilinear Koopman for rope sim data.")
    p.add_argument("--csv", type=str, nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--koopman-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-hidden", type=int, default=2)
    p.add_argument("--no-rates", action="store_true",
                   help="drop thetadot: angles only, a quasi-static approximation")
    p.add_argument("--no-bilinear", action="store_true",
                   help="plain z+ = Az + Bu; keeps an MPC built on this a QP")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--w-pred", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=0.5)
    p.add_argument("--w-lin", type=float, default=0.1)
    p.add_argument("--w-node0", type=float, default=0.0,
                   help="supervise node 0's exact kinematic trajectory")
    p.add_argument("--w-diss", type=float, default=0.0,
                   help="penalise bending-energy INCREASE on autonomous (u=0) steps")
    p.add_argument("--w-stab", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--results-dir", type=str, default="training_results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"exp_angle_{int(time())}"
    results_file = open(results_dir / f"{exp_name}.txt", "w")
    writer = SummaryWriter(log_dir=str(results_dir / "runs" / exp_name))

    def log(msg):
        print(msg); results_file.write(msg + "\n"); results_file.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    log(f"Args: {vars(args)}")

    dataset = RopeAngleDataset(args.csv, horizon=args.horizon,
                               normalize=not args.no_normalize,
                               use_rates=not args.no_rates)
    log(f"Loaded {len(dataset)} windows (H={args.horizon}) | {dataset.num_nodes} nodes, "
        f"{dataset.n_edges} edges | state_dim {dataset.state_dim} "
        f"({'cos,sin,thetadot' if not args.no_rates else 'cos,sin'}) | device {device}")
    log(f"Cases: {', '.join(f'{i}={n}' for i, n in enumerate(dataset.case_names))}")
    log(f"anchor {dataset.anchor.numpy()} | ell {dataset.ell:.4f} mm")

    train_set, val_set, test_set = split_by_trajectory(
        dataset, args.val_frac, args.test_frac, args.seed)
    log(f"Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)} windows")

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
    train_loader, val_loader, test_loader = mk(train_set, True), mk(val_set, False), mk(test_set, False)

    model = KoopmanModel(dataset.state_dim, dataset.control_dim, args.koopman_dim,
                         args.hidden_dim, args.num_hidden,
                         bilinear=not args.no_bilinear).to(device)
    log(f"Parameters: {sum(q.numel() for q in model.parameters()):,} "
        f"| bilinear: {not args.no_bilinear}")

    loss_fn = Loss(dataset, args.w_pred, args.w_recon, args.w_lin,
                   args.w_node0, args.w_diss, args.w_stab, args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, loss_fn, dataset, optimizer)
        va = run_epoch(model, val_loader, device, loss_fn, dataset, None)
        A = model.operator.A.detach().cpu().numpy()
        ev = np.abs(np.linalg.eigvals(A))
        rho, n_int = ev.max(), int((ev > 0.98).sum())

        flag = ""
        if va["rmse_H_mm"] < best_val:
            best_val, flag = va["rmse_H_mm"], "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        log(f"epoch {epoch:03d} | train pred {tr['pred']:.3f} recon {tr['recon']:.5f} "
            f"lin {tr['lin']:.5f} n0 {tr['node0']:.3f} | val rmse "
            f"{va['rmse_1_mm']:.3f}/{va['rmse_H_mm']:.3f} mm  node0 {va['node0_H_mm']:.3f} mm "
            f"| rho {rho:.4f} int {n_int}{flag}")

        for split, m in (("train", tr), ("val", va)):
            for k in ("loss", "pred", "recon", "lin", "node0", "diss",
                      "rmse_1_mm", "rmse_H_mm", "node0_H_mm"):
                writer.add_scalar(f"{split}/{k}", m[k], epoch)
        writer.add_scalar("spectral_radius", rho, epoch)
        writer.add_scalar("integrator_modes", n_int, epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
        ck = results_dir / "checkpoints" / exp_name; ck.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), ck / "koopman_enc.pt")
        torch.save(model.decoder.state_dict(), ck / "koopman_dec.pt")
        torch.save(model.operator.state_dict(), ck / "koopman_op.pt")

    te = run_epoch(model, test_loader, device, loss_fn, dataset, None)
    log(f"\nTest position RMSE {te['rmse_1_mm']:.3f} mm @h=1, "
        f"{te['rmse_H_mm']:.3f} mm @h={args.horizon} | node0 {te['node0_H_mm']:.3f} mm")
    log("Per-horizon test RMSE (mm): " +
        ", ".join(f"h{h+1}={np.sqrt(v):.3f}" for h, v in enumerate(te["per_step"])))
    if len(dataset.case_names) > 1:
        log("Per-case test RMSE (mm):")
        for c, v in sorted(te["per_case"].items()):
            log(f"    {dataset.case_names[c]:<24} {v:.3f}")

    A = model.operator.A.detach().cpu().numpy()
    ev = np.abs(np.linalg.eigvals(A))
    log(f"rho(A) {ev.max():.6f} | modes >0.98: {int((ev > 0.98).sum())} "
        f"| top 8: {np.round(np.sort(ev)[-8:], 4)}")
    if model.operator.N is not None:
        N = model.operator.N.detach().cpu().numpy()
        log(f"||N|| / ||A|| = {np.linalg.norm(N) / np.linalg.norm(A):.4f}  "
            f"(how much the bilinear term contributes)")

    results_file.close(); writer.close()


if __name__ == "__main__":
    main()