"""
neural_koopman.py

Koopman-with-control model for rope manipulation data collected from a SOFA simulation
(30 chain nodes, 29 edges, node 0 kinematically driven, node 29 fixed).

Pure-MLP baseline: no graph networks, no torch_geometric. The rope state is flattened
into one vector per timestep; chain topology is implicit in node ordering and the model
must learn it from data.

Data convention (IMPORTANT)
---------------------------
In the collection script the command is applied *before* the physics step that produces
the row's state:

    target += cmd ; write node0 = target ; env.step() ; record positions, cmd

So row k's state is the RESULT of applying row k's command. The input driving the
transition x_{k-1} -> x_k is therefore cmd on row k, NOT cmd on row k-1. This module
pairs accordingly:  u_k = cmd[k+1]  for the transition (row k -> row k+1).

Because the command is now logged per step (zero on held steps) rather than repeated
across an action block, transitions may cross action_idx boundaries: every row's input
is fully described by its own cmd. Windows are therefore cut per trajectory_id, not per
(trajectory_id, action_idx), which recovers the samples the old pairing threw away.

State
-----
The Markov state is positions + velocities (a rope is second-order: the same shape with
different node velocities evolves differently). Derived quantities -- curvature, edge
vectors, edge lengths -- are deterministic functions of positions and are OFF by default;
they add dimension without adding information. Enable them with --features to test them
as observables.

Model
-----
  encoder       : MLP,    x_k              -> z_k                    (lifted / Koopman space)
  input_encoder : MLP,    u_k              -> u_k_lifted              (raw cmd_dx, cmd_dy lifted
                                                                        into a higher-dim control space)
  operator      : affine, z_k, u_k_lifted  -> z_k+1 = A z_k + B u_k_lifted
  decoder       : MLP,    z                -> x_hat                  (full state vector)

Note: lifting u through a nonlinear input_encoder makes z_k+1 nonlinear in the *raw* command,
trading away the linear-in-u convexity a plain B u_k term would give an MPC built on this
model (see the operator's docstring). Kept because a 2-d (cmd_dx, cmd_dy) command is too
low-dimensional to expect a single linear map into a 256-d Koopman space to capture its
effect well; the lift gives the operator more directions to place that control authority in.

Losses (see KoopmanLoss)
------------------------
  pred   : multi-step rollout error, decode(z_k+h) vs x_k+h, discounted over h = 1..H
  recon  : autoencoder consistency, decode(encode(x_k)) vs x_k
  lin    : latent linearity, rolled-forward z_k+h vs encode(x_k+h)
  stab   : hinge on the spectral norm of A, keeps long rollouts from diverging

Usage
-----
  python neural_koopman.py --csv case_1/*.csv case_5/*.csv --horizon 10 --epochs 100
"""

import argparse
import re
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from pygments.unistring import Lo
from sympy import gamma
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
# name -> (column prefixes, "node" or "edge")
# "pos" must stay first so the position block is a known slice of the state vector,
# which is what the mm-scale error metric and any future linear readout rely on.
FEATURE_GROUPS = {
    "pos":    (("pos_x_", "pos_y_"), "node"),
    "vel":    (("vel_x_", "vel_y_"), "node"),
}
DEFAULT_FEATURES = ("pos", "vel") 

ROPE_NUM_NODES = 30
ROPE_LENGTH_MM = 300.0
# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class RopeKoopmanDataset(Dataset):
    """
    Builds fixed-length rollout windows from rope-sim CSV log(s).

    __getitem__ returns:
        x0   : [D]        state at the window start
        us   : [H, 2]     commands driving each of the H transitions
        xs   : [H, D]     states after each transition (targets)
        case : []         index of the source CSV (for per-case reporting)

    With horizon=1 this reduces to the usual (x_k, u_k, x_k+1) triple.
    """

    REQUIRED = ("trajectory_id", "action_idx", "step_idx", "cmd_dx", "cmd_dy")

    def __init__(self, csv_paths, features=DEFAULT_FEATURES, horizon=1,
                 normalize=True, clamp_tol_mm=None):
        super().__init__()
        if "pos" not in features:
            raise ValueError("'pos' must be included in --features (used for the mm error metric).")
        features = tuple(sorted(features, key=lambda f: list(FEATURE_GROUPS).index(f)))
        self.features = features
        self.horizon = horizon

        df, self.case_names = self._load_merged(csv_paths)
        self.num_nodes = self._count(df, "pos_x_")
        self.num_edges = self._count(df, "edge_dx_")
        self._validate(df)

        df = df.sort_values(["case_id", "trajectory_id", "action_idx", "step_idx"]).reset_index(drop=True)
        self._report_clamping(df, clamp_tol_mm)

        # ---- assemble the state matrix, one contiguous block per feature group ----
        blocks, self.feature_slices, offset = [], {}, 0
        for name in features:
            prefixes, kind = FEATURE_GROUPS[name]
            n = self.num_nodes if kind == "node" else self.num_edges
            cols = [f"{p}{i}" for p in prefixes for i in range(n)]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Feature group '{name}' needs missing columns, e.g. {missing[:3]}")
            # [rows, P, n] -> [rows, n, P] so each node's/edge's features stay contiguous
            arr = df[cols].to_numpy(np.float32).reshape(-1, len(prefixes), n).transpose(0, 2, 1)
            arr = arr.reshape(len(df), -1)
            blocks.append(arr)
            self.feature_slices[name] = slice(offset, offset + arr.shape[1])
            offset += arr.shape[1]

        states = np.concatenate(blocks, axis=1)                     # [R, D]
        self.state_dim = states.shape[1]
        self.control_dim = 2
        cmd = df[["cmd_dx", "cmd_dy"]].to_numpy(np.float32)         # [R, 2]
        traj = df["trajectory_id"].to_numpy(np.int64)
        case = df["case_id"].to_numpy(np.int64)

        # ---- normalisation ---------------------------------------------------
        # Fitted on all rows. Constant columns (e.g. the fixed node's coordinates,
        # which never move) get std 1 so they normalise to a constant 0 rather than
        # being blown up by a near-zero divisor.
        self.normalize = normalize
        if normalize:
            self.state_mean = states.mean(axis=0, keepdims=True)
            std = states.std(axis=0, keepdims=True)
            self.state_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            states = (states - self.state_mean) / self.state_std
            self.cmd_scale = float(np.abs(cmd).max()) + 1e-8
            cmd = cmd / self.cmd_scale
        else:
            self.state_mean = np.zeros((1, self.state_dim), dtype=np.float32)
            self.state_std = np.ones((1, self.state_dim), dtype=np.float32)
            self.cmd_scale = 1.0

        self.states = torch.from_numpy(states)
        self.cmds = torch.from_numpy(cmd)
        self.case_ids_row = torch.from_numpy(case)

        # ---- window index ----------------------------------------------------
        # Windows are cut per trajectory (commands are per-step, so action boundaries
        # are ordinary transitions) and never straddle a trajectory or a case.
        key = case.astype(np.int64) * (traj.max() + 1) + traj
        starts = []
        boundaries = np.flatnonzero(np.diff(key)) + 1
        for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(key)]):
            if hi - lo > horizon:
                starts.append(np.arange(lo, hi - horizon))
        if not starts:
            raise ValueError(f"No trajectory is longer than horizon={horizon}.")
        self.window_starts = torch.from_numpy(np.concatenate(starts))
        self.traj_ids = torch.from_numpy(traj[self.window_starts.numpy()])
        self.case_ids = torch.from_numpy(case[self.window_starts.numpy()])

        # Position block, kept for reporting error in millimetres
        self.pos_slice = self.feature_slices["pos"]

    # -- construction helpers ------------------------------------------------ #
    @staticmethod
    def _load_merged(csv_paths):
        """Loads one CSV or merges several. Each file is a separate case; trajectory ids
        are offset per file so ids that restart at 0 in every file don't collide."""
        paths = [csv_paths] if isinstance(csv_paths, str) else list(csv_paths)
        frames, names = [], []
        for i, p in enumerate(paths):
            d = pd.read_csv(p)
            d["trajectory_id"] = d["trajectory_id"].astype(np.int64) + i * 1_000_000
            d["case_id"] = i
            frames.append(d)
            # case_<n>_<name>/collected_....csv -> "case_<n>_<name>"
            names.append(Path(p).parent.name or Path(p).stem)
        return pd.concat(frames, axis=0, ignore_index=True), names

    @staticmethod
    def _count(df, prefix):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        idxs = sorted(int(m.group(1)) for m in map(pattern.match, df.columns) if m)
        if not idxs:
            raise ValueError(f"No columns found matching prefix '{prefix}'")
        if idxs != list(range(len(idxs))):
            raise ValueError(f"Columns '{prefix}*' are not contiguous from 0: {idxs}")
        return len(idxs)

    def _validate(self, df):
        missing = set(self.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    def _report_clamping(self, df, tol_mm):
        """Flag rows where node 0 did not reach its commanded target.

        clamp_to_reach=True in the collector silently saturates node 0 when the target
        leaves the rope's reach, so the logged command describes motion that never
        happened. Those rows are misleading supervision for B.
        """
        self.clamped_frac = None
        if tol_mm is None or "target_x" not in df.columns:
            return
        err = np.hypot(df["target_x"].to_numpy() - df["pos_x_0"].to_numpy(),
                       df["target_y"].to_numpy() - df["pos_y_0"].to_numpy())
        self.clamped_frac = float((err > tol_mm).mean())

    # -- interface ----------------------------------------------------------- #
    def __len__(self):
        return self.window_starts.numel()

    def __getitem__(self, idx):
        s = int(self.window_starts[idx])
        H = self.horizon
        x0 = self.states[s]
        xs = self.states[s + 1: s + 1 + H]          # [H, D]
        us = self.cmds[s + 1: s + 1 + H]            # [H, 2]  (cmd on row k+1 drives k -> k+1)
        return x0, us, xs, self.case_ids[idx]

    # -- units --------------------------------------------------------------- #
    def position_rmse_mm(self, pred_states, true_states):
        """RMSE over node positions in millimetres, undoing normalisation."""
        sl = self.pos_slice
        std = torch.as_tensor(self.state_std[0, sl], device=pred_states.device)
        diff = (pred_states[..., sl] - true_states[..., sl]) * std
        return torch.sqrt((diff ** 2).mean()).item()


def split_by_trajectory(dataset, val_frac=0.2, test_frac=0.1, seed=0):
    """Splits by trajectory, stratified by case.

    Splitting on trajectories (not windows) matters because windows overlap by H, so a
    per-sample split would leak almost perfectly.

    Stratifying by case matters when several CSVs are merged: the cases have very
    different row counts (case 8 is 200 rows/traj, case 1 is 100), so a pooled shuffle
    can leave a case absent from val or test and make the per-case report meaningless.
    Each case contributes the same *fraction* of its own trajectories to each split.
    """
    traj = dataset.traj_ids.numpy()
    case = dataset.case_ids.numpy()
    rng = np.random.RandomState(seed)
    train_ids, val_ids, test_ids = [], [], []

    for c in np.unique(case):
        uniq = np.unique(traj[case == c])
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * val_frac))) if val_frac > 0 else 0
        n_test = max(1, int(round(len(uniq) * test_frac))) if test_frac > 0 else 0
        if n_val + n_test >= len(uniq):
            raise ValueError(
                f"Case {c} has only {len(uniq)} trajectories; val+test fractions leave "
                f"nothing to train on. Collect more trajectories or lower the fractions."
            )
        val_ids.append(uniq[:n_val])
        test_ids.append(uniq[n_val:n_val + n_test])
        train_ids.append(uniq[n_val + n_test:])

    sub = lambda parts: Subset(
        dataset, np.flatnonzero(np.isin(traj, np.concatenate(parts))).tolist())
    return sub(train_ids), sub(val_ids), sub(test_ids)


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


class KoopmanEncoder(nn.Module):
    """Lifts the flat rope state into Koopman space: x -> z."""

    def __init__(self, state_dim, koopman_dim, hidden_dim=512, num_hidden=2):
        super().__init__()
        self.mlp = MLP([state_dim] + [hidden_dim] * num_hidden + [koopman_dim])

    def forward(self, x):
        return self.mlp(x)


class KoopmanDecoder(nn.Module):
    """Projects Koopman space back to the full state vector: z -> x_hat."""

    def __init__(self, koopman_dim, state_dim, hidden_dim=512, num_hidden=2):
        super().__init__()
        self.mlp = MLP([koopman_dim] + [hidden_dim] * num_hidden + [state_dim])

    def forward(self, z):
        return self.mlp(z)

    # class InputEncoder(nn.Module):
    #     """Lifts the raw 2-d command (cmd_dx, cmd_dy) into the control space the
    #     KoopmanOperator consumes: u -> u_lifted. This is what makes z_k -> z_k+1
    #     nonlinear in the raw command even though KoopmanOperator itself is affine.
    #     """

    #     # hidden_dim is much smaller than KoopmanEncoder's (64 vs 512): the input here
    #     # is only 2-d, so a small MLP is enough to lift it.
    #     def __init__(self, control_dim, encoded_dim, hidden_dim=64, num_hidden=2):
    #         super().__init__()
    #         self.mlp = MLP([control_dim] + [hidden_dim] * num_hidden + [encoded_dim])

    #     def forward(self, u):
    #         return self.mlp(u)


class KoopmanOperator(nn.Module):
    """Affine latent dynamics: z_k+1 = A z_k + B u_k, where u_k is the *lifted* control
    (InputEncoder(raw cmd_dx, cmd_dy)), not the raw 2-d command.

    A starts near identity: one sim step changes the state only slightly, so the
    untrained operator is roughly the identity map, which keeps early rollouts stable.
    The bias c absorbs command-independent drift (gravity, ongoing settling). The operator
    itself stays affine in whatever control vector it's given -- it's the InputEncoder
    upstream that makes the overall z_k -> z_k+1 map nonlinear in the raw command; see the
    module docstring's Model section for that tradeoff.
    """

    def __init__(self, koopman_dim, control_dim, bias=True):
        super().__init__()
        self.A = nn.Parameter(torch.eye(koopman_dim) + 0.01 * torch.randn(koopman_dim, koopman_dim))
        self.B = nn.Parameter(0.01 * torch.randn(koopman_dim, control_dim))
        # self.c = nn.Parameter(torch.zeros(koopman_dim)) if bias else None
        self.register_buffer("_u_iter", torch.randn(koopman_dim))

    def forward(self, z_k, u_k):
        z = z_k @ self.A.T + u_k @ self.B.T
        #return z if self.c is None else z + self.c
        return z

    def spectral_norm(self, n_iter=2):
        """Largest singular value of A via power iteration.

        Differentiable and MPS-friendly, unlike torch.linalg.svdvals / eigvals. Used for
        the stability penalty; sigma_max >= rho(A), so bounding it bounds the spectral
        radius too (conservatively).
        """
        u = self._u_iter
        with torch.no_grad():
            for _ in range(n_iter):
                v = torch.mv(self.A.T, u); v = v / (v.norm() + 1e-12)
                u = torch.mv(self.A, v);   u = u / (u.norm() + 1e-12)
            self._u_iter.copy_(u)
        v = torch.mv(self.A.T, u)
        return v.norm()

    @torch.no_grad()
    def spectral_radius(self):
        """Largest |eigenvalue| of A. >1 means latent rollouts diverge.

        Evaluated on CPU: torch.linalg.eigvals has no MPS kernel, and A is small.
        """
        return torch.linalg.eigvals(self.A.detach().cpu()).abs().max().item()


class KoopmanModel(nn.Module):
    def __init__(self, state_dim, control_dim, koopman_dim, encoded_control_dim=16,
                 hidden_dim=256, num_hidden=2, control_hidden_dim=64, control_num_hidden=2,
                 operator_bias=True):
        super().__init__()
        self.encoder = KoopmanEncoder(state_dim, koopman_dim, hidden_dim, num_hidden)
        # self.input_encoder = InputEncoder(control_dim, encoded_control_dim,
        #                                    control_hidden_dim, control_num_hidden)
        self.operator = KoopmanOperator(koopman_dim, encoded_control_dim, bias=operator_bias)
        self.decoder = KoopmanDecoder(koopman_dim, state_dim, hidden_dim, num_hidden)

    def rollout(self, x0, us):
        """Encode once, then roll the latent forward under the recorded commands.

        Args:
            x0: [B, D]      initial state
            us: [B, H, 2]   raw commands for each transition (lifted internally)
        Returns:
            x_hat: [B, H, D]  decoded predicted states
            z_hat: [B, H, Z]  latent trajectory
            z0:    [B, Z]     initial latent
        """
        z = z0 = self.encoder(x0)
        x_hat, z_hat = [], []
        for h in range(us.size(1)):
            # u_lifted = self.input_encoder(us[:, h])
            u_lifted = us[:, h]  # Directly use the raw command - no input encoder
            z = self.operator(z, u_lifted)
            z_hat.append(z)
            x_hat.append(self.decoder(z))
        return torch.stack(x_hat, 1), torch.stack(z_hat, 1), z0


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class Loss:
    """Each loss term as its own function, plus a weighted total.

    prediction     -- what you ultimately care about: does the decoded rollout track the
                      true rope? Discounted over the horizon so near-term steps dominate
                      while distant ones still constrain the operator.
    reconstruction -- keeps encoder/decoder a faithful autoencoder, so the latent cannot
                      "solve" prediction by discarding the state.
    linearity      -- the actual Koopman assumption: rolling z forward linearly must land
                      where encoding the true future state lands. This is the term to
                      watch when asking whether a linear operator is viable at all.
    stability      -- hinge on sigma_max(A); without it a model can ace one-step error
                      while its rollout diverges.
    """

    def __init__(self, w_pred=1.0, w_recon=0.5, w_lin=0.1, w_stab=0.0, w_seg=0.0,
                 pos_slice=None, pos_mean=None, pos_std=None, rest_len_mm=None,
                 gamma=1.0, stab_margin=1.0):
        self.w_pred, self.w_recon = w_pred, w_recon
        self.w_lin, self.w_stab = w_lin, w_stab
        self.w_seg = w_seg

        # Dataset params
        self.pos_slice = pos_slice
        self.rest_len_mm = rest_len_mm
        if pos_mean is None or pos_std is None or pos_slice is None:
            self._pos_mean = self._pos_std = None
        else:
            self._pos_mean = torch.as_tensor(
                np.asarray(pos_mean)[0, pos_slice], dtype=torch.float32)
            self._pos_std = torch.as_tensor(
                np.asarray(pos_std)[0, pos_slice], dtype=torch.float32)

        self.gamma, self.stab_margin = gamma, stab_margin
        self.mse = nn.MSELoss(reduction="none")

    def _discount(self, H, device):
        w = self.gamma ** torch.arange(H, device=device, dtype=torch.float32)
        return w / w.sum()

    # -- individual terms ---------------------------------------------------- #
    def prediction(self, x_hat, xs):
        """Multi-step decoded rollout error. x_hat, xs: [B, H, D]."""
        per_step = self.mse(x_hat, xs).mean(dim=(0, 2))              # [H]
        
        return (per_step * self._discount(x_hat.size(1), x_hat.device)).sum(), per_step.detach()

    def reconstruction(self, model, x0, xs):
        """Autoencoder consistency on every state in the window."""
        x_all = torch.cat([x0.unsqueeze(1), xs], dim=1)              # [B, H+1, D]
        flat = x_all.reshape(-1, x_all.size(-1))
        return self.mse(model.decoder(model.encoder(flat)), flat).mean()

    def linearity(self, model, z_hat, xs):
        """Rolled-forward latent vs the encoding of the true future state."""
        z_true = model.encoder(xs.reshape(-1, xs.size(-1))).view_as(z_hat)
        per_step = self.mse(z_hat, z_true).mean(dim=(0, 2))          # [H]
        return (per_step * self._discount(z_hat.size(1), z_hat.device)).sum()

    def stability(self, model):
        """Hinge penalty once sigma_max(A) exceeds the margin."""
        if self.w_stab == 0.0:
            return torch.zeros((), device=model.operator.A.device)
        return torch.relu(model.operator.spectral_norm() - self.stab_margin) ** 2

    def segment(self, x_hat):
        """Inextensibility penalty on the predicted chain.

        Denormalises the position block, takes edge vectors between consecutive
        nodes, and penalises departure from the rest segment length in mm. The
        position block is node-major ([x0,y0,x1,y1,...]), matching how
        RopeKoopmanDataset assembles it.
        """
        if self.w_seg == 0.0 or self.pos_slice is None:
            return torch.zeros((), device=x_hat.device)
        if self._pos_mean.device != x_hat.device:
            self._pos_mean = self._pos_mean.to(x_hat.device)
            self._pos_std = self._pos_std.to(x_hat.device)

        pos = x_hat[..., self.pos_slice] * self._pos_std + self._pos_mean   # Denormalize
        p = pos.reshape(*pos.shape[:-1], -1, 2)                             # [..., N, 2]
        edge_len = torch.linalg.norm(p[..., 1:, :] - p[..., :-1, :], dim=-1)
        return ((edge_len - self.rest_len_mm) ** 2).mean()

    # -- total --------------------------------------------------------------- #
    def __call__(self, model: KoopmanModel, x0, us, xs):
        x_hat, z_hat, _ = model.rollout(x0, us)
        pred, per_step = self.prediction(x_hat, xs) # multi-step rollout error: decode(rolled-forward z_k+h) vs x_k+h
        recon = self.reconstruction(model, x0, xs) # autoencoder consistency: decode(encode(x_k)) vs x_k
        lin = self.linearity(model, z_hat, xs) # latent linearity: encode(x_k+h) vs rolled-forward z_k+h
        stab = self.stability(model) # Hinge on sigma_max(A)
        seg = self.segment(x_hat) # Inextensibility penalty
        total = self.w_pred * pred + self.w_recon * recon + self.w_lin * lin + self.w_stab * stab + self.w_seg * seg
        parts = {"loss": total, "pred": pred, "recon": recon, "lin": lin, "stab": stab, "seg": seg}
        return total, parts, x_hat, per_step


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, loss_fn, dataset, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    sums = {k: 0.0 for k in ("loss", "pred", "recon", "lin", "stab", "seg")}
    n, per_step_sum = 0, None
    rmse_1, rmse_H = 0.0, 0.0
    per_case = {}

    for x0, us, xs, case in loader:
        x0, us, xs = x0.to(device), us.to(device), xs.to(device)
        bs = x0.size(0)

        with torch.set_grad_enabled(train):
            total, parts, x_hat, per_step = loss_fn(model, x0, us, xs)

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k in sums:
            sums[k] += parts[k].item() * bs
        per_step_sum = per_step * bs if per_step_sum is None else per_step_sum + per_step * bs
        rmse_1 += dataset.position_rmse_mm(x_hat[:, 0], xs[:, 0]) * bs
        rmse_H += dataset.position_rmse_mm(x_hat[:, -1], xs[:, -1]) * bs

        for c in case.unique():
            m = case == c
            acc = per_case.setdefault(int(c), [0.0, 0])
            acc[0] += torch.nn.functional.mse_loss(x_hat[m], xs[m]).item() * int(m.sum())
            acc[1] += int(m.sum())
        n += bs

    out = {k: v / n for k, v in sums.items()}
    out["rmse_1_mm"] = rmse_1 / n
    out["rmse_H_mm"] = rmse_H / n
    out["per_step"] = (per_step_sum / n).cpu().numpy()
    out["per_case"] = {c: s / cnt for c, (s, cnt) in per_case.items()}
    return out


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    p = argparse.ArgumentParser(description="Koopman-with-control model for rope sim data.")
    p.add_argument("--csv", type=str, nargs="+", required=True,
                   help="Path(s) to case CSV(s); each file is treated as a separate case.")
    p.add_argument("--features", type=str, nargs="+", default=list(DEFAULT_FEATURES),
                   choices=list(FEATURE_GROUPS),
                   help="State feature groups. 'pos' is required. Default: pos vel.")
    p.add_argument("--horizon", type=int, default=1,
                   help="Rollout length H for the multi-step loss (1 = one-step only).")
    p.add_argument("--koopman-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-hidden", type=int, default=2)
    p.add_argument("--encoded-control-dim", type=int, default=16,
                   help="Dim the raw 2-d (cmd_dx, cmd_dy) command is lifted into before the operator.")
    p.add_argument("--control-hidden-dim", type=int, default=64)
    p.add_argument("--control-num-hidden", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--w-pred", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=0.5)
    p.add_argument("--w-lin", type=float, default=0.1)
    p.add_argument("--w-stab", type=float, default=0.0, help="Weight on the sigma_max(A) hinge.")
    p.add_argument("--w-seg", type=float, default=0.0, help="Weight on the inextensibility penalty.")
    p.add_argument("--gamma", type=float, default=1.0, help="Horizon discount for pred/lin.")
    p.add_argument("--clamp-tol-mm", type=float, default=1.0,
                   help="Report the fraction of rows where node 0 missed its target by more than this.")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-operator-bias", action="store_true")
    p.add_argument("--results-dir", type=str,
                   default="training_results", help="Directory to save logs and TensorBoard runs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"exp_simple_{int(time())}"
    results_file = open(results_dir / f"{exp_name}.txt", "w")
    writer = SummaryWriter(log_dir=str(results_dir / "runs" / exp_name))

    def log(msg):
        print(msg)
        results_file.write(msg + "\n")
        results_file.flush()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    log(f"Args: {vars(args)}")

    dataset = RopeKoopmanDataset(args.csv, features=tuple(args.features), horizon=args.horizon,
                                 normalize=not args.no_normalize, clamp_tol_mm=args.clamp_tol_mm)
    log(f"Loaded {len(dataset)} windows (H={args.horizon}) | {dataset.num_nodes} nodes, "
        f"{dataset.num_edges} edges | features {list(dataset.features)} -> state_dim "
        f"{dataset.state_dim} | control_dim {dataset.control_dim} | device {device}")
    log(f"Cases: {', '.join(f'{i}={n}' for i, n in enumerate(dataset.case_names))}")
    if dataset.clamped_frac is not None:
        level = "WARNING: " if dataset.clamped_frac > 0.01 else ""
        log(f"{level}{dataset.clamped_frac:.2%} of rows have node 0 further than "
            f"{args.clamp_tol_mm} mm from its commanded target (clamped / still catching up).")

    train_set, val_set, test_set = split_by_trajectory(
        dataset, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    n_traj = len(np.unique(dataset.traj_ids.numpy()))
    log(f"Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)} "
        f"windows from {n_traj} trajectories")
    if n_traj < 20:
        log("WARNING: few trajectories -- held-out metrics will be noisy across seeds.")

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
    train_loader, val_loader, test_loader = mk(train_set, True), mk(val_set, False), mk(test_set, False)

    model = KoopmanModel(
        state_dim=dataset.state_dim,
        control_dim=dataset.control_dim,
        koopman_dim=args.koopman_dim,
        encoded_control_dim=args.encoded_control_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        control_hidden_dim=args.control_hidden_dim,
        control_num_hidden=args.control_num_hidden,
        operator_bias=not args.no_operator_bias,
    ).to(device)

    log(f"Parameters: {sum(q.numel() for q in model.parameters()):,}")

    loss_fn = Loss(w_pred=args.w_pred, w_recon=args.w_recon, w_lin=args.w_lin,
                   w_stab=args.w_stab, w_seg=args.w_seg, gamma=args.gamma,
                   pos_slice=dataset.pos_slice,
                   pos_mean=dataset.state_mean, pos_std=dataset.state_std,
                   rest_len_mm=ROPE_LENGTH_MM / (dataset.num_nodes - 1))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, loss_fn, dataset, optimizer=optimizer)
        va = run_epoch(model, val_loader, device, loss_fn, dataset, optimizer=None)
        rho = model.operator.spectral_radius()

        flag = ""
        if va["pred"] < best_val:
            best_val, flag = va["pred"], "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Logging to console and TensorBoard
        log(f"epoch {epoch:03d} | train {tr['loss']:.5f} (pred {tr['pred']:.5f} "
            f"recon {tr['recon']:.5f} lin {tr['lin']:.5f} seg {tr['seg']:.5f}) | val {va['loss']:.5f} "
            f"(pred {va['pred']:.5f} seg {va['seg']:.5f}) | val rmse {va['rmse_1_mm']:.3f}/{va['rmse_H_mm']:.3f} mm "
            f"| rho(A) {rho:.3f}{flag}")

        for split, m in (("train", tr), ("val", va)):
            for k in ("loss", "pred", "recon", "lin", "stab", "seg", "rmse_1_mm", "rmse_H_mm"):
                writer.add_scalar(f"{split}/{k}", m[k], epoch)
            # train vs val on one chart, so overfitting is visible at a glance
            writer.add_scalars("compare/pred", {split: m["pred"]}, epoch)
        writer.add_scalar("spectral_radius", rho, epoch)
        if args.horizon > 1:
            for h, v in enumerate(va["per_step"]):
                writer.add_scalar(f"val_per_horizon/h{h + 1}", v, epoch)
        for c, v in va["per_case"].items():
            writer.add_scalar(f"val_per_case/{dataset.case_names[c]}", v, epoch)

    log(f"\nBest val rollout MSE: {best_val:.6f}"
        + ("  (normalized units)" if dataset.normalize else ""))

    if best_state is not None:
        model.load_state_dict(best_state)

        # Save best pt file - koopman_enc.pt, koopman_dec.pt, koopman_op.pt, koopman_input_enc.pt
        ckpt_dir = results_dir / "checkpoints" / exp_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), ckpt_dir / "koopman_enc.pt")
        torch.save(model.decoder.state_dict(), ckpt_dir / "koopman_dec.pt")
        torch.save(model.operator.state_dict(), ckpt_dir / "koopman_op.pt")
        torch.save(model.input_encoder.state_dict(), ckpt_dir / "koopman_input_enc.pt")

    te = run_epoch(model, test_loader, device, loss_fn, dataset, optimizer=None)
    log(f"Test rollout MSE {te['pred']:.6f} | position RMSE "
        f"{te['rmse_1_mm']:.3f} mm @h=1, {te['rmse_H_mm']:.3f} mm @h={args.horizon}")
    log("Per-horizon test MSE: " + ", ".join(f"h{h+1}={v:.5f}" for h, v in enumerate(te["per_step"])))
    if len(dataset.case_names) > 1:
        log("Per-case test MSE:")
        for c, v in sorted(te["per_case"].items()):
            log(f"    {dataset.case_names[c]:<24} {v:.6f}")


    # Printing Koopman parameters for inspection
    log(f"Koopman operator A:\n{model.operator.A.detach().cpu().numpy()}")
    log(f"Koopman operator B:\n{model.operator.B.detach().cpu().numpy()}")
    # if model.operator.c is not None:
    #     log(f"Koopman operator c:\n{model.operator.c.detach().cpu().numpy()}")
        
    for k in ("loss", "pred", "recon", "lin", "seg", "rmse_1_mm", "rmse_H_mm"):
        writer.add_scalar(f"test/{k}", te[k], args.epochs)

    # hparams view: every run in <results-dir>/runs shows up as one row, so a
    # koopman-dim / horizon / feature sweep is directly comparable in TensorBoard.
    writer.add_hparams(
        {
            "koopman_dim": args.koopman_dim, "hidden_dim": args.hidden_dim,
            "num_hidden": args.num_hidden, "horizon": args.horizon, "lr": args.lr,
            "batch_size": args.batch_size, "w_recon": args.w_recon, "w_lin": args.w_lin,
            "w_stab": args.w_stab, "w_seg": args.w_seg, "gamma": args.gamma, "seed": args.seed,
            "features": "+".join(dataset.features), "state_dim": dataset.state_dim,
            "num_cases": len(dataset.case_names),
            "encoded_control_dim": args.encoded_control_dim,
            "control_hidden_dim": args.control_hidden_dim,
            "control_num_hidden": args.control_num_hidden,
        },
        {
            "hparam/val_pred": best_val, "hparam/test_pred": te["pred"],
            "hparam/test_rmse_1_mm": te["rmse_1_mm"],
            "hparam/test_rmse_H_mm": te["rmse_H_mm"],
            "hparam/test_lin": te["lin"], "hparam/test_seg": te["seg"], "hparam/rho_A": model.operator.spectral_radius(),
        },
    )
    results_file.close()
    writer.close()


if __name__ == "__main__":
    main()