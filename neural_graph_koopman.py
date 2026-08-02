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
  encoder   : MLP,    x_k          -> z_k                    (lifted / Koopman space)
  operator  : affine, z_k, u_k     -> z_k+1 = A z_k + B u_k + c
  decoder   : MLP,    z            -> x_hat                  (full state vector)

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
    "angvel": (("ang_vel_z_",), "node"),
    "curv":   (("curvature_",), "node"),
    "edge":   (("edge_dx_", "edge_dy_", "edge_length_"), "edge"),
}
DEFAULT_FEATURES = ("pos", "vel")


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
        # `layout` records where each group lives in the flat vector and whether it is
        # indexed by node or by edge, so a graph model can reshape the flat state back
        # into [B, N, C_node] / [B, E, C_edge] without re-parsing the CSV.
        blocks, self.feature_slices, self.layout, offset = [], {}, [], 0
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
            self.layout.append(dict(name=name, kind=kind, n=n, p=len(prefixes),
                                    start=offset, size=arr.shape[1]))
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

        # Position block, kept for reporting error in millimetres and for letting the
        # graph encoder recover true (un-normalised) geometry when it builds edge features.
        self.pos_slice = self.feature_slices["pos"]
        self.pos_mean = self.state_mean[0, self.pos_slice].reshape(self.num_nodes, 2).copy()
        self.pos_std = self.state_std[0, self.pos_slice].reshape(self.num_nodes, 2).copy()
        pos_mm = (self.states[:, self.pos_slice].numpy() * self.state_std[0, self.pos_slice]
                  + self.state_mean[0, self.pos_slice]).reshape(-1, self.num_nodes, 2)
        self.edge_len_scale = float(
            np.linalg.norm(np.diff(pos_mm, axis=1), axis=-1).mean()) + 1e-8

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
# class MLP(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         layers = []
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             if i < len(dims) - 2:
#                 layers.append(nn.ReLU())
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


# class KoopmanEncoder(nn.Module):
#     """Lifts the flat rope state into Koopman space: x -> z."""

#     def __init__(self, state_dim, koopman_dim, hidden_dim=256, num_hidden=2):
#         super().__init__()
#         self.mlp = MLP([state_dim] + [hidden_dim] * num_hidden + [koopman_dim])

#     def forward(self, x):
#         return self.mlp(x)


# class KoopmanDecoder(nn.Module):
#     """Projects Koopman space back to the full state vector: z -> x_hat."""

#     def __init__(self, koopman_dim, state_dim, hidden_dim=256, num_hidden=2):
#         super().__init__()
#         self.mlp = MLP([koopman_dim] + [hidden_dim] * num_hidden + [state_dim])

#     def forward(self, z):
#         return self.mlp(z)


# --------------------------------------------------------------------------- #
# Graph encoder / decoder
#
# The rope is a path graph: node i is connected only to i-1 and i+1. An MLP on the
# flattened state has to discover that locality from data -- every hidden unit sees all
# 30 nodes at once, so it spends capacity learning that node 3 and node 27 are unrelated,
# and that consecutive nodes sit a fixed distance apart. A message-passing network gets
# both for free: weights are shared across nodes (so one bending rule is learned once
# instead of 30 times) and edge features carry the segment vector and length directly,
# which is where inextensibility lives.
#
# No torch_geometric: on a path graph, message passing is just slicing neighbours along
# the node axis, which is faster and keeps the dependency list short.
# --------------------------------------------------------------------------- #
class StateLayout:
    """Reshapes between the flat state vector and per-node / per-edge feature tensors."""

    def __init__(self, layout, num_nodes, num_edges):
        self.layout = layout
        self.num_nodes, self.num_edges = num_nodes, num_edges
        self.state_dim = sum(g["size"] for g in layout)
        self.node_groups = [g for g in layout if g["kind"] == "node"]
        self.edge_groups = [g for g in layout if g["kind"] == "edge"]
        self.node_channels = sum(g["p"] for g in self.node_groups)
        self.edge_channels = sum(g["p"] for g in self.edge_groups)
        if not self.node_groups or self.node_groups[0]["name"] != "pos":
            raise ValueError("Graph model expects 'pos' as the first node feature group.")

    def split(self, x):
        """[B, D] -> ([B, N, C_node], [B, E, C_edge])."""
        B = x.size(0)
        node = [x[:, g["start"]:g["start"] + g["size"]].view(B, g["n"], g["p"])
                for g in self.node_groups]
        edge = [x[:, g["start"]:g["start"] + g["size"]].view(B, g["n"], g["p"])
                for g in self.edge_groups]
        node = torch.cat(node, -1)
        edge = torch.cat(edge, -1) if edge else x.new_zeros(B, self.num_edges, 0)
        return node, edge

    def merge(self, node, edge):
        """([B, N, C_node], [B, E, C_edge]) -> [B, D], inverse of split."""
        B = node.size(0)
        out, n_off, e_off = [], 0, 0
        for g in self.layout:
            if g["kind"] == "node":
                out.append(node[..., n_off:n_off + g["p"]].reshape(B, -1)); n_off += g["p"]
            else:
                out.append(edge[..., e_off:e_off + g["p"]].reshape(B, -1)); e_off += g["p"]
        return torch.cat(out, -1)


class ChainMessagePassing(nn.Module):
    """One round of message passing on a path graph, both directions.

        m_{i->j} = phi_e([h_i, h_j, e_ij])
        h_i'     = h_i + phi_n([h_i, sum_j m_{j->i}])

    Residual + LayerNorm because stacking several rounds otherwise washes out the
    node-local information the decoder needs.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = MLP([2 * node_dim + edge_dim, hidden_dim, node_dim])
        self.node_mlp = MLP([2 * node_dim, hidden_dim, node_dim])
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, e):
        left, right = h[:, :-1], h[:, 1:]                      # endpoints of edge i
        m_fwd = self.edge_mlp(torch.cat([left, right, e], -1))   # i   -> i+1
        m_bwd = self.edge_mlp(torch.cat([right, left, e], -1))   # i+1 -> i
        zero = torch.zeros_like(m_fwd[:, :1])
        agg = torch.cat([zero, m_fwd], 1) + torch.cat([m_bwd, zero], 1)
        return self.norm(h + self.node_mlp(torch.cat([h, agg], -1)))


class _GeometricEdges(nn.Module):
    """Builds (dx, dy, length) per edge from the position channels, in millimetres.

    Positions are per-column standardised, so differencing normalised coordinates would
    not give a true displacement (each column has its own std). The mean/std are stored
    as buffers and undone here, so segment length is the real physical quantity the
    inextensibility constraint acts on; it is then divided by the mean rest segment
    length to bring it back to order 1.
    """

    def __init__(self, pos_mean, pos_std, edge_len_scale):
        super().__init__()
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean))     # [N, 2]
        self.register_buffer("pos_std", torch.as_tensor(pos_std))
        self.edge_len_scale = edge_len_scale

    def forward(self, node_feats):
        pos = node_feats[..., :2] * self.pos_std + self.pos_mean        # [B, N, 2] mm
        d = pos[:, 1:] - pos[:, :-1]                                    # [B, E, 2]
        length = d.norm(dim=-1, keepdim=True)
        return torch.cat([d / self.edge_len_scale, length / self.edge_len_scale], -1)


class GNNEncoder(nn.Module):
    """Message passing over the chain -> per-node latents (+ a pooled global vector).

    The lifted state is the concatenation of all per-node latents plus the global vector,
    so z stays a plain flat vector and the Koopman operator is unchanged: still a single
    linear A acting on the whole rope. Pooling alone would not do -- a global summary
    cannot be decoded back to 30 distinct node states.
    """

    def __init__(self, layout: StateLayout, node_latent=8, global_latent=32,
                 hidden_dim=128, num_layers=4, geo=None):
        super().__init__()
        self.layout, self.geo = layout, geo
        self.num_nodes = layout.num_nodes
        self.node_latent, self.global_latent = node_latent, global_latent

        edge_in = layout.edge_channels + (3 if geo is not None else 0)
        self.node_embed = MLP([layout.node_channels, hidden_dim, hidden_dim])
        self.edge_embed = MLP([edge_in, hidden_dim, hidden_dim]) if edge_in else None
        self.layers = nn.ModuleList(
            ChainMessagePassing(hidden_dim, hidden_dim if self.edge_embed else 0, hidden_dim)
            for _ in range(num_layers))
        self.node_head = nn.Linear(hidden_dim, node_latent)
        self.global_head = MLP([2 * hidden_dim, hidden_dim, global_latent]) if global_latent else None

    @property
    def koopman_dim(self):
        return self.num_nodes * self.node_latent + self.global_latent

    def forward(self, x):
        node, edge = self.layout.split(x)
        if self.geo is not None:
            edge = torch.cat([edge, self.geo(node)], -1)
        h = self.node_embed(node)
        e = self.edge_embed(edge) if self.edge_embed else h.new_zeros(h.size(0), self.num_nodes - 1, 0)
        for layer in self.layers:
            h = layer(h, e)
        z = self.node_head(h).reshape(h.size(0), -1)
        if self.global_head is not None:
            pooled = torch.cat([h.mean(1), h.max(1).values], -1)
            z = torch.cat([z, self.global_head(pooled)], -1)
        return z


class GNNDecoder(nn.Module):
    """Per-node latents (+ global) -> message passing -> node and edge state channels.

    Edge features are regenerated from the endpoint embeddings rather than carried, since
    at decode time there is no edge input -- only the latent.
    """

    def __init__(self, layout: StateLayout, node_latent=8, global_latent=32,
                 hidden_dim=128, num_layers=4):
        super().__init__()
        self.layout = layout
        self.num_nodes, self.node_latent = layout.num_nodes, node_latent
        self.global_latent = global_latent

        self.lift = MLP([node_latent + global_latent, hidden_dim, hidden_dim])
        self.layers = nn.ModuleList(
            ChainMessagePassing(hidden_dim, 0, hidden_dim) for _ in range(num_layers))
        self.node_head = MLP([hidden_dim, hidden_dim, layout.node_channels])
        self.edge_head = (MLP([2 * hidden_dim, hidden_dim, layout.edge_channels])
                          if layout.edge_channels else None)

    def forward(self, z):
        B = z.size(0)
        n_part = z[:, :self.num_nodes * self.node_latent].view(B, self.num_nodes, self.node_latent)
        if self.global_latent:
            g = z[:, self.num_nodes * self.node_latent:].unsqueeze(1).expand(-1, self.num_nodes, -1)
            n_part = torch.cat([n_part, g], -1)
        h = self.lift(n_part)
        empty = h.new_zeros(B, self.num_nodes - 1, 0)
        for layer in self.layers:
            h = layer(h, empty)
        node_out = self.node_head(h)
        if self.edge_head is not None:
            edge_out = self.edge_head(torch.cat([h[:, :-1], h[:, 1:]], -1))
        else:
            edge_out = h.new_zeros(B, self.num_nodes - 1, 0)
        return self.layout.merge(node_out, edge_out)


class KoopmanOperator(nn.Module):
    """Affine latent dynamics with control: z_k+1 = A z_k + B u_k + c.

    A starts near identity: one sim step changes the state only slightly, so the
    untrained operator is roughly the identity map, which keeps early rollouts stable.
    The bias c absorbs command-independent drift (gravity, ongoing settling). Keeping the
    dynamics affine preserves the convex QP that motivates the Koopman approach.
    """

    def __init__(self, koopman_dim, control_dim, bias=True):
        super().__init__()
        self.A = nn.Parameter(torch.eye(koopman_dim) + 0.01 * torch.randn(koopman_dim, koopman_dim))
        self.B = nn.Parameter(0.01 * torch.randn(koopman_dim, control_dim))
        self.c = nn.Parameter(torch.zeros(koopman_dim)) if bias else None
        self.register_buffer("_u_iter", torch.randn(koopman_dim))

    def forward(self, z_k, u_k):
        z = z_k @ self.A.T + u_k @ self.B.T
        return z if self.c is None else z + self.c

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
    """Encoder + affine operator + decoder.

    The operator is identical either way -- swapping the MLP encoder/decoder for the
    graph ones changes only how x is lifted and projected, not the linear dynamics. That
    is what makes the two architectures directly comparable.
    """

    def __init__(self, encoder, decoder, koopman_dim, control_dim, operator_bias=True):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.operator = KoopmanOperator(koopman_dim, control_dim, bias=operator_bias)

    @classmethod
    def mlp(cls, state_dim, control_dim, koopman_dim, hidden_dim=256, num_hidden=2,
            operator_bias=True):
        return cls(KoopmanEncoder(state_dim, koopman_dim, hidden_dim, num_hidden),
                   KoopmanDecoder(koopman_dim, state_dim, hidden_dim, num_hidden),
                   koopman_dim, control_dim, operator_bias)

    @classmethod
    def gnn(cls, dataset, control_dim, node_latent=8, global_latent=32, hidden_dim=128,
            num_layers=4, operator_bias=True):
        layout = StateLayout(dataset.layout, dataset.num_nodes, dataset.num_edges)
        geo = _GeometricEdges(dataset.pos_mean, dataset.pos_std, dataset.edge_len_scale)
        enc = GNNEncoder(layout, node_latent, global_latent, hidden_dim, num_layers, geo)
        dec = GNNDecoder(layout, node_latent, global_latent, hidden_dim, num_layers)
        return cls(enc, dec, enc.koopman_dim, control_dim, operator_bias)

    def rollout(self, x0, us):
        """Encode once, then roll the latent forward under the recorded commands.

        Args:
            x0: [B, D]      initial state
            us: [B, H, 2]   commands for each transition
        Returns:
            x_hat: [B, H, D]  decoded predicted states
            z_hat: [B, H, Z]  latent trajectory
            z0:    [B, Z]     initial latent
        """
        z = z0 = self.encoder(x0)
        x_hat, z_hat = [], []
        for h in range(us.size(1)):
            z = self.operator(z, us[:, h])
            z_hat.append(z)
            x_hat.append(self.decoder(z))
        return torch.stack(x_hat, 1), torch.stack(z_hat, 1), z0


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class KoopmanLoss:
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

    def __init__(self, w_pred=1.0, w_recon=0.5, w_lin=0.1, w_stab=0.0,
                 gamma=1.0, stab_margin=1.0):
        self.w_pred, self.w_recon = w_pred, w_recon
        self.w_lin, self.w_stab = w_lin, w_stab
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

    # -- total --------------------------------------------------------------- #
    def __call__(self, model, x0, us, xs):
        x_hat, z_hat, _ = model.rollout(x0, us)
        pred, per_step = self.prediction(x_hat, xs)
        recon = self.reconstruction(model, x0, xs)
        lin = self.linearity(model, z_hat, xs)
        stab = self.stability(model)
        total = self.w_pred * pred + self.w_recon * recon + self.w_lin * lin + self.w_stab * stab
        parts = {"loss": total, "pred": pred, "recon": recon, "lin": lin, "stab": stab}
        return total, parts, x_hat, per_step


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, loss_fn, dataset, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    sums = {k: 0.0 for k in ("loss", "pred", "recon", "lin", "stab")}
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
    p.add_argument("--arch", choices=("mlp", "gnn"), default="mlp",
                   help="Encoder/decoder architecture. The Koopman operator is identical in both.")
    p.add_argument("--koopman-dim", type=int, default=64, help="[mlp] latent width.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden", type=int, default=2, help="[mlp] hidden layers.")
    p.add_argument("--node-latent", type=int, default=8,
                   help="[gnn] latent channels per node; koopman_dim = N*node_latent + global_latent.")
    p.add_argument("--global-latent", type=int, default=32,
                   help="[gnn] pooled global latent width (0 to disable).")
    p.add_argument("--gnn-layers", type=int, default=4,
                   help="[gnn] message-passing rounds. Information travels this many nodes along "
                        "the chain, so it bounds how far a disturbance can propagate per step.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--w-pred", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=0.5)
    p.add_argument("--w-lin", type=float, default=0.1)
    p.add_argument("--w-stab", type=float, default=0.0, help="Weight on the sigma_max(A) hinge.")
    p.add_argument("--gamma", type=float, default=1.0, help="Horizon discount for pred/lin.")
    p.add_argument("--clamp-tol-mm", type=float, default=1.0,
                   help="Report the fraction of rows where node 0 missed its target by more than this.")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-operator-bias", action="store_true")
    p.add_argument("--results-dir", type=str,
                   default="sofa_env/scenes/rope_manip_koopman/koopman_analysis")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"exp_{int(time())}"
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

    if args.arch == "gnn":
        model = KoopmanModel.gnn(
            dataset, control_dim=dataset.control_dim,
            node_latent=args.node_latent, global_latent=args.global_latent,
            hidden_dim=args.hidden_dim, num_layers=args.gnn_layers,
            operator_bias=not args.no_operator_bias,
        ).to(device)
        koopman_dim = model.encoder.koopman_dim
        log(f"Arch: gnn | {args.gnn_layers} message-passing rounds, "
            f"node_latent {args.node_latent}, global_latent {args.global_latent} "
            f"-> koopman_dim {koopman_dim}")
    else:
        model = KoopmanModel.mlp(
            state_dim=dataset.state_dim, control_dim=dataset.control_dim,
            koopman_dim=args.koopman_dim, hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden, operator_bias=not args.no_operator_bias,
        ).to(device)
        koopman_dim = args.koopman_dim
        log(f"Arch: mlp | koopman_dim {koopman_dim}")

    n_enc = sum(q.numel() for q in model.encoder.parameters())
    n_dec = sum(q.numel() for q in model.decoder.parameters())
    n_op = sum(q.numel() for q in model.operator.parameters())
    log(f"Parameters: {n_enc + n_dec + n_op:,} "
        f"(encoder {n_enc:,} | operator {n_op:,} | decoder {n_dec:,})")

    loss_fn = KoopmanLoss(w_pred=args.w_pred, w_recon=args.w_recon, w_lin=args.w_lin,
                          w_stab=args.w_stab, gamma=args.gamma)
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
            if args.save:
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, args.save)

        log(f"epoch {epoch:03d} | train {tr['loss']:.5f} (pred {tr['pred']:.5f} "
            f"recon {tr['recon']:.5f} lin {tr['lin']:.5f}) | val {va['loss']:.5f} "
            f"(pred {va['pred']:.5f}) | val rmse {va['rmse_1_mm']:.3f}/{va['rmse_H_mm']:.3f} mm "
            f"| rho(A) {rho:.3f}{flag}")

        for split, m in (("train", tr), ("val", va)):
            for k in ("loss", "pred", "recon", "lin", "stab", "rmse_1_mm", "rmse_H_mm"):
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
    te = run_epoch(model, test_loader, device, loss_fn, dataset, optimizer=None)
    log(f"Test rollout MSE {te['pred']:.6f} | position RMSE "
        f"{te['rmse_1_mm']:.3f} mm @h=1, {te['rmse_H_mm']:.3f} mm @h={args.horizon}")
    log("Per-horizon test MSE: " + ", ".join(f"h{h+1}={v:.5f}" for h, v in enumerate(te["per_step"])))
    if len(dataset.case_names) > 1:
        log("Per-case test MSE:")
        for c, v in sorted(te["per_case"].items()):
            log(f"    {dataset.case_names[c]:<24} {v:.6f}")

    for k in ("loss", "pred", "recon", "lin", "rmse_1_mm", "rmse_H_mm"):
        writer.add_scalar(f"test/{k}", te[k], args.epochs)

    # hparams view: every run in <results-dir>/runs shows up as one row, so a
    # koopman-dim / horizon / feature sweep is directly comparable in TensorBoard.
    writer.add_hparams(
        {
            "arch": args.arch, "koopman_dim": koopman_dim, "hidden_dim": args.hidden_dim,
            "num_hidden": args.num_hidden, "gnn_layers": args.gnn_layers,
            "node_latent": args.node_latent, "horizon": args.horizon, "lr": args.lr,
            "batch_size": args.batch_size, "w_recon": args.w_recon, "w_lin": args.w_lin,
            "w_stab": args.w_stab, "gamma": args.gamma, "seed": args.seed,
            "features": "+".join(dataset.features), "state_dim": dataset.state_dim,
            "num_cases": len(dataset.case_names),
        },
        {
            "hparam/val_pred": best_val, "hparam/test_pred": te["pred"],
            "hparam/test_rmse_1_mm": te["rmse_1_mm"],
            "hparam/test_rmse_H_mm": te["rmse_H_mm"],
            "hparam/test_lin": te["lin"], "hparam/rho_A": model.operator.spectral_radius(),
        },
    )
    results_file.close()
    writer.close()


if __name__ == "__main__":
    main()