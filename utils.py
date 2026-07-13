"""
utils.py

Utilities for DFM training/inference for discrete cell type dynamics.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(path: str | os.PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() in [".json"]:
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError("PyYAML not installed; use JSON config or install pyyaml.") from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config file extension: {path.suffix}")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_rdata_dataframe(rdata_path: str | os.PathLike, key: str = "Data"):
    """Read an .RData file and return the specified object as a pandas DataFrame.

    Requires pyreadr.
    """
    import pyreadr  # local import to avoid hard dependency during static checks

    r = pyreadr.read_r(str(rdata_path))
    if key not in r:
        raise KeyError(f"'{rdata_path}' missing variable '{key}'. Available: {list(r.keys())}")
    df = r[key]
    return df


@dataclass(frozen=True)
class TimeEncoding:
    """Normalization parameters for real time conditioning."""
    t_min: float
    t_max: float
    dt_min: float
    dt_max: float

    def norm_t(self, t: np.ndarray) -> np.ndarray:
        # Map to [0,1]
        denom = (self.t_max - self.t_min) if (self.t_max - self.t_min) > 0 else 1.0
        return (t - self.t_min) / denom

    def norm_dt(self, dt: np.ndarray) -> np.ndarray:
        denom = (self.dt_max - self.dt_min) if (self.dt_max - self.dt_min) > 0 else 1.0
        return (dt - self.dt_min) / denom



def build_type_name_mapping(df, label_col: str = "celltype", name_col: str = "typeName") -> dict:
    """Build a mapping from raw celltype label -> human-readable typeName.

    If multiple names occur for the same raw label, choose the most frequent.
    """
    import pandas as pd  # type: ignore

    if label_col not in df.columns or name_col not in df.columns:
        return {}

    tmp = df[[label_col, name_col]].copy()
    # Most frequent name per label
    mapping = (
        tmp.groupby(label_col)[name_col]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {int(k): str(v) for k, v in mapping.items()}

def build_label_mapping(df, label_col: str = "celltype") -> Tuple[Dict[int, int], Dict[int, int]]:
    """Map raw labels to contiguous [0..K-1].

    Returns:
        raw_to_idx: dict mapping raw label -> idx
        idx_to_raw: dict mapping idx -> raw label
    """
    raw = np.asarray(df[label_col]).astype(int)
    uniq = np.unique(raw)
    raw_to_idx = {int(v): i for i, v in enumerate(uniq)}
    idx_to_raw = {i: int(v) for i, v in enumerate(uniq)}
    return raw_to_idx, idx_to_raw


def build_pools_by_time(
    df,
    time_col: str = "timepoint",
    label_col: str = "celltype",
    raw_to_idx: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Build per-timepoint pools of labels (as contiguous indices).

    Returns:
        timepoints_sorted: array shape [N]
        pools: list length N, each is array of label indices shape [M_n]
    """
    times = np.asarray(df[time_col]).astype(float)
    timepoints_sorted = np.sort(np.unique(times))

    if raw_to_idx is None:
        raw_to_idx, _ = build_label_mapping(df, label_col=label_col)

    raw_labels = np.asarray(df[label_col]).astype(int)
    idx_labels = np.vectorize(raw_to_idx.get)(raw_labels).astype(int)

    pools: List[np.ndarray] = []
    for t in timepoints_sorted:
        mask = (times == t)
        pools.append(idx_labels[mask])
    return timepoints_sorted, pools


def compute_time_encoding(timepoints: np.ndarray) -> TimeEncoding:
    t_min, t_max = float(np.min(timepoints)), float(np.max(timepoints))
    dts = np.diff(timepoints)
    dt_min, dt_max = (float(np.min(dts)), float(np.max(dts))) if len(dts) > 0 else (1.0, 1.0)
    return TimeEncoding(t_min=t_min, t_max=t_max, dt_min=dt_min, dt_max=dt_max)


def empirical_distribution(labels: np.ndarray, num_states: int, smoothing: float = 0.0) -> np.ndarray:
    """Compute empirical distribution over K states from a label array."""
    counts = np.bincount(labels.astype(int), minlength=num_states).astype(np.float64)
    if smoothing > 0:
        counts = counts + smoothing
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.ones(num_states) / num_states
    return probs


def sample_from_pool(pool: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(low=0, high=len(pool), size=batch_size, endpoint=False)
    return pool[idx]


def sample_interval_indices(num_intervals: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=num_intervals, size=batch_size, endpoint=False)


def kappa_linear(s: Tensor) -> Tensor:
    return s


def kappa_linear_dot(s: Tensor) -> Tensor:
    return torch.ones_like(s)


def dfm_one_step_transition_probs(
    p1: Tensor,          # [B, K] denoiser probs at current state
    z: Tensor,           # [B] current state indices
    s: float,
    h: float,
    kappa_fn=kappa_linear,
    kappa_dot_fn=kappa_linear_dot,
) -> Tensor:
    """Compute transition probs for one Euler step in replacement-path DFM.

    With replacement path:
      u = (kappa_dot/(1-kappa)) * (p1 - delta_z)
      T = delta_z + h*u = (1-alpha)*delta_z + alpha*p1
    where alpha = h * kappa_dot(s) / (1 - kappa(s)).

    Args:
        p1: [B,K] denoiser distribution over terminal x1 given current state z and time s.
        z: [B] current state.
        s: scalar float (current normalized time).
        h: scalar float (step size).
    Returns:
        probs: [B, K] transition probability for next state.
    """
    # alpha computed at scalar s
    s_t = torch.tensor([s], device=p1.device, dtype=p1.dtype).repeat(p1.shape[0])
    k = kappa_fn(s_t)
    kd = kappa_dot_fn(s_t)
    alpha = (h * kd) / torch.clamp(1.0 - k, min=1e-6)  # [B]
    # enforce alpha in [0,1] for numerical stability
    alpha = torch.clamp(alpha, min=0.0, max=1.0)

    probs = p1 * alpha[:, None]  # [B,K]
    probs.scatter_add_(1, z[:, None], (1.0 - alpha)[:, None])
    # probs should sum to 1, but re-normalize for safety
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(np.abs(p - q)))


@torch.no_grad()
def propagate_distribution_deterministic(
    model,
    p0: np.ndarray,
    t_start_norm: float,
    dt_norm: float,
    num_steps: int,
    device: torch.device,
    s_end: float = 1.0,
) -> np.ndarray:
    """Deterministically propagate a K-dim distribution using the DFM transition kernel.

    This avoids Monte Carlo variance by enumerating current states z=0..K-1 each step.

    Args:
        model: denoiser model; forward(x_s, s, t_start, dt) -> logits [B,K]
            (the model may ignore some conditioning inputs depending on its config).
        p0: numpy array [K] start distribution.
        t_start_norm: normalized start time scalar in [0,1].
        dt_norm: normalized delta time scalar in [0,1].
        num_steps: number of Euler steps to push s:0->s_end, with variable last step.
        s_end: final normalized progress s in [0,1]. Default 1.0 (full interval).
    """
    K = int(p0.shape[0])
    p = p0.astype(np.float64).copy()
    p = p / p.sum()

    # build state tensor for all z
    z_all = torch.arange(K, device=device, dtype=torch.long)  # [K]

    s_end = float(np.clip(s_end, 0.0, 1.0))
    if s_end <= 0.0 + 1e-12:
        return p

    s = 0.0
    h_max = s_end / max(num_steps, 1)
    for _ in range(num_steps):
        if s >= s_end - 1e-9:
            break
        h = min(h_max, s_end - s)
        s_batch = torch.full((K,), float(s), device=device, dtype=torch.float32)
        t0_batch = torch.full((K,), float(t_start_norm), device=device, dtype=torch.float32)
        dt_batch = torch.full((K,), float(dt_norm), device=device, dtype=torch.float32)

        logits = model(z_all, s_batch, t0_batch, dt_batch)  # [K,K]
        p1 = torch.softmax(logits, dim=-1)  # [K,K], row corresponds to current state z

        # Transition matrix T[z, x] = (1-alpha)*I + alpha*p1[z,x]
        # alpha = h/(1-s) for kappa(s)=s
        alpha = h / max(1e-6, (1.0 - s))
        alpha = min(max(alpha, 0.0), 1.0)

        T = (alpha * p1).cpu().numpy().astype(np.float64)  # [K,K]
        for z in range(K):
            T[z, z] += (1.0 - alpha)

        # p_next[x] = sum_z p[z] * T[z,x]
        p = p @ T
        p = np.clip(p, 0.0, None)
        p = p / p.sum()

        s += h

    return p


@torch.no_grad()
def collect_transition_matrices(
    model,
    t_start_norm: float,
    dt_norm: float,
    num_steps: int,
    device: torch.device,
    s_end: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Collect per-step transition matrices T for a single interval.

    Returns:
        T_list: list of length <= num_steps, each [K, K] numpy array.
        p0: uniform start distribution used internally (not propagated).
    """
    K = int(model.cfg.num_states) if hasattr(model, "cfg") else None
    if K is None:
        raise AttributeError("Model is expected to have attribute cfg.num_states for transition export.")

    z_all = torch.arange(K, device=device, dtype=torch.long)  # [K]
    s_end = float(np.clip(s_end, 0.0, 1.0))
    if s_end <= 0.0 + 1e-12:
        return [], np.ones((K,), dtype=np.float64) / max(K, 1)

    s = 0.0
    h_max = s_end / max(num_steps, 1)
    T_list: List[np.ndarray] = []

    while s < s_end - 1e-9:
        h = min(h_max, s_end - s)
        s_batch = torch.full((K,), float(s), device=device, dtype=torch.float32)
        t0_batch = torch.full((K,), float(t_start_norm), device=device, dtype=torch.float32)
        dt_batch = torch.full((K,), float(dt_norm), device=device, dtype=torch.float32)

        logits = model(z_all, s_batch, t0_batch, dt_batch)  # [K, K]
        p1 = torch.softmax(logits, dim=-1)  # [K, K]

        alpha = h / max(1e-6, (1.0 - s))
        alpha = min(max(alpha, 0.0), 1.0)

        T = (alpha * p1).cpu().numpy().astype(np.float64)
        for z in range(K):
            T[z, z] += (1.0 - alpha)

        T = np.clip(T, 0.0, None)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        T = T / row_sums

        T_list.append(T)
        s += h

    return T_list, np.ones((K,), dtype=np.float64) / max(K, 1)


@torch.no_grad()
def predict_trajectory_deterministic(
    model,
    pools: List[np.ndarray],
    timepoints: np.ndarray,
    time_enc: TimeEncoding,
    num_states: int,
    num_steps: int,
    device: torch.device,
    smoothing: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the full time trajectory deterministically across all timepoints.

    We start from the empirical distribution at the first observed timepoint and then
    propagate interval-by-interval using the model-conditioned DFM kernel.

    Returns:
        p_true: array [N, K] true distributions from data
        p_pred: array [N, K] predicted distributions (p_pred[0] == p_true[0])
    """
    N = len(timepoints)
    p_true = np.zeros((N, num_states), dtype=np.float64)
    for i in range(N):
        p_true[i] = empirical_distribution(pools[i], num_states=num_states, smoothing=smoothing)

    p_pred = np.zeros_like(p_true)
    p_pred[0] = p_true[0].copy()

    for n in range(N - 1):
        t0_norm = float(time_enc.norm_t(np.array([timepoints[n]]))[0])
        dt_norm = float(time_enc.norm_dt(np.array([timepoints[n + 1] - timepoints[n]]))[0])

        p_pred[n + 1] = propagate_distribution_deterministic(
            model=model,
            p0=p_pred[n],
            t_start_norm=t0_norm,
            dt_norm=dt_norm,
            num_steps=num_steps,
            device=device,
        )

    return p_true, p_pred

# ----------------------------
# Coupling utilities: global tSNE centroid cost + entropic OT + U-mix
# ----------------------------

def _infer_embedding_columns(
    df: pd.DataFrame,
    prefix: str = "embedding",
) -> List[str]:
    """Infer embedding feature columns from a dataframe.

    By convention, embedding columns are named like:
        embedding1, embedding2, ..., embeddingD
    (an optional underscore is also accepted: embedding_1, ...).

    Returns the columns sorted by their numeric suffix. If no such columns
    exist, returns an empty list.
    """
    pat = re.compile(rf"^{re.escape(prefix)}_?(\d+)$")
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            cols.append((int(m.group(1)), str(c)))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def compute_global_centroids(
    df,
    raw_to_idx: Dict[int, int],
    tsne_cols: Optional[Sequence[str]] = ("tSNE_1", "tSNE_2"),
    label_col: str = "celltype",
    stat: str = "median",
    embedding_prefix: str = "embedding",
) -> np.ndarray:
    """Compute global (cross-time) per-class centroids in an embedding space.

    The centroid space is determined as follows:
      1) If columns matching `{embedding_prefix}{i}` (e.g., embedding1..embeddingD)
         are present, they are used (D is inferred from the dataframe).
      2) Otherwise, fall back to `tsne_cols` (default: (tSNE_1, tSNE_2)) for
         backward compatibility.

    Args:
        df: pandas DataFrame containing per-cell rows.
        raw_to_idx: Mapping from raw labels to contiguous indices 0..K-1.
        tsne_cols: Optional legacy coordinate columns used if embedding columns
            are absent.
        label_col: Column name for cell type labels.
        stat: 'median' (default) or 'mean' aggregation for centroids.
        embedding_prefix: Prefix for embedding columns (default: 'embedding').

    Returns:
        centroids: np.ndarray of shape [K, D], where D is the number of inferred
        embedding dimensions (or len(tsne_cols) if falling back).
    """
    if label_col not in df.columns:
        raise KeyError(f"DataFrame must contain label column '{label_col}'")

    # Prefer variable-dimensional embedding columns if present.
    embed_cols = _infer_embedding_columns(df, prefix=embedding_prefix)
    if len(embed_cols) > 0:
        feature_cols = embed_cols
    else:
        if tsne_cols is None:
            raise KeyError(
                f"No '{embedding_prefix}{{i}}' columns found and tsne_cols=None. "
                "Provide embedding columns or legacy coordinate columns."
            )
        missing = [c for c in tsne_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing embedding columns '{embedding_prefix}{{i}}' and legacy columns {missing}. "
                f"Available columns (sample): {list(df.columns)[:20]}"
            )
        feature_cols = list(tsne_cols)

    tmp = df[[label_col, *feature_cols]].copy()

    # pyreadr may return embedding columns as object/strings; force numeric so aggregations work
    for col in feature_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp["__idx__"] = tmp[label_col].map(raw_to_idx)
    tmp = tmp.dropna(subset=["__idx__"])
    tmp["__idx__"] = tmp["__idx__"].astype(int)

    if stat == "mean":
        agg = tmp.groupby("__idx__")[feature_cols].mean()
    else:
        agg = tmp.groupby("__idx__")[feature_cols].median()

    K = len(raw_to_idx)
    D = len(feature_cols)
    centroids = np.zeros((K, D), dtype=np.float64)
    for idx in range(K):
        if idx in agg.index:
            centroids[idx, :] = agg.loc[idx].to_numpy(dtype=np.float64)
        else:
            centroids[idx, :] = 0.0
    return centroids
def build_global_cost_matrix(
    centroids: np.ndarray,
    power: int = 2,
    normalize: str | None = "median_offdiag",
) -> np.ndarray:
    """Build a [K, K] cost matrix from per-class centroids.

    Args:
        centroids: Array of shape [K, D] (D can be any positive integer).
        power: If 1, use Euclidean distance; otherwise use squared Euclidean distance.
        normalize: If 'median_offdiag', divide by the median of positive off-diagonal
            entries to stabilize scale. If None, no normalization is applied.

    Returns:
        C: Cost matrix of shape [K, K], dtype float64.
    """
    K = int(centroids.shape[0])
    diffs = centroids[:, None, :] - centroids[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    C = np.sqrt(np.maximum(d2, 0.0)) if power == 1 else d2
    np.fill_diagonal(C, 0.0)
    if normalize == "median_offdiag":
        off = C[~np.eye(K, dtype=bool)]
        med = np.median(off[off > 0]) if np.any(off > 0) else 1.0
        if med > 0:
            C = C / med
    return C.astype(np.float64)
def sinkhorn_ot_coupling(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float = 0.1,
    n_iters: int = 200,
    support_only: bool = False,
    eps_denom: float = 1e-12,
) -> np.ndarray:
    """标准平衡熵正则 OT（Sinkhorn）。support_only=True 时只在 a>0,b>0 的子集上解。"""
    K = int(C.shape[0])
    if support_only:
        S0 = np.where(a > 0)[0]
        S1 = np.where(b > 0)[0]
        if S0.size == 0 or S1.size == 0:
            return np.outer(a, b)
        a_s, b_s = a[S0], b[S1]
        C_s = C[np.ix_(S0, S1)]
        Kmat = np.exp(-C_s / max(eps, 1e-12))
        u = np.ones_like(a_s, dtype=np.float64) / max(a_s.size, 1)
        v = np.ones_like(b_s, dtype=np.float64) / max(b_s.size, 1)
        for _ in range(max(n_iters, 1)):
            Kv = Kmat @ v + eps_denom;  u = a_s / Kv
            KTu = Kmat.T @ u + eps_denom;  v = b_s / KTu
        pi_s = (u[:, None] * Kmat) * v[None, :]
        total = pi_s.sum()
        if total > 0: pi_s = pi_s / total
        pi = np.zeros((K, K), dtype=np.float64)
        pi[np.ix_(S0, S1)] = pi_s
        return pi
    else:
        Kmat = np.exp(-C / max(eps, 1e-12))
        u = np.ones_like(a, dtype=np.float64) / max(a.size, 1)
        v = np.ones_like(b, dtype=np.float64) / max(b.size, 1)
        for _ in range(max(n_iters, 1)):
            Kv = Kmat @ v + eps_denom;  u = a / Kv
            KTu = Kmat.T @ u + eps_denom; v = b / KTu
        pi = (u[:, None] * Kmat) * v[None, :]
        total = pi.sum()
        if total > 0: pi = pi / total
        return pi


def mix_coupling(pi_ot: np.ndarray, a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    pi = (1.0 - lam) * pi_ot + lam * np.outer(a, b)
    s = pi.sum()
    if s > 0: pi = pi / s
    return pi


def precompute_ot_mix_for_intervals(
    df,
    raw_to_idx: Dict[int, int],
    pools: List[np.ndarray],
    timepoints: np.ndarray,
    tsne_cols: Optional[Sequence[str]] = ("tSNE_1", "tSNE_2"),
    centroid_stat: str = "median",
    cost_power: int = 2,
    cost_normalize: str | None = "median_offdiag",
    ot_eps: float = 0.1,
    sinkhorn_iters: int = 200,
    lam_mix: float = 0.1,
    support_only: bool = False,
    smoothing_for_marginals: float = 0.0,
    # Cost feature configuration
    label_col: str = "celltype",
    embedding_prefix: str = "embedding",
) -> List[np.ndarray]:
    """Precompute mixed couplings π_mix for each adjacent interval n→n+1.

    The OT cost is computed once globally from per-class centroids in the
    embedding space (see `compute_global_centroids`), and reused across all
    intervals. Interval-specific marginals (a, b) are derived from the pools.
    """
    K = len(raw_to_idx)
    centroids = compute_global_centroids(
        df=df,
        raw_to_idx=raw_to_idx,
        tsne_cols=tsne_cols,
        label_col=label_col,
        stat=centroid_stat,
        embedding_prefix=embedding_prefix,
    )
    C = build_global_cost_matrix(centroids, power=cost_power, normalize=cost_normalize)

    num_intervals = len(timepoints) - 1
    pi_mix_list: List[np.ndarray] = []
    for n in range(num_intervals):
        a = empirical_distribution(pools[n],     num_states=K, smoothing=smoothing_for_marginals); a = a / a.sum()
        b = empirical_distribution(pools[n + 1], num_states=K, smoothing=smoothing_for_marginals); b = b / b.sum()
        pi_ot  = sinkhorn_ot_coupling(a=a, b=b, C=C, eps=ot_eps, n_iters=sinkhorn_iters, support_only=support_only)
        pi_mix = mix_coupling(pi_ot, a, b, lam=lam_mix)
        pi_mix_list.append(pi_mix.astype(np.float64))
    return pi_mix_list
def sample_pairs_from_coupling(pi: np.ndarray, rng: np.random.Generator, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """从 π 采样 (i,j) 对。"""
    K = pi.shape[0]
    flat = np.maximum(pi.reshape(-1).astype(np.float64), 0.0)
    s = flat.sum()
    flat = np.ones(K*K, dtype=np.float64)/(K*K) if (not np.isfinite(s) or s <= 0) else flat/s
    idx = rng.choice(flat.size, size=batch_size, replace=True, p=flat)
    i = (idx // K).astype(np.int64); j = (idx % K).astype(np.int64)
    return i, j
