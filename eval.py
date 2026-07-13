"""eval.py

Held-out evaluation utilities for the discrete Flow Matching (DFM) single-cell
cell-type proportion dynamics project.

This module implements the *post-training* held-out evaluation protocol:

* Training is performed on a masked timepoint (held-out) using only training
  timepoints/intervals.
* Model selection (best checkpoint) uses metrics on training intervals only.
* After training, we evaluate the held-out timepoint using a Sinkhorn-approximated
  Wasserstein-1 (W1) distance between the predicted and true distributions.

Important: This file **must not** change the training objective, optimizer, or
model architecture. It is evaluation-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils import (
    TimeEncoding,
    build_global_cost_matrix,
    compute_global_centroids,
    empirical_distribution,
    js_divergence,
    kl_divergence,
    l1_distance,
    propagate_distribution_deterministic,
    sinkhorn_ot_coupling,
)
from generate_sim import (
    default_transitions_for_K,
    build_transition_matrix,
    build_module_plan,
)


@dataclass(frozen=True)
class HeldOutEvalConfig:
    """Configuration for held-out evaluation."""

    # Sinkhorn settings for W1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 300
    support_only: bool = False

    # Numerical stability for distributions (strategy A)
    smoothing_eps: float = 1e-8

    # Integration steps
    interp_steps: int = 50
    extrap_steps: int = 25

    # Cost matrix cache
    c_full_cache_path: Optional[Path] = None


def _safe_normalize(p: np.ndarray, eps: float = 0.0) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if eps > 0:
        p = p + float(eps)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(p, dtype=np.float64) / max(p.size, 1)
    return p / s


def compute_or_load_c_full(
    df_full,
    raw_to_idx: Dict[int, int],
    tsne_cols: Optional[Sequence[str]],
    centroid_stat: str,
    cost_power: int = 1,
    cost_normalize: Optional[str] = None,
    cache_path: Optional[Path] = None,
    # Compatibility parameters (unused for cost computation, except label_col)
    time_col: Optional[str] = None,
    label_col: str = "celltype",
    embedding_prefix: str = "embedding",
    **_unused,
) -> np.ndarray:
    """Compute (or load cached) full-data cost matrix C_full.

    C_full is used *only* for held-out evaluation W1 and is computed on all
    timepoints (including held-out). Embedding columns are inferred from the
    dataframe if present (see `utils.compute_global_centroids`); otherwise
    legacy `tsne_cols` can be used as a fallback.

    If cache_path is provided and exists, it is loaded directly; otherwise it
    is computed and saved to cache_path (if provided).
    """

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            C = np.load(str(cache_path))
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                raise ValueError(f"Cached C_full has invalid shape: {C.shape}")
            return C.astype(np.float64)

    # Note: `time_col` is accepted for compatibility but is not required for
    # centroid computation; centroids are computed globally across all rows.
    centroids = compute_global_centroids(
        df=df_full,
        raw_to_idx=raw_to_idx,
        tsne_cols=tsne_cols,
        label_col=label_col,
        stat=centroid_stat,
        embedding_prefix=embedding_prefix,
    )
    C = build_global_cost_matrix(centroids, power=cost_power, normalize=cost_normalize)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), C.astype(np.float64))

    return C.astype(np.float64)
def sinkhorn_w1(
    p: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    eps: float,
    iters: int,
    support_only: bool,
    smoothing_eps: float,
) -> float:
    """Approximate Wasserstein-1 using entropic Sinkhorn OT.

    We compute a coupling pi via Sinkhorn, then return <pi, C>.
    Uses "strategy A" for stability: if numerical issues arise, apply a small
    additive smoothing to both distributions and retry.
    """

    a = _safe_normalize(p, eps=0.0)
    b = _safe_normalize(q, eps=0.0)

    def _try(a0: np.ndarray, b0: np.ndarray) -> Optional[float]:
        pi = sinkhorn_ot_coupling(a=a0, b=b0, C=C, eps=eps, n_iters=iters, support_only=support_only)
        if not np.all(np.isfinite(pi)):
            return None
        cost = float(np.sum(pi * C))
        return cost if np.isfinite(cost) else None

    cost = _try(a, b)
    if cost is not None:
        return cost

    # Retry with small smoothing
    a_s = _safe_normalize(a, eps=smoothing_eps)
    b_s = _safe_normalize(b, eps=smoothing_eps)
    cost = _try(a_s, b_s)
    if cost is None:
        raise RuntimeError("Sinkhorn W1 failed (numerical instability) even after smoothing.")
    return cost


def evaluate_held_out(
    *,
    model: torch.nn.Module,
    held_out_k: int,
    timepoints_full: np.ndarray,
    pools_full: List[np.ndarray],
    timepoints_train: np.ndarray,
    pools_train: List[np.ndarray],
    time_enc_train: TimeEncoding,
    num_states: int,
    device: torch.device,
    C_full: np.ndarray,
    cfg: HeldOutEvalConfig,
    data_smoothing: float = 1e-6,
) -> Dict[str, float]:
    """Evaluate held-out timepoint using W1 on distributions.

    held_out_k is 1-indexed in the *full* timepoints ordering. We only support
    k in {2, ..., M}.
    """

    M = int(len(timepoints_full))
    if held_out_k < 2 or held_out_k > M:
        raise ValueError(f"held_out_k must be in [2, {M}], got {held_out_k}.")

    # True distribution at held-out timepoint (from full data)
    k0 = held_out_k - 1
    p_true = empirical_distribution(pools_full[k0], num_states=num_states, smoothing=data_smoothing)

    dt_used: float
    dt_norm_used: float

    if held_out_k == M:
        # Extrapolation: from last training timepoint to held-out last timepoint
        t_start = float(timepoints_train[-1])
        t_end = float(timepoints_full[-1])
        dt = float(t_end - t_start)
        dt_used = dt

        t0_norm = float(time_enc_train.norm_t(np.array([t_start], dtype=np.float64))[0])
        dt_norm = float(time_enc_train.norm_dt(np.array([dt], dtype=np.float64))[0])
        dt_norm_used = dt_norm

        # Start distribution from last training timepoint
        # timepoints_train is full without held-out; when k=M, last training is t_{M-1}
        p0 = empirical_distribution(pools_train[-1], num_states=num_states, smoothing=data_smoothing)

        p_pred = propagate_distribution_deterministic(
            model=model,
            p0=p0,
            t_start_norm=t0_norm,
            dt_norm=dt_norm,
            num_steps=int(cfg.extrap_steps),
            device=device,
            s_end=1.0,
        )

        s_star = 1.0
        mode = "extrapolation"

    else:
        # Interpolation inside interval [t_{k-1}, t_{k+1}] using the trained bridge
        t_left = float(timepoints_full[k0 - 1])
        t_mid = float(timepoints_full[k0])
        t_right = float(timepoints_full[k0 + 1])
        dt_total = float(t_right - t_left)
        if dt_total <= 0:
            raise ValueError(f"Non-positive dt_total for interpolation: {dt_total}")
        dt_used = dt_total
        s_star = float((t_mid - t_left) / dt_total)
        s_star = float(np.clip(s_star, 0.0, 1.0))

        # Start distribution from t_left (which is present in training)
        # In training timepoints, t_left exists and is some index; find its pool.
        # Since timepoints_train is sorted subset of timepoints_full, we can locate by value.
        idx_left = int(np.where(timepoints_train == t_left)[0][0])
        p0 = empirical_distribution(pools_train[idx_left], num_states=num_states, smoothing=data_smoothing)

        t0_norm = float(time_enc_train.norm_t(np.array([t_left], dtype=np.float64))[0])
        dt_norm = float(time_enc_train.norm_dt(np.array([dt_total], dtype=np.float64))[0])
        dt_norm_used = dt_norm

        p_pred = propagate_distribution_deterministic(
            model=model,
            p0=p0,
            t_start_norm=t0_norm,
            dt_norm=dt_norm,
            num_steps=int(cfg.interp_steps),
            device=device,
            s_end=s_star,
        )

        mode = "interpolation"

    w1 = sinkhorn_w1(
        p=p_true,
        q=p_pred,
        C=C_full,
        eps=float(cfg.sinkhorn_eps),
        iters=int(cfg.sinkhorn_iters),
        support_only=bool(cfg.support_only),
        smoothing_eps=float(cfg.smoothing_eps),
    )

    dt_ood = bool((dt_used < time_enc_train.dt_min - 1e-12) or (dt_used > time_enc_train.dt_max + 1e-12))

    return {
        "held_out_k": int(held_out_k),
        "mode": str(mode),
        "s_star": float(s_star),
        "dt_used": float(dt_used),
        "dt_norm_used": float(dt_norm_used),
        "dt_train_min": float(time_enc_train.dt_min),
        "dt_train_max": float(time_enc_train.dt_max),
        "dt_is_ood": bool(dt_ood),
        "w1_sinkhorn": float(w1),
        "js": float(js_divergence(p_true, p_pred)),
        "kl": float(kl_divergence(p_true, p_pred)),
        "l1": float(l1_distance(p_true, p_pred)),
    }


def write_eval_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------
# Transition alignment utilities (sim data)
# ----------------------------


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 0 and nv <= 0:
        return 1.0
    if nu <= 0 or nv <= 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _load_transition_matrix_with_labels(path: Path) -> tuple[np.ndarray, List[str]]:
    import pandas as pd  # local import

    df = pd.read_csv(path, index_col=0)
    labels = [str(x) for x in df.index.tolist()]
    T = df.to_numpy(dtype=np.float64)
    return T, labels


def generate_truth_transition_matrix_from_default(K: int) -> tuple[np.ndarray, any]:
    """Build ground-truth transition matrix from default module transitions for a given K."""
    plan = build_module_plan(K)
    transitions = default_transitions_for_K(K)
    T_truth = build_transition_matrix(
        K=K,
        transitions=transitions,
        stable_types=plan.stable_types,
    )
    return T_truth, plan


def save_transition_matrix_with_labels(T: np.ndarray, labels: List[str | int], out_path: Path) -> None:
    import pandas as pd  # local import

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(T, index=labels, columns=labels)
    df.to_csv(out_path)


def compare_transition_matrices_by_module(
    T_pred: np.ndarray,
    T_truth: np.ndarray,
    plan,
    labels: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Compute cosine similarity per module (averaged over member states) and overall mean."""
    module_scores: List[Dict[str, any]] = []
    for m in plan.modules:
        idxs = list(range(m.start, m.start + m.size))
        sims = []
        for i in idxs:
            vec_true = T_truth[i, idxs]
            vec_pred = T_pred[i, idxs]
            sims.append(_cosine_similarity(vec_true, vec_pred))
        mod_mean = float(np.mean(sims)) if len(sims) > 0 else 0.0
        module_scores.append(
            {
                "module_type": m.module_type,
                "copy_id": int(m.copy_id),
                "start": int(m.start),
                "size": int(m.size),
                "mean_cosine": mod_mean,
            }
        )
    overall = float(np.mean([m["mean_cosine"] for m in module_scores])) if len(module_scores) > 0 else 0.0
    return {
        "overall_mean_cosine": overall,
        "modules": module_scores,
        "labels": labels,
    }


def compare_transition_matrices_by_module_concat(
    T_pred: np.ndarray,
    T_truth: np.ndarray,
    plan,
    labels: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Compute cosine similarity per module by concatenating rows (self+outgoing within module) into one vector."""
    module_scores: List[Dict[str, any]] = []
    for m in plan.modules:
        idxs = list(range(m.start, m.start + m.size))
        # Extract submatrix for this module and flatten row-major
        vec_true = T_truth[np.ix_(idxs, idxs)].reshape(-1)
        vec_pred = T_pred[np.ix_(idxs, idxs)].reshape(-1)
        sim = _cosine_similarity(vec_true, vec_pred)
        module_scores.append(
            {
                "module_type": m.module_type,
                "copy_id": int(m.copy_id),
                "start": int(m.start),
                "size": int(m.size),
                "cosine": float(sim),
            }
        )
    overall = float(np.mean([m["cosine"] for m in module_scores])) if len(module_scores) > 0 else 0.0
    return {
        "overall_mean_cosine": overall,
        "modules": module_scores,
        "labels": labels,
    }


def compare_transition_matrices_by_rows(
    T_pred: np.ndarray,
    T_truth: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Compute cosine similarity per row (full K-dim vector) and overall mean."""
    K = T_pred.shape[0]
    sims = []
    per_row: List[Dict[str, any]] = []
    for i in range(K):
        vec_true = T_truth[i, :]
        vec_pred = T_pred[i, :]
        sim = _cosine_similarity(vec_true, vec_pred)
        sims.append(sim)
        per_row.append({"idx": int(i), "label": None if labels is None else labels[i], "cosine": float(sim)})
    overall = float(np.mean(sims)) if len(sims) > 0 else 0.0
    return {
        "overall_mean_cosine": overall,
        "rows": per_row,
        "labels": labels,
    }


def evaluate_transition_alignment(
    pred_path: Path,
    truth_out_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Load predicted transition matrix (e.g., T_multiple_mean.csv), build default truth, save it, and compare.

    Args:
        pred_path: Path to predicted transition matrix CSV (with labels as header/index).
        truth_out_path: Optional path to save the generated T_truth.csv (defaults to pred_path.parent/T_truth.csv).

    Returns:
        dict with overall_mean_cosine and per-module scores.
    """
    pred_path = Path(pred_path)
    T_pred, labels = _load_transition_matrix_with_labels(pred_path)
    K = T_pred.shape[0]
    T_truth, plan = generate_truth_transition_matrix_from_default(K)

    # Save T_truth
    truth_path = truth_out_path if truth_out_path is not None else pred_path.parent / "T_truth.csv"
    save_transition_matrix_with_labels(T_truth, labels, truth_path)

    return compare_transition_matrices_by_module(
        T_pred=T_pred,
        T_truth=T_truth,
        plan=plan,
        labels=labels,
    )


def evaluate_transition_alignment_concat(
    pred_path: Path,
    truth_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Load predicted/true transition matrices and compare by module-concatenated vectors."""
    pred_path = Path(pred_path)
    T_pred, labels = _load_transition_matrix_with_labels(pred_path)
    K = T_pred.shape[0]
    plan = build_module_plan(K)

    truth_path = Path(truth_path) if truth_path is not None else pred_path.parent / "T_truth.csv"
    T_truth, _labels_truth = _load_transition_matrix_with_labels(truth_path)

    return compare_transition_matrices_by_module_concat(
        T_pred=T_pred,
        T_truth=T_truth,
        plan=plan,
        labels=labels,
    )


def evaluate_transition_alignment_rows(
    pred_path: Path,
    truth_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Load predicted/true transition matrices and compare by full-row cosine similarities."""
    pred_path = Path(pred_path)
    T_pred, labels = _load_transition_matrix_with_labels(pred_path)
    truth_path = Path(truth_path) if truth_path is not None else pred_path.parent / "T_truth.csv"
    T_truth, _labels_truth = _load_transition_matrix_with_labels(truth_path)
    return compare_transition_matrices_by_rows(
        T_pred=T_pred,
        T_truth=T_truth,
        labels=labels,
    )


def evaluate_js_mean_all_intervals(
    *,
    model: torch.nn.Module,
    pools: List[np.ndarray],
    timepoints: np.ndarray,
    time_enc: TimeEncoding,
    num_states: int,
    num_steps_eval: int,
    device: torch.device,
    smoothing: float = 1e-6,
    cost_matrix: Optional[np.ndarray] = None,
    sinkhorn_eps: float = 0.1,
    sinkhorn_iters: int = 300,
    support_only: bool = False,
    smoothing_eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute one-step-ahead metrics over all consecutive intervals.

    This matches the training-time interval evaluation used in train_dfm.py, but is
    exposed here for pipeline/hyperparameter search usage.

    Returns a dict with keys: js_mean, kl_mean, l1_mean.
    """
    from utils import js_divergence, kl_divergence, l1_distance

    js_list, kl_list, l1_list = [], [], []
    w1_list: List[float] = []
    for n in range(len(timepoints) - 1):
        p_start = empirical_distribution(pools[n], num_states=num_states, smoothing=smoothing)
        p_true = empirical_distribution(pools[n + 1], num_states=num_states, smoothing=smoothing)

        t0_norm = float(time_enc.norm_t(np.array([timepoints[n]], dtype=np.float64))[0])
        dt_norm = float(time_enc.norm_dt(np.array([timepoints[n + 1] - timepoints[n]], dtype=np.float64))[0])

        p_pred = propagate_distribution_deterministic(
            model=model,
            p0=p_start,
            t_start_norm=t0_norm,
            dt_norm=dt_norm,
            num_steps=num_steps_eval,
            device=device,
        )

        js_list.append(js_divergence(p_true, p_pred))
        kl_list.append(kl_divergence(p_true, p_pred))
        l1_list.append(l1_distance(p_true, p_pred))
        if cost_matrix is not None:
            w1_val = sinkhorn_w1(
                p=p_true,
                q=p_pred,
                C=cost_matrix,
                eps=float(sinkhorn_eps),
                iters=int(sinkhorn_iters),
                support_only=bool(support_only),
                smoothing_eps=float(smoothing_eps),
            )
            w1_list.append(w1_val)

    return {
        "js_mean": float(np.mean(js_list)),
        "kl_mean": float(np.mean(kl_list)),
        "l1_mean": float(np.mean(l1_list)),
        **({} if cost_matrix is None or len(w1_list) == 0 else {"w1_mean": float(np.mean(w1_list))}),
    }
