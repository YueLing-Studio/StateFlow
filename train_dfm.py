"""
train_dfm.py

Train a global shared DFM denoiser for discrete cell type dynamics using U-coupling
and the replacement path.

Usage (from project root):
    python train_dfm.py --config config/dfm_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from models import DFMModelConfig, DFMTimeDenoiser
from eval import HeldOutEvalConfig, compute_or_load_c_full, evaluate_held_out, write_eval_json
from utils import (
    TimeEncoding,
    build_label_mapping,
    build_pools_by_time,
    compute_time_encoding,
    empirical_distribution,
    ensure_dir,
    get_device,
    load_config,
    propagate_distribution_deterministic,
    read_rdata_dataframe,
    set_seed,
    precompute_ot_mix_for_intervals,   # 新增
    sample_interval_indices,           # 原有（若未引入）
    sample_from_pool,                  # 原有（若未引入）
    sample_pairs_from_coupling,        # 新增（备用）
)



def _make_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optim"]
    return Adam(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )


def _save_checkpoint(
    out_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "meta": meta,
        },
        str(out_path),
    )


def _load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, int, dict]:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0)), ckpt.get("meta", {})


def _sample_training_batch(
    pools: List[np.ndarray],
    timepoints: np.ndarray,
    time_enc: TimeEncoding,
    batch_size: int,
    rng: np.random.Generator,
    pi_mix_list: Optional[List[np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one training batch for coupling-enhanced replacement path.

    If `pi_mix_list` is provided, sample (x0,x1) from the per-interval
    coupling pi_mix_list[n]; otherwise fall back to U-coupling.
    """
    num_intervals = len(timepoints) - 1
    if num_intervals < 1:
        raise ValueError("Need at least 2 distinct timepoints to form intervals.")

    # sample interval indices
    interval_idx = rng.integers(low=0, high=num_intervals, size=batch_size)

    # (x0, x1)
    x0 = np.empty((batch_size,), dtype=np.int64)
    x1 = np.empty((batch_size,), dtype=np.int64)
    t0 = np.empty((batch_size,), dtype=np.float32)
    dt = np.empty((batch_size,), dtype=np.float32)

    if pi_mix_list is not None:
        uniq = np.unique(interval_idx)
        for n in uniq:
            mask = np.where(interval_idx == n)[0]
            ii, jj = sample_pairs_from_coupling(pi_mix_list[int(n)], rng=rng, batch_size=mask.size)
            x0[mask], x1[mask] = ii, jj
            t0[mask] = float(timepoints[int(n)])
            dt[mask] = float(timepoints[int(n)+1] - timepoints[int(n)])
    else:
        for b in range(batch_size):
            n = int(interval_idx[b])
            pool0 = pools[n]; pool1 = pools[n+1]
            x0[b] = pool0[rng.integers(0, len(pool0))]
            x1[b] = pool1[rng.integers(0, len(pool1))]
            t0[b] = float(timepoints[n])
            dt[b] = float(timepoints[n+1] - timepoints[n])

    # s ~ U(0,1), replacement path
    s = rng.random(batch_size, dtype=np.float32)
    choose_x1 = rng.random(batch_size) < s
    x_s = np.where(choose_x1, x1, x0).astype(np.int64)

    # normalize time cond
    t0_norm = time_enc.norm_t(t0.astype(np.float64)).astype(np.float32)
    dt_norm = time_enc.norm_dt(dt.astype(np.float64)).astype(np.float32)

    return (
        torch.from_numpy(x_s).long(),
        torch.from_numpy(s).float(),
        torch.from_numpy(t0_norm).float(),
        torch.from_numpy(dt_norm).float(),
        torch.from_numpy(x1).long(),
    )









@torch.no_grad()
def evaluate_one_step_all_intervals(
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
    """Evaluate model by predicting each p(t_{n+1}) from p(t_n) and averaging metrics."""
    from utils import js_divergence, kl_divergence, l1_distance
    from eval import sinkhorn_w1

    js_list, kl_list, l1_list, w1_list = [], [], [], []

    for n in range(len(timepoints) - 1):
        p_start = empirical_distribution(pools[n], num_states=num_states, smoothing=smoothing)
        p_true = empirical_distribution(pools[n + 1], num_states=num_states, smoothing=smoothing)

        t0_norm = float(time_enc.norm_t(np.array([timepoints[n]]))[0])
        dt_norm = float(time_enc.norm_dt(np.array([timepoints[n + 1] - timepoints[n]]))[0])

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
        "js_std": float(np.std(js_list)),
        "kl_std": float(np.std(kl_list)),
        "l1_std": float(np.std(l1_list)),
        **({} if cost_matrix is None or len(w1_list) == 0 else {
            "w1_mean": float(np.mean(w1_list)),
            "w1_std": float(np.std(w1_list)),
        }),
    }


@torch.no_grad()
def export_transition_matrices(
    *,
    model: torch.nn.Module,
    pools: List[np.ndarray],
    timepoints: np.ndarray,
    time_enc: TimeEncoding,
    num_steps: int,
    device: torch.device,
    out_dir: Path,
    s_end: float = 1.0,
    labels: Optional[List[str | int]] = None,
) -> None:
    """Export per-interval transition matrices to CSV files."""
    from utils import collect_transition_matrices

    import pandas as pd  # local import to avoid hard dependency elsewhere

    base_dir = out_dir / "transition_matrix"
    base_dir.mkdir(parents=True, exist_ok=True)

    interval_means: List[np.ndarray] = []
    interval_products: List[np.ndarray] = []

    num_intervals = len(timepoints) - 1
    for n in range(num_intervals):
        t0_norm = float(time_enc.norm_t(np.array([timepoints[n]], dtype=np.float64))[0])
        dt_norm = float(time_enc.norm_dt(np.array([timepoints[n + 1] - timepoints[n]], dtype=np.float64))[0])

        T_list, _p0 = collect_transition_matrices(
            model=model,
            t_start_norm=t0_norm,
            dt_norm=dt_norm,
            num_steps=num_steps,
            device=device,
            s_end=s_end,
        )

        interval_dir = base_dir / f"interval_{n+1}"
        interval_dir.mkdir(parents=True, exist_ok=True)

        for j, T in enumerate(T_list, start=1):
            if labels is not None:
                df = pd.DataFrame(T, index=labels, columns=labels)
                df.to_csv(interval_dir / f"T_interval{n+1}_step{j}.csv")
            else:
                np.savetxt(interval_dir / f"T_interval{n+1}_step{j}.csv", T, delimiter=",")

        if len(T_list) > 0:
            T_mean = np.mean(np.stack(T_list, axis=0), axis=0)
            T_prod = T_list[0]
            for T in T_list[1:]:
                T_prod = T_prod @ T
            if labels is not None:
                pd.DataFrame(T_mean, index=labels, columns=labels).to_csv(interval_dir / f"T_interval{n+1}_mean.csv")
                pd.DataFrame(T_prod, index=labels, columns=labels).to_csv(interval_dir / f"T_interval{n+1}_multiple.csv")
            else:
                np.savetxt(interval_dir / f"T_interval{n+1}_mean.csv", T_mean, delimiter=",")
                np.savetxt(interval_dir / f"T_interval{n+1}_multiple.csv", T_prod, delimiter=",")
            interval_means.append(T_mean)
            interval_products.append(T_prod)

    if len(interval_means) > 0:
        T_mean_global = np.mean(np.stack(interval_means, axis=0), axis=0)
        if labels is not None:
            pd.DataFrame(T_mean_global, index=labels, columns=labels).to_csv(base_dir / "T_mean.csv")
        else:
            np.savetxt(base_dir / "T_mean.csv", T_mean_global, delimiter=",")

    if len(interval_products) > 0:
        T_prod_global = interval_products[0]
        for T in interval_products[1:]:
            T_prod_global = T_prod_global @ T
        if labels is not None:
            pd.DataFrame(T_prod_global, index=labels, columns=labels).to_csv(base_dir / "T_multiple.csv")
        else:
            np.savetxt(base_dir / "T_multiple.csv", T_prod_global, delimiter=",")
        # Elementwise mean of interval products
        T_multiple_mean = np.mean(np.stack(interval_products, axis=0), axis=0)
        if labels is not None:
            pd.DataFrame(T_multiple_mean, index=labels, columns=labels).to_csv(base_dir / "T_multiple_mean.csv")
        else:
            np.savetxt(base_dir / "T_multiple_mean.csv", T_multiple_mean, delimiter=",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config JSON/YAML.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    experiment_name = str(cfg.get("experiment_name", "default") or "default")
    base_out_dir = ensure_dir(cfg["paths"]["out_dir"])
    out_dir = ensure_dir(base_out_dir / experiment_name)

    def _exp_path(path_value: str | Path, default_name: str) -> Path:
        raw_path = Path(path_value) if path_value is not None else Path(default_name)
        if raw_path.is_absolute():
            return raw_path
        try:
            relative = raw_path.relative_to(base_out_dir)
        except ValueError:
            relative = raw_path
        return out_dir / relative

    def _root_path(path_value: str | Path, default_name: str) -> Path:
        """Resolve a path relative to the *output root* (base_out_dir), not the experiment folder.

        This is used for artifacts meant to be shared across multiple held-out runs
        (e.g., C_full cache).
        """
        raw_path = Path(path_value) if path_value is not None else Path(default_name)
        if raw_path.is_absolute():
            return raw_path
        return base_out_dir / raw_path

    ckpt_path = _exp_path(cfg["paths"].get("checkpoint_path", "dfm_last.pth"), "dfm_last.pth")
    best_ckpt_path = _exp_path(cfg["paths"].get("best_checkpoint_path", "dfm_best.pth"), "dfm_best.pth")

    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))

    # Load data
    df = read_rdata_dataframe(cfg["paths"]["rdata_path"], key=cfg["data"].get("rdata_key", "Data"))
    required = {"celltype", "typeName", "timepoint"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing columns: {missing}. Present: {list(df.columns)}")

    # ----------------------------
    # Held-out handling (training-time masking + label-space unification)
    # ----------------------------
    time_col = cfg["data"].get("time_col", "timepoint")
    label_col = cfg["data"].get("label_col", "celltype")
    type_name_col = cfg["data"].get("name_col", "typeName")

    # Full timepoints (sorted) are used to interpret held_out as a 1-indexed position.
    timepoints_full = np.sort(np.unique(np.asarray(df[time_col]).astype(float)))
    M = int(len(timepoints_full))

    held_out = cfg.get("held_out", False)
    held_out_k: Optional[int] = None
    held_out_time: Optional[float] = None
    held_out_only_raw_labels: List[int] = []
    other_raw_label: Optional[int] = None

    if held_out not in (False, None, 0, "false", "False"):
        try:
            held_out_k = int(held_out)
        except Exception as e:
            raise ValueError(f"Invalid held_out value: {held_out!r}. Use false or an integer k in [2..M].") from e
        if held_out_k < 2 or held_out_k > M:
            raise ValueError(f"held_out must be false or an integer k in [2, {M}], got {held_out_k}.")
        held_out_time = float(timepoints_full[held_out_k - 1])

        # Mask held-out timepoint for training
        df_train_tmp = df[df[time_col].astype(float) != held_out_time].copy()
        all_raw = set(np.asarray(df[label_col]).astype(int).tolist())
        train_raw = set(np.asarray(df_train_tmp[label_col]).astype(int).tolist())
        held_out_only_raw_labels = sorted(list(all_raw - train_raw))

        if len(held_out_only_raw_labels) > 0:
            # Merge held-out-only celltypes into a single "other" raw label *before* label mapping.
            other_raw_label = int(max(all_raw)) + 1
            df = df.copy()
            df[label_col] = np.asarray(df[label_col]).astype(int)
            mask_other = df[label_col].isin(held_out_only_raw_labels)
            df.loc[mask_other, label_col] = other_raw_label
            if type_name_col in df.columns:
                df.loc[mask_other, type_name_col] = "other"

        # Recompute training dataframe after the merge (training timepoint is still masked)
        df_train = df[df[time_col].astype(float) != held_out_time].copy()
    else:
        df_train = df

    # Label mapping is built on the full (possibly merged) dataset so evaluation distributions
    # share the same K-dimensional state space.
    raw_to_idx, idx_to_raw = build_label_mapping(df, label_col=label_col)
    num_states = len(raw_to_idx)

    # Training pools/timepoints exclude held-out timepoint when held_out is enabled.
    timepoints, pools = build_pools_by_time(
        df_train,
        time_col=time_col,
        label_col=label_col,
        raw_to_idx=raw_to_idx,
    )
    time_enc = compute_time_encoding(timepoints)

    # Full pools/timepoints (for held-out evaluation only)
    timepoints_full_after, pools_full = build_pools_by_time(
        df,
        time_col=time_col,
        label_col=label_col,
        raw_to_idx=raw_to_idx,
    )

    # Precompute per-interval coupling pi_mix if enabled (training only)
    coupling_cfg = cfg.get("coupling", {})
    cost_cfg = coupling_cfg.get("cost", {})
    tsne_cols = tuple(cost_cfg.get("cols", ["tSNE_1", "tSNE_2"]))
    centroid_stat = str(cost_cfg.get("centroid_stat", "median")).lower()
    cost_power = int(cost_cfg.get("power", 2))
    cost_norm = cost_cfg.get("normalize", "median_offdiag")

    use_coupling = bool(coupling_cfg.get("enable", True))
    pi_mix_list = None
    ot_precompute_time = 0.0
    if use_coupling:
        _t_ot0 = time.perf_counter()
        ot_eps = float(coupling_cfg.get("ot_eps", 0.1))
        sinkhorn_iters = int(coupling_cfg.get("sinkhorn_iters", 200))
        lam_mix = float(coupling_cfg.get("mix_lambda", 0.1))
        support_only = bool(coupling_cfg.get("support_only", False))
        smoothing_for_marginals = float(coupling_cfg.get("smoothing", cfg["data"].get("smoothing", 1e-6)))

        # Important: coupling cost/OT is computed from *training* data only (held-out timepoint excluded).
        pi_mix_list = precompute_ot_mix_for_intervals(
            df=df_train,
            raw_to_idx=raw_to_idx,
            pools=pools,
            timepoints=timepoints,
            tsne_cols=tsne_cols,
            centroid_stat=centroid_stat,
            cost_power=cost_power,
            cost_normalize=cost_norm,
            ot_eps=ot_eps,
            sinkhorn_iters=sinkhorn_iters,
            lam_mix=lam_mix,
            support_only=support_only,
            smoothing_for_marginals=smoothing_for_marginals,
        )
        ot_precompute_time += (time.perf_counter() - _t_ot0)

    eval_cfg_raw = cfg.get("eval", {})
    c_full_cache_path = _root_path(eval_cfg_raw.get("c_full_cache_path", "C_full.npy"), "C_full.npy")
    eval_sinkhorn_eps = float(eval_cfg_raw.get("sinkhorn_eps", 0.1))
    eval_sinkhorn_iters = int(eval_cfg_raw.get("sinkhorn_iters", 300))
    eval_support_only = bool(eval_cfg_raw.get("support_only", False))
    eval_smoothing_eps = float(eval_cfg_raw.get("smoothing_eps", 1e-8))
    held_out_eval_cfg: Optional[HeldOutEvalConfig] = None

    C_full: Optional[np.ndarray] = None
    # Compute cost matrix for W1 (used for both held-out and training interval W1)
    C_full = compute_or_load_c_full(
        df_full=df,
        raw_to_idx=raw_to_idx,
        tsne_cols=tsne_cols,
        centroid_stat=centroid_stat,
        cost_power=cost_power,
        cost_normalize=cost_norm,
        cache_path=c_full_cache_path,
    )

    # ----------------------------
    # Held-out evaluation: compute (or load) C_full once and reuse
    # ----------------------------
    run_post_training_heldout_eval = bool(eval_cfg_raw.get("run_post_training_heldout_eval", True))
    if held_out_k is not None and run_post_training_heldout_eval:
        held_out_eval_cfg = HeldOutEvalConfig(
            sinkhorn_eps=float(eval_cfg_raw.get("sinkhorn_eps", 0.1)),
            sinkhorn_iters=int(eval_cfg_raw.get("sinkhorn_iters", 300)),
            support_only=bool(eval_cfg_raw.get("support_only", False)),
            smoothing_eps=float(eval_cfg_raw.get("smoothing_eps", 1e-8)),
            interp_steps=int(eval_cfg_raw.get("interp_steps", 50)),
            extrap_steps=int(eval_cfg_raw.get("extrap_steps", int(cfg.get("train", {}).get("num_steps_eval", 25)))),
            c_full_cache_path=c_full_cache_path,
        )

        C_full = compute_or_load_c_full(
            df_full=df,
            raw_to_idx=raw_to_idx,
            tsne_cols=tsne_cols,
            centroid_stat=centroid_stat,
            cost_power=cost_power,
            cost_normalize=cost_norm,
            cache_path=c_full_cache_path,
        )





    # Build model
    mcfg = cfg["model"]
    model_cfg = DFMModelConfig(
        num_states=num_states,
        emb_dim=int(mcfg.get("emb_dim", 64)),
        s_condition=bool(mcfg.get("s_condition", True)),
        t_condition=bool(mcfg.get("t_condition", True)),
        time_mlp_hidden=int(mcfg.get("time_mlp_hidden", 64)),
        trunk_hidden=int(mcfg.get("trunk_hidden", 128)),
        trunk_layers=int(mcfg.get("trunk_layers", 3)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    model = DFMTimeDenoiser(model_cfg).to(device)

    optimizer = _make_optimizer(cfg, model)
    start_step, start_epoch = 0, 0
    train_time_seconds = 0.0

    # Resume if requested and checkpoint exists
    if bool(cfg["train"].get("resume_if_exists", True)) and ckpt_path.exists():
        try:
            start_step, start_epoch, _meta = _load_checkpoint(ckpt_path, model, optimizer)
            train_time_seconds = float((_meta or {}).get("train_time_seconds", 0.0))
            print(f"[Resume] Loaded checkpoint: {ckpt_path} (epoch={start_epoch}, step={start_step}, train_time_seconds={train_time_seconds:.2f})")
        except Exception as e:
            # Most common cause is model signature/shape changes (e.g., different conditioning).
            print(f"[Resume] Failed to load checkpoint: {ckpt_path} ({type(e).__name__}: {e}). Starting from scratch.")
            model = DFMTimeDenoiser(model_cfg).to(device)
            optimizer = _make_optimizer(cfg, model)
            start_step, start_epoch = 0, 0
            train_time_seconds = 0.0

    # Training loop
    train_cfg = cfg["train"]
    batch_size = int(train_cfg.get("batch_size", 1024))
    total_epochs = int(train_cfg.get("epochs", 50))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 200))
    eval_every = int(train_cfg.get("eval_every_epochs", 1))
    print_every = int(train_cfg.get("print_every_epochs", 10))
    log_every_steps = int(train_cfg.get("log_every_steps", 0))
    save_loss_history = bool(train_cfg.get("save_loss_history", True))
    loss_history_path = _exp_path(train_cfg.get("loss_history_path", "loss_history.json"), "loss_history.json")

    num_steps_eval = int(train_cfg.get("num_steps_eval", 20))

    use_amp = bool(train_cfg.get("amp", False)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 12345)

    best_js = float("inf")

    loss_history: List[dict] = []

    sampling_time = 0.0
    transfer_time = 0.0
    train_core_time = 0.0
    eval_time_total = 0.0
    wall_clock_start = time.perf_counter()

    try:
        import psutil  # type: ignore
        _proc = psutil.Process()
        peak_rss = float(_proc.memory_info().rss)
    except Exception:
        psutil = None
        _proc = None
        peak_rss = 0.0

    peak_gpu_mem = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    meta = {
        "experiment_name": experiment_name,
        "timepoints_train": timepoints.tolist(),
        "timepoints_full": timepoints_full_after.tolist(),
        "held_out": False if held_out_k is None else int(held_out_k),
        "held_out_time": None if held_out_time is None else float(held_out_time),
        "held_out_only_raw_labels": held_out_only_raw_labels,
        "other_raw_label": other_raw_label,
        "raw_to_idx": raw_to_idx,
        "idx_to_raw": idx_to_raw,
        "time_encoding": {
            "t_min": time_enc.t_min, "t_max": time_enc.t_max,
            "dt_min": time_enc.dt_min, "dt_max": time_enc.dt_max,
        },
        "c_full_cache_path": None if (held_out_eval_cfg is None or held_out_eval_cfg.c_full_cache_path is None) else str(held_out_eval_cfg.c_full_cache_path),
        "config_path": str(Path(args.config).resolve()),
    }

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running = 0.0

        for _ in range(steps_per_epoch):
            _t_sample0 = time.perf_counter()
            x_s, s, t0_norm, dt_norm, x1 = _sample_training_batch(
                pools=pools,
                timepoints=timepoints,
                time_enc=time_enc,
                batch_size=batch_size,
                rng=rng,
                pi_mix_list=pi_mix_list,  
            )
            sampling_time += (time.perf_counter() - _t_sample0)

            _t_transfer0 = time.perf_counter()
            x_s = x_s.to(device)
            s = s.to(device)
            t0_norm = t0_norm.to(device)
            dt_norm = dt_norm.to(device)
            x1 = x1.to(device)
            transfer_time += (time.perf_counter() - _t_transfer0)

            _t_train0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_s, s, t0_norm, dt_norm)  # [B,K]
                loss = criterion(logits, x1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _train_elapsed = (time.perf_counter() - _t_train0)
            train_time_seconds += _train_elapsed
            train_core_time += _train_elapsed

            running += float(loss.detach().cpu().item())
            start_step += 1

            if log_every_steps > 0 and (start_step % log_every_steps == 0):
                # Instantaneous loss snapshot (not averaged)
                print(f"    [Step {start_step}] loss={float(loss.detach().cpu().item()):.5f}")

            if _proc is not None:
                try:
                    peak_rss = max(peak_rss, float(_proc.memory_info().rss))
                except Exception:
                    pass
            if device.type == "cuda":
                peak_gpu_mem = max(peak_gpu_mem, float(torch.cuda.max_memory_allocated(device)))

        avg_loss = running / max(steps_per_epoch, 1)

        # Save checkpoint every epoch (fast and safe for 1-day delivery)
        meta["train_time_seconds"] = float(train_time_seconds)
        _save_checkpoint(ckpt_path, model, optimizer, step=start_step, epoch=epoch + 1, meta=meta)

        loss_history.append({"epoch": int(epoch + 1), "step": int(start_step), "train_loss": float(avg_loss)})
        if save_loss_history:
            loss_history_path.parent.mkdir(parents=True, exist_ok=True)
            loss_history_path.write_text(json.dumps(loss_history, indent=2), encoding="utf-8")


        if ((epoch + 1) % print_every == 0) or (epoch == start_epoch) or ((epoch + 1) == total_epochs):
            print(f"[Epoch {epoch+1:03d}/{total_epochs}] loss={avg_loss:.5f} ckpt={ckpt_path}")

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs:
            model.eval()
            _t_eval0 = time.perf_counter()
            metrics = evaluate_one_step_all_intervals(
                model=model,
                pools=pools,
                timepoints=timepoints,
                time_enc=time_enc,
                num_states=num_states,
                num_steps_eval=num_steps_eval,
                device=device,
                smoothing=float(cfg["data"].get("smoothing", 1e-6)),
                cost_matrix=C_full,
                sinkhorn_eps=eval_sinkhorn_eps,
                sinkhorn_iters=eval_sinkhorn_iters,
                support_only=eval_support_only,
                smoothing_eps=eval_smoothing_eps,
            )
            eval_time_total += (time.perf_counter() - _t_eval0)
            if "w1_mean" in metrics:
                print(f"  [Eval] js_mean={metrics['js_mean']:.6f} l1_mean={metrics['l1_mean']:.6f} kl_mean={metrics['kl_mean']:.6f} w1_mean={metrics['w1_mean']:.6f}")
            else:
                print(f"  [Eval] js_mean={metrics['js_mean']:.6f} l1_mean={metrics['l1_mean']:.6f} kl_mean={metrics['kl_mean']:.6f}")
            # Attach eval to the most recent epoch record
            if len(loss_history) > 0 and loss_history[-1].get("epoch", None) == int(epoch + 1):
                loss_history[-1].update({
                    "js_mean": float(metrics['js_mean']),
                    "l1_mean": float(metrics['l1_mean']),
                    "kl_mean": float(metrics['kl_mean']),
                    **({"w1_mean": float(metrics["w1_mean"])} if "w1_mean" in metrics else {}),
                })
                if save_loss_history:
                    loss_history_path.write_text(json.dumps(loss_history, indent=2), encoding="utf-8")


            # Track best and save "best" checkpoint
            if metrics["js_mean"] < best_js:
                best_js = metrics["js_mean"]
                meta["train_time_seconds"] = float(train_time_seconds)
                _save_checkpoint(best_ckpt_path, model, optimizer, step=start_step, epoch=epoch + 1, meta=meta)
                print(f"  [Best] Saved best checkpoint to {best_ckpt_path} (js_mean={best_js:.6f})")

    wall_clock_total = time.perf_counter() - wall_clock_start
    if _proc is not None:
        try:
            peak_rss = max(peak_rss, float(_proc.memory_info().rss))
        except Exception:
            pass
    if device.type == "cuda":
        peak_gpu_mem = max(peak_gpu_mem, float(torch.cuda.max_memory_allocated(device)))

    total_training_time = float(
        ot_precompute_time + sampling_time + transfer_time + train_core_time + eval_time_total
    )

    training_cost = {
        "ot_precompute_seconds": float(ot_precompute_time),
        "sampling_seconds": float(sampling_time),
        "transfer_seconds": float(transfer_time),
        "train_core_seconds": float(train_core_time),
        "eval_seconds": float(eval_time_total),
        "total_training_seconds": total_training_time,
        "wall_clock_seconds": float(wall_clock_total),
        "peak_rss_bytes": float(peak_rss),
    }
    if device.type == "cuda":
        training_cost["peak_gpu_bytes"] = float(peak_gpu_mem)

    training_cost_path = out_dir / "training_cost.json"
    training_cost_path.write_text(json.dumps(training_cost, indent=2), encoding="utf-8")

    # ----------------------------
    # Held-out evaluation (post-training only)
    # ----------------------------
    if held_out_k is not None and run_post_training_heldout_eval:
        assert held_out_eval_cfg is not None, "Held-out config was not constructed."
        assert C_full is not None, "C_full was not computed for held-out evaluation."

        # Evaluate with the selected checkpoint (best on training metrics).
        eval_ckpt = best_ckpt_path if best_ckpt_path.exists() else ckpt_path
        if eval_ckpt.exists():
            ckpt = torch.load(str(eval_ckpt), map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            print(f"[Held-out] Loaded checkpoint for evaluation: {eval_ckpt}")
        else:
            print(f"[Held-out] Warning: checkpoint not found at {eval_ckpt}; evaluating current in-memory model.")

        model.eval()
        held_out_metrics = evaluate_held_out(
            model=model,
            held_out_k=int(held_out_k),
            timepoints_full=timepoints_full_after,
            pools_full=pools_full,
            timepoints_train=timepoints,
            pools_train=pools,
            time_enc_train=time_enc,
            num_states=num_states,
            device=device,
            C_full=C_full,
            cfg=held_out_eval_cfg,
            data_smoothing=float(cfg["data"].get("smoothing", 1e-6)),
        )

        print(
            "[Held-out] "
            f"k={int(held_out_k)} mode={held_out_metrics['mode']} "
            f"s*={held_out_metrics['s_star']:.4f} "
            f"W1(Sinkhorn)={held_out_metrics['w1_sinkhorn']:.6f} "
            f"dt_used={held_out_metrics['dt_used']:.4f} dt_is_ood={held_out_metrics['dt_is_ood']}"
        )

        if bool(held_out_metrics.get("dt_is_ood", False)):
            print(
                "[Held-out] Warning: dt_used is outside the training dt range "
                f"([{held_out_metrics['dt_train_min']:.4f}, {held_out_metrics['dt_train_max']:.4f}]). "
                "This is expected for extrapolation and some interpolation settings."
            )

        held_out_eval_path = _exp_path(cfg.get("eval", {}).get("held_out_eval_path", "held_out_eval.json"), "held_out_eval.json")
        write_eval_json(
            held_out_eval_path,
            {
                "held_out": int(held_out_k),
                "held_out_time": float(held_out_time) if held_out_time is not None else None,
                "checkpoint_used": str(eval_ckpt),
                "metrics": held_out_metrics,
            },
        )
        print(f"[Held-out] Saved evaluation to {held_out_eval_path}")

    # ----------------------------
    # Export transition matrices from best checkpoint
    # ----------------------------
    export_steps = int(cfg.get("inference", {}).get("num_steps", num_steps_eval))
    export_ckpt = best_ckpt_path if best_ckpt_path.exists() else ckpt_path
    labels_for_export: Optional[List[str | int]] = None
    # Prefer raw celltype labels (idx_to_raw), fallback to typeName if available
    labels_for_export = [idx_to_raw[i] for i in range(num_states)]
    if export_ckpt.exists():
        ckpt = torch.load(str(export_ckpt), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        export_transition_matrices(
            model=model,
            pools=pools,
            timepoints=timepoints,
            time_enc=time_enc,
            num_steps=export_steps,
            device=device,
            out_dir=out_dir,
            s_end=1.0,
            labels=labels_for_export,
        )
        print(f"[Export] Saved transition matrices to {out_dir / 'transition_matrix'} using checkpoint {export_ckpt}")
    else:
        print(f"[Export] Skip transition matrix export because checkpoint not found at {export_ckpt}")


if __name__ == "__main__":
    main()
