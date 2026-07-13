
"""
pipeline_search.py

Hyperparameter search pipeline for simulation datasets using stage-wise ASHA
and Pareto+knee selection across (metric, train_wall_time) points.

Design constraints:
- Do NOT modify model architecture, objective, or optimizer definitions.
- The pipeline runs training by invoking train_dfm.py repeatedly with resume,
  saving checkpoints at milestone epochs and evaluating only at milestone epochs.
- Evaluation:
    * held_out=false  -> js_mean (training intervals)
    * held_out=k>=2   -> held-out W1 (Sinkhorn), using fixed C_full for this (K, held_out) job.

This module is intended to be called from notebook/pipeline.ipynb.
"""
import dataclasses
import math
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


import json
from pathlib import Path
from typing import Any, Dict, Union

def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


from utils import (
    TimeEncoding,
    build_label_mapping,
    build_pools_by_time,
    compute_time_encoding,
    read_rdata_dataframe,
)
from eval import (
    HeldOutEvalConfig,
    compute_or_load_c_full,
    evaluate_held_out,
    evaluate_js_mean_all_intervals,
)


Json = Dict[str, Any]


@dataclass(frozen=True)
class StageSchedule:
    """Stage schedule for ASHA.

    epochs: list of milestone epochs for this stage (strictly increasing).
    """
    name: str
    epochs: List[int]


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration for a single (K, held_out) job."""
    k_value: int
    held_out: Union[bool, int]
    base_config_path: Path
    rdata_path: Path
    out_root: Path
    experiment_name: str

    # ASHA
    stages: List[StageSchedule]
    eta: int
    n_trials: int
    seed: int

    # Search space: dict of param path -> allowed values
    # Example key: "model.emb_dim", "optim.lr"
    search_space: Dict[str, List[Any]]

    # Fixed training knobs for comparability (can be set in base config, but
    # pipeline can also override them consistently here)
    steps_per_epoch: Optional[int] = None
    num_steps_eval: Optional[int] = None

    # Do not run post-training held-out evaluation inside train_dfm.py; pipeline does it.
    disable_post_training_heldout_eval: bool = True

    # Where to store results table and plots
    results_filename: str = "results.jsonl"
    plot_filename: str = "pareto.png"
    dataset_id: Optional[str] = None  # for real datasets; if set, used for output directory naming


def _deepcopy_json(cfg: Json) -> Json:
    return json.loads(json.dumps(cfg))


def _set_by_dotted_path(cfg: Json, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_by_dotted_path(cfg: Json, dotted: str, default: Any = None) -> Any:
    parts = dotted.split(".")
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _validate_schedule(stages: Sequence[StageSchedule]) -> None:
    if len(stages) == 0:
        raise ValueError("stages must be non-empty.")
    prev_max = -1
    for st in stages:
        if len(st.epochs) == 0:
            raise ValueError(f"Stage '{st.name}' must have at least one epoch.")
        if any(e <= 0 for e in st.epochs):
            raise ValueError(f"Stage '{st.name}' epochs must be positive integers: {st.epochs}")
        if st.epochs != sorted(st.epochs):
            raise ValueError(f"Stage '{st.name}' epochs must be strictly increasing: {st.epochs}")
        if len(set(st.epochs)) != len(st.epochs):
            raise ValueError(f"Stage '{st.name}' epochs contain duplicates: {st.epochs}")
        if min(st.epochs) <= prev_max:
            raise ValueError(
                f"Stage '{st.name}' min epoch {min(st.epochs)} must be > previous stage max {prev_max}."
            )
        prev_max = max(st.epochs)


def _milestones(stages: Sequence[StageSchedule]) -> List[int]:
    e: List[int] = []
    for st in stages:
        e.extend(st.epochs)
    return sorted(list(dict.fromkeys(e)))


def _load_base_cfg(path: Path) -> Json:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _run_train(config_path: Path, project_root: Path) -> None:
    cmd = [sys.executable, "-u", str(project_root / "train_dfm.py"), "--config", str(config_path)]
    proc = subprocess.run(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed (exit={proc.returncode}). Output:\n{proc.stdout}")


def _read_checkpoint_train_time_seconds(ckpt_path: Path) -> float:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    meta = ckpt.get("meta", {}) or {}
    return float(meta.get("train_time_seconds", 0.0))


def _load_model_from_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _build_model_for_eval(trial_cfg: Json, num_states: int) -> torch.nn.Module:
    """Build the denoiser model for evaluation.

    NOTE: models.DFMTimeDenoiser expects a `DFMModelConfig` dataclass, not a raw dict.
    We mirror the construction used in train_dfm.py to ensure architectural parity.
    """

    from models import DFMModelConfig, DFMTimeDenoiser  # local import to avoid side effects

    mcfg = trial_cfg.get("model", {}) or {}
    model_cfg = DFMModelConfig(
        num_states=int(num_states),
        emb_dim=int(mcfg.get("emb_dim", 64)),
        s_condition=bool(mcfg.get("s_condition", True)),
        t_condition=bool(mcfg.get("t_condition", True)),
        time_mlp_hidden=int(mcfg.get("time_mlp_hidden", 64)),
        trunk_hidden=int(mcfg.get("trunk_hidden", 128)),
        trunk_layers=int(mcfg.get("trunk_layers", 3)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    return DFMTimeDenoiser(model_cfg)


def _prepare_full_and_train_pools(
    *,
    rdata_path: Path,
    rdata_key: str,
    held_out: Union[bool, int],
    time_col: str,
    label_col: str,
    type_name_col: str,
) -> Tuple[
    Any,  # df_full_after_merge (pandas DataFrame)
    Json,  # df_meta
    np.ndarray, List[np.ndarray],  # timepoints_full, pools_full
    np.ndarray, List[np.ndarray],  # timepoints_train, pools_train
    TimeEncoding,  # time_enc_train
    Dict[int, int],  # raw_to_idx
]:
    """Replicate the preprocessing logic in train_dfm.py (including held-out-only -> other merge)."""
    df = read_rdata_dataframe(str(rdata_path), key=rdata_key)

    # Ensure numeric types
    df[time_col] = df[time_col].astype(float)
    df[label_col] = df[label_col].astype(int)

    timepoints_full = np.sort(np.unique(df[time_col].to_numpy(dtype=float)))
    M = int(len(timepoints_full))

    held_out_k: Optional[int] = None
    held_out_time: Optional[float] = None
    held_out_only_raw_labels: List[int] = []
    other_raw_label: Optional[int] = None

    if held_out not in (False, None, 0, "false", "False"):
        held_out_k = int(held_out)
        if held_out_k < 2 or held_out_k > M:
            raise ValueError(f"held_out must be false or an integer k in [2,{M}], got {held_out_k}")
        held_out_time = float(timepoints_full[held_out_k - 1])

        # Mask held-out for training to detect held-out-only labels
        df_train_tmp = df[df[time_col].astype(float) != held_out_time].copy()
        all_raw = set(df[label_col].astype(int).tolist())
        train_raw = set(df_train_tmp[label_col].astype(int).tolist())
        held_out_only_raw_labels = sorted(list(all_raw - train_raw))

        if len(held_out_only_raw_labels) > 0:
            other_raw_label = int(max(all_raw)) + 1
            df = df.copy()
            mask_other = df[label_col].isin(held_out_only_raw_labels)
            df.loc[mask_other, label_col] = other_raw_label
            if type_name_col in df.columns:
                df.loc[mask_other, type_name_col] = "other"

        df_train = df[df[time_col].astype(float) != held_out_time].copy()
    else:
        df_train = df

    raw_to_idx, _idx_to_raw = build_label_mapping(df, label_col=label_col)
    timepoints_train, pools_train = build_pools_by_time(df_train, time_col=time_col, label_col=label_col, raw_to_idx=raw_to_idx)
    timepoints_full_after, pools_full = build_pools_by_time(df, time_col=time_col, label_col=label_col, raw_to_idx=raw_to_idx)

    time_enc_train = compute_time_encoding(timepoints_train)

    df_meta: Json = {
        "timepoints_full": timepoints_full_after.tolist(),
        "timepoints_train": timepoints_train.tolist(),
        "held_out_k": False if held_out_k is None else int(held_out_k),
        "held_out_time": None if held_out_time is None else float(held_out_time),
        "held_out_only_raw_labels": held_out_only_raw_labels,
        "other_raw_label": None if other_raw_label is None else int(other_raw_label),
        "num_states": int(len(raw_to_idx)),
    }
    return df, df_meta, timepoints_full_after, pools_full, timepoints_train, pools_train, time_enc_train, raw_to_idx


def _trial_dir(job_dir: Path, trial_id: int) -> Path:
    return job_dir / f"trial_{trial_id:04d}"


def _sample_trial_hparams(rng: np.random.Generator, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    for k, vals in search_space.items():
        if len(vals) == 0:
            raise ValueError(f"search_space[{k}] is empty.")
        hp[k] = vals[int(rng.integers(0, len(vals)))]
    return hp


def _compute_stage_score(metrics_by_epoch: Dict[int, float], stage_epochs: Sequence[int]) -> float:
    vals = [metrics_by_epoch[e] for e in stage_epochs if e in metrics_by_epoch]
    if len(vals) == 0:
        raise ValueError("No metrics available for stage scoring.")
    return float(np.min(vals))


def _pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return Pareto front for minimizing time and metric.

    points: list of dicts containing 'train_time' and 'metric'
    """
    pts = sorted(points, key=lambda d: (float(d["train_time"]), float(d["metric"])))
    front: List[Dict[str, Any]] = []
    best_metric = float("inf")
    for p in pts:
        m = float(p["metric"])
        if m < best_metric - 1e-12:
            front.append(p)
            best_metric = m
    return front


def _knee_point(front: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select knee point on Pareto front using max distance to the line connecting endpoints."""
    if len(front) == 0:
        raise ValueError("Empty Pareto front.")
    if len(front) == 1:
        return front[0]
    # Normalize to [0,1]
    t = np.array([float(p["train_time"]) for p in front], dtype=float)
    y = np.array([float(p["metric"]) for p in front], dtype=float)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    tn = (t - t_min) / (t_max - t_min + 1e-12)
    yn = (y - y_min) / (y_max - y_min + 1e-12)

    # endpoints in normalized space: fastest (tn=0) and best metric (yn min, likely at end)
    x1, y1 = float(tn[0]), float(yn[0])
    x2, y2 = float(tn[-1]), float(yn[-1])

    # line distance for each point
    # distance from point (x0,y0) to line through (x1,y1)-(x2,y2)
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dists = []
    for x0, y0 in zip(tn, yn):
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        dists.append(num / denom)
    idx = int(np.argmax(np.array(dists)))
    return front[idx]


def _plot_pareto(points: List[Dict[str, Any]], front: List[Dict[str, Any]], knee: Dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = [float(p["train_time"]) for p in points]
    y = [float(p["metric"]) for p in points]
    plt.figure()
    plt.scatter(x, y, s=12)

    if len(front) > 0:
        xf = [float(p["train_time"]) for p in front]
        yf = [float(p["metric"]) for p in front]
        plt.plot(xf, yf)

    plt.scatter([float(knee["train_time"])], [float(knee["metric"])], s=50, marker="x")
    plt.xlabel("train_wall_time (seconds)")
    plt.ylabel("metric (lower is better)")
    plt.title("Pareto front and knee")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def run_pipeline_job(job: PipelineConfig) -> Dict[str, Any]:
    """Run hyperparameter search for a single (K, held_out) combination."""
    _validate_schedule(job.stages)
    project_root = job.base_config_path.parent.parent  # config/.. -> project root
    job_base = job.dataset_id if job.dataset_id else f"K{job.k_value}"
    job_dir = job.out_root / job_base / (f"ho{job.held_out}" if job.held_out is not False else "no_ho")
    job_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(job.seed)

    base_cfg = _load_base_cfg(job.base_config_path)
    cfg_proto = _deepcopy_json(base_cfg)

    # Wire in data/paths
    cfg_proto.setdefault("paths", {})
    cfg_proto["paths"]["rdata_path"] = str(job.rdata_path)
    cfg_proto["paths"]["out_dir"] = str(job_dir)
    cfg_proto["experiment_name"] = job.experiment_name
    cfg_proto["held_out"] = job.held_out

    # Make training deterministic-ish per trial (seed controls sampling rng in train_dfm.py)
    cfg_proto.setdefault("train", {})
    cfg_proto["train"]["seed"] = int(job.seed)
    cfg_proto["train"]["resume_if_exists"] = True
    # Reduce internal evaluations to end-of-run only
    cfg_proto["train"]["eval_every_epochs"] = int(10**9)

    if job.steps_per_epoch is not None:
        cfg_proto["train"]["steps_per_epoch"] = int(job.steps_per_epoch)
    if job.num_steps_eval is not None:
        cfg_proto["train"]["num_steps_eval"] = int(job.num_steps_eval)

    # Disable post-training held-out evaluation inside train_dfm.py (pipeline evaluates at milestones)
    cfg_proto.setdefault("eval", {})
    if job.disable_post_training_heldout_eval:
        cfg_proto["eval"]["run_post_training_heldout_eval"] = False

    # Data prep for evaluation and C_full (fixed within this job)
    data_cfg = cfg_proto.get("data", {})
    rdata_key = str(data_cfg.get("rdata_key", "Data"))
    time_col = str(data_cfg.get("time_col", "timepoint"))
    label_col = str(data_cfg.get("label_col", "celltype"))
    type_name_col = str(data_cfg.get("type_name_col", "typeName"))

    df_full_after, df_meta, timepoints_full, pools_full, timepoints_train, pools_train, time_enc_train, raw_to_idx = _prepare_full_and_train_pools(
        rdata_path=job.rdata_path,
        rdata_key=rdata_key,
        held_out=job.held_out,
        time_col=time_col,
        label_col=label_col,
        type_name_col=type_name_col,
    )
    num_states = int(df_meta["num_states"])

    # Compute/load C_full for this job (per K & held_out)
    # Cache path inside job_dir so trials reuse it.
    c_cache = job_dir / "C_full.npy"
    # Cost config lives under coupling.cost in dfm_config.json.
    # For robustness, we also accept a legacy top-level "cost".
    coupling_cfg = cfg_proto.get("coupling", {}) or {}
    cost_cfg = coupling_cfg.get("cost", {}) or cfg_proto.get("cost", {}) or {}
    # Column key is "cols" in the current config; accept "tsne_cols" for compatibility.
    tsne_cols = cost_cfg.get("cols", cost_cfg.get("tsne_cols", ["tSNE_1", "tSNE_2"]))
    centroid_stat = str(cost_cfg.get("centroid_stat", "median"))
    cost_power = int(cost_cfg.get("power", 1))
    cost_normalize = cost_cfg.get("normalize", None)

    C_full = compute_or_load_c_full(
        df_full=df_full_after,
        time_col=time_col,
        label_col=label_col,
        raw_to_idx=raw_to_idx,
        tsne_cols=tsne_cols,
        centroid_stat=centroid_stat,
        cost_power=cost_power,
        cost_normalize=cost_normalize,
        cache_path=c_cache,
    )

    # Held-out eval config
    eval_cfg_raw = cfg_proto.get("eval", {})
    held_out_eval_cfg = HeldOutEvalConfig(
        sinkhorn_eps=float(eval_cfg_raw.get("sinkhorn_eps", 0.1)),
        sinkhorn_iters=int(eval_cfg_raw.get("sinkhorn_iters", 300)),
        support_only=bool(eval_cfg_raw.get("support_only", False)),
        interp_steps=int(eval_cfg_raw.get("interp_steps", 50)),
        extrap_steps=int(eval_cfg_raw.get("extrap_steps", 50)),
    )

    # Device for evaluation model forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    milestones_all = _milestones(job.stages)

    # Trial generation
    trials: List[Dict[str, Any]] = []
    seen = set()
    while len(trials) < job.n_trials:
        hp = _sample_trial_hparams(rng, job.search_space)
        key = json.dumps(hp, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append({"trial_id": len(trials), "hparams": hp})

    # Run Stage1 for all trials
    stage_states: Dict[int, Dict[str, Any]] = {}
    for t in trials:
        tid = int(t["trial_id"])
        tdir = _trial_dir(job_dir, tid)
        tdir.mkdir(parents=True, exist_ok=True)
        stage_states[tid] = {
            "trial_id": tid,
            "hparams": t["hparams"],
            "current_epoch": 0,
            "metrics_by_epoch": {},  # epoch -> metric
            "train_time_by_epoch": {},  # epoch -> train_time_seconds
        }

        # Write a base trial config
        trial_cfg = _deepcopy_json(cfg_proto)
        # Per-trial unique experiment name / output paths
        trial_cfg["experiment_name"] = f"{job.experiment_name}_trial{tid:04d}"
        trial_cfg["paths"]["out_dir"] = str(tdir)
        trial_cfg["paths"]["checkpoint_path"] = "dfm_last.pth"
        trial_cfg["paths"]["best_checkpoint_path"] = "dfm_best.pth"
        trial_cfg["train"]["loss_history_path"] = "loss_history.json"
        trial_cfg["train"]["save_loss_history"] = True

        # Apply hyperparams
        for kpath, v in t["hparams"].items():
            _set_by_dotted_path(trial_cfg, kpath, v)

        # Some pipeline-level overrides: batch_size affects speed; ensure int
        if "train.batch_size" in t["hparams"]:
            trial_cfg["train"]["batch_size"] = int(trial_cfg["train"]["batch_size"])

        _write_json(tdir / "config_trial.json", trial_cfg)

    # Stage loop with ASHA
    alive: List[int] = [int(t["trial_id"]) for t in trials]
    results_path = job_dir / job.results_filename
    if results_path.exists():
        results_path.unlink()

    def append_result(rec: Dict[str, Any]) -> None:
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    for si, stage in enumerate(job.stages):
        stage_end = max(stage.epochs)
        # Train/eval each alive trial through all epochs in this stage
        for tid in list(alive):
            st = stage_states[tid]
            tdir = _trial_dir(job_dir, tid)
            cfg_path = tdir / "config_trial.json"
            trial_cfg = _load_base_cfg(cfg_path)

            for e in stage.epochs:
                # Train until epoch e
                trial_cfg["train"]["epochs"] = int(e)
                _write_json(cfg_path, trial_cfg)
                _run_train(cfg_path, project_root=project_root)

                # Copy checkpoint
                # NOTE: train_dfm.py writes checkpoints under:
                #   out_dir_root / experiment_name / <checkpoint_path>
                # In the pipeline, we set out_dir_root = tdir, so the actual checkpoint lives in
                #   tdir / trial_cfg["experiment_name"] / trial_cfg["paths"]["checkpoint_path"]
                exp_name = str(trial_cfg.get("experiment_name", "default") or "default")
                exp_dir = Path(trial_cfg["paths"]["out_dir"]) / exp_name
                ckpt_path = exp_dir / trial_cfg["paths"]["checkpoint_path"]
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing checkpoint after training: {ckpt_path}")
                ckpt_copy = tdir / "checkpoints" / f"ckpt_epoch{e}.pth"
                ckpt_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ckpt_path, ckpt_copy)

                # Read train_time_seconds
                train_time_sec = _read_checkpoint_train_time_seconds(ckpt_path)
                st["train_time_by_epoch"][int(e)] = float(train_time_sec)

                # Evaluate metric at this milestone.
                # Build the model with the same config semantics as train_dfm.py.
                model = _build_model_for_eval(trial_cfg, num_states=num_states)
                model = _load_model_from_checkpoint(model, ckpt_copy, device=device)

                metric_val: float
                if job.held_out is False:
                    # JS mean over training intervals (full timepoints)
                    m = evaluate_js_mean_all_intervals(
                        model=model,
                        pools=pools_full,  # full, because held_out=false
                        timepoints=timepoints_full,
                        time_enc=compute_time_encoding(timepoints_full),
                        num_states=num_states,
                        num_steps_eval=int(trial_cfg["train"].get("num_steps_eval", 25)),
                        device=device,
                        cost_matrix=C_full,
                        sinkhorn_eps=float(held_out_eval_cfg.sinkhorn_eps),
                        sinkhorn_iters=int(held_out_eval_cfg.sinkhorn_iters),
                        support_only=bool(held_out_eval_cfg.support_only),
                    )
                    metric_val = float(m["js_mean"])
                    metric_name = "js_mean"
                    metrics_payload = m
                else:
                    w = evaluate_held_out(
                        model=model,
                        held_out_k=int(job.held_out),
                        timepoints_full=timepoints_full,
                        pools_full=pools_full,
                        timepoints_train=timepoints_train,
                        pools_train=pools_train,
                        time_enc_train=time_enc_train,
                        num_states=num_states,
                        device=device,
                        C_full=C_full,
                        cfg=held_out_eval_cfg,
                    )
                    metric_val = float(w["w1_sinkhorn"])
                    metric_name = "w1_sinkhorn"
                    metrics_payload = w

                st["metrics_by_epoch"][int(e)] = float(metric_val)
                st["current_epoch"] = int(e)

                rec = {
                    "K": int(job.k_value),
                    "dataset_id": job.dataset_id,
                    "held_out": job.held_out,
                    "trial_id": int(tid),
                    "stage": stage.name,
                    "epoch": int(e),
                    "metric_name": metric_name,
                    "metric": float(metric_val),
                    "train_time": float(train_time_sec),
                    "hparams": st["hparams"],
                    "checkpoint": str(ckpt_copy),
                    "metrics_full": metrics_payload,
                }
                append_result(rec)

        # Stage scoring and pruning (skip after last stage)
        if si < len(job.stages) - 1:
            scores = []
            for tid in alive:
                st = stage_states[tid]
                score = _compute_stage_score(st["metrics_by_epoch"], stage.epochs)
                # tie-break: earliest epoch achieving min
                best_ep = min(stage.epochs, key=lambda e: (float(st["metrics_by_epoch"].get(e, float("inf"))), int(e)))
                score = float(st["metrics_by_epoch"].get(best_ep, float("inf")))
                scores.append((score, best_ep, st["train_time_by_epoch"].get(best_ep, float("inf")), tid))
            scores.sort()
            keep_n = max(1, int(np.ceil(len(scores) / max(job.eta, 2))))
            alive = [tid for *_rest, tid in scores[:keep_n]]

    # Aggregate points across all trials and epochs
    all_points: List[Dict[str, Any]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            all_points.append(json.loads(line))

    front = _pareto_front(all_points)
    knee = _knee_point(front)
    _plot_pareto(all_points, front, knee, job_dir / job.plot_filename)

    best = {
        "K": int(job.k_value),
        "dataset_id": job.dataset_id,
        "held_out": job.held_out,
        "best_trial_id": int(knee["trial_id"]),
        "best_stop_epoch": int(knee["epoch"]),
        "best_metric": float(knee["metric"]),
        "best_train_time": float(knee["train_time"]),
        "best_metric_name": knee["metric_name"],
        "best_checkpoint": knee["checkpoint"],
        "best_hparams": knee["hparams"],
        "job_dir": str(job_dir),
    }
    _write_json(job_dir / "best_report.json", best)

    # Also write the best config snapshot
    best_trial_dir = _trial_dir(job_dir, int(knee["trial_id"]))
    best_cfg = _load_base_cfg(best_trial_dir / "config_trial.json")
    best_cfg["train"]["epochs"] = int(knee["epoch"])
    _write_json(job_dir / "best_config.json", best_cfg)

    return best


def run_pipeline(
    *,
    K_list: Sequence[int],
    held_out_list: Sequence[Union[bool, int]],
    base_config_path: Union[str, Path],
    sim_data_dir: Union[str, Path],
    out_root: Union[str, Path],
    stages: Sequence[StageSchedule],
    eta: int,
    n_trials: int,
    seed: int,
    search_space: Dict[str, List[Any]],
    steps_per_epoch: Optional[int] = None,
    num_steps_eval: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run pipeline across multiple K and held-out values."""
    base_config_path = Path(base_config_path)
    sim_data_dir = Path(sim_data_dir)
    out_root = Path(out_root)

    results: List[Dict[str, Any]] = []
    for K in K_list:
        rdata_path = sim_data_dir / f"Data_sim_{int(K)}.RData"
        if not rdata_path.exists():
            raise FileNotFoundError(f"Missing RData: {rdata_path}")
        for ho in held_out_list:
            ho_suffix = f"ho{ho}" if ho is not False else "no_ho"
            exp_name = f"pipeline_sim_K{int(K)}_{ho_suffix}"
            job = PipelineConfig(
                k_value=int(K),
                held_out=ho,
                base_config_path=base_config_path,
                rdata_path=rdata_path,
                out_root=out_root,
                experiment_name=exp_name,
                stages=list(stages),
                eta=int(eta),
                n_trials=int(n_trials),
                seed=int(seed),
                search_space=search_space,
                steps_per_epoch=steps_per_epoch,
                num_steps_eval=num_steps_eval,
            )
            best = run_pipeline_job(job)
            results.append(best)
    return results
def run_pipeline_rdata(
    *,
    rdata_path: Union[str, Path],
    held_out_list: Optional[Sequence[Union[bool, int]]] = None,
    base_config_path: Union[str, Path],
    out_root: Union[str, Path],
    stages: Sequence[Dict[str, Any]],
    eta: int,
    n_trials: int,
    seed: int,
    search_space: Dict[str, List[Any]],
    steps_per_epoch: int,
    num_steps_eval: int,
    dataset_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the pipeline on a single dataset (.RData path), typically a real dataset.

    Mirrors `run_pipeline` but removes the outer loop over K. If `held_out_list`
    is not provided, defaults to [False] (i.e., full training only).
    """
    base_config_path = Path(base_config_path)
    cfg_proto = _load_json(base_config_path)

    rdata_path = Path(rdata_path)
    if not rdata_path.exists():
        raise FileNotFoundError(f"RData file not found: {rdata_path}")

    if held_out_list is None:
        held_out_list = [False]

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ds_id = str(dataset_id) if dataset_id else rdata_path.stem

    time_col = str(_get_by_dotted_path(cfg_proto, "data.time_col", "time"))
    label_col = str(_get_by_dotted_path(cfg_proto, "data.label_col", "cell_type"))
    rdata_key = str(_get_by_dotted_path(cfg_proto, "data.rdata_key", "Data"))

    df_full = read_rdata_dataframe(str(rdata_path), key=rdata_key)
    if label_col not in df_full.columns:
        raise KeyError(
            f"label_col '{label_col}' not found in RData dataframe columns. "
            f"Available columns (first 30): {list(df_full.columns)[:30]}"
        )
    k_nominal = int(df_full[label_col].nunique())

    stage_objs: List[StageSchedule] = []
    for s in stages:
        if isinstance(s, StageSchedule):
            stage_objs.append(s)
        else:
            # assume dict-like
            stage_objs.append(
                StageSchedule(
                    name=str(s.get("name")),
                    epochs=list(map(int, s.get("epochs", []))),
                )
            )

    results: List[Dict[str, Any]] = []
    for ho in held_out_list:
        ho_suffix = f"ho{ho}" if ho is not False else "no_ho"
        exp_name = f"pipeline_real_{ds_id}_{ho_suffix}"
        job = PipelineConfig(
            k_value=k_nominal,
            held_out=ho,
            base_config_path=base_config_path,
            rdata_path=rdata_path,
            out_root=out_root,
            experiment_name=exp_name,
            dataset_id=ds_id,
            stages=stage_objs,
            eta=int(eta),
            n_trials=int(n_trials),
            seed=int(seed),
            search_space=search_space,
            steps_per_epoch=int(steps_per_epoch),
            num_steps_eval=int(num_steps_eval),
        )
        best = run_pipeline_job(job)
        results.append(best)
    return results
