"""scl_dfm_new.generate_sim

Simulation utilities for generating *clean* time-series single-cell expression data
compatible with the existing DFM pipeline in this repository.

Design goals (for simulation data):
1) Per-cell table (one row per cell) with gene expression columns.
2) A *known* discrete state space (15 true cell types by default) whose
   population proportions evolve across 4 timepoints via a sparse Markov
   transition graph.
3) Expression geometry reflects the *dynamical modules* and their adjacency.

   The current default generator uses a **hierarchical factor + transitional mixing**
   design:

   - Hierarchical factors:
       * Module-level gene blocks are strong and large, making cross-module
         differences dominate the geometry.
       * Within-module programs (stage/branch/sink) are weaker/smaller and overlap
         along edges, making adjacent types within a module closer.
       * Small type-specific programs preserve discrete clusterability.

   - Transitional mixing ("bridge" cells):
       For a small fraction of cells in types connected by an edge in the
       transition graph, we generate expression means as a convex mixture of the
       two endpoint type means. This creates kNN connectivity between adjacent
       types so that UMAP/tSNE can recover module structure (avoiding
       "isolated islands").

4) Minimal housekeeping genes (default 5) to avoid drowning the designed signal.

This file does NOT modify any model/training code. It only generates data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import string


# -----------------------------
# Public config + helpers
# -----------------------------


@dataclass
class SimulationConfig:
    """Configuration for simulation data generation.

    This simulator produces a per-cell time-series single-cell expression table and the
    corresponding *true* population proportions P[t, k] for a discrete state space of
    cell types.

    State space (true types):
      - The baseline design has K=15 true types arranged into 5 dynamical modules:
        Module A (chain):      4 types
        Module B (branch):     5 types
        Module C (converge):   3 types
        Module D (absorb):     2 types
        Module S (stable):     1 type
      - For K > 15 (K must be >= 15 and a multiple of 5), the simulator *replicates* these
        module templates to reach the requested K. Replicates are independent (no cross-
        replicate transitions).

    Default indexing convention:
      - The first 15 types match the original K=15 layout:
        A0: 0-3, B0: 4-8, C0: 9-11, D0: 12-13, S0: 14
      - Additional module replicates are appended in a deterministic order.
    """

    # Structure
    n_timepoints: int = 4
    # Total number of true cell types K (must be >= 15 and a multiple of 5).
    n_true_celltypes: int = 15
    # Number of genes G. If None, it will be inferred from K and the program sizes below.
    n_genes: Optional[int] = None
    # Safety margin added on top of the inferred gene requirement (only used if n_genes is None).
    n_genes_buffer: int = 32
    cells_per_timepoint: int = 3000
    rng_seed: int = 7

    # Expression levels (Poisson means)
    # - low_mean: background for non-program genes
    # - module_level: strong module-defining program level (drives cross-module distance)
    # - path_level: within-module stage/branch/sink programs (drives local structure)
    # - type_level: small type-specific tweak (keeps clustering stable without destroying geometry)
    low_mean: float = 1.5
    module_level: float = 15.0
    path_level: float = 6.0
    type_level: float = 4.0

    housekeeping_mean: float = 5.0
    housekeeping_n: int = 5

    # Optional: normalize each cell's mean expression so library size does not dominate embedding.
    # Set to None to disable.
    target_total_mean: Optional[float] = 900.0

    # Transitional mixing ("bridge" cells) to create kNN connectivity between adjacent states.
    # - bridge_frac: fraction of cells in each type that are generated as a mixture with a neighbor
    # - bridge_lambda: mixture strength; 0.2 means 20% neighbor + 80% own
    bridge_frac: float = 0.15
    bridge_lambda: float = 0.25

    # Gene program sizes
    # Module-defining programs (large and strong; ensure cross-module distance dominates)
    module_A_block: int = 25
    module_B_block: int = 25
    module_C_block: int = 20
    module_D_block: int = 15
    module_S_block: int = 15

    # Module A: overlapping stage programs (sliding window) to induce local chain geometry
    A_stage_n_sets: int = 5
    A_stage_set_size: int = 8

    # Module B: progenitor -> branches -> mature
    B_progenitor: int = 10
    B_early_common: int = 6
    B_branch1: int = 8
    B_branch2: int = 8
    B_mature1: int = 8
    B_mature2: int = 8

    # Module C: two sources converge to sink
    C_source_common: int = 6
    C_source9: int = 8
    C_source10: int = 8
    C_sink11: int = 10

    # Module D: absorb
    D_source12: int = 8
    D_sink13: int = 8

    # Type-specific programs (small)
    type_specific_size: int = 3

    # Dynamics
    # If None, stable types are inferred from the module plan (all Module S replicates).
    stable_types: Optional[List[int]] = None
    # Total mass assigned to stable types at t=0 (the total is fixed; it is evenly split across stable types).
    stable_total: float = 0.02
    # If None, transitions are generated from the replicated module templates.
    transitions: Optional[List[Tuple[int, int, float]]] = None
    # If None, p0 source-bias factors are generated from the replicated module templates.
    p0_bias_sources: Optional[Dict[int, float]] = None

    # Batch-effect simulation
    # - batch_effect=False keeps the original clean simulator behavior exactly.
    # - If batch_effect=True, gene_bias and detection_dropout can be switched on/off independently.
    # - gene_bias_sd controls the timepoint-specific log-scale multiplicative gene-bias strength.
    #   It can be either a scalar or a length-n_timepoints sequence, e.g. [0.05, 0.10, 0.08, 0.10].
    #   The generated B[t, :] vector is internally mean-centered; no reference timepoint is used.
    # - dropout_pattern directly specifies the timepoint-specific dropout severities alpha_t.
    #   If None, a mild default pattern is generated according to n_timepoints.
    batch_effect: bool = False
    gene_bias: bool = True
    detection_dropout: bool = True
    gene_bias_sd: Optional[Union[float, Sequence[float]]] = None
    dropout_pattern: Optional[Union[float, Sequence[float]]] = None
    dropout_beta: float = 1.0

def int_to_type_name(i: int) -> str:
    """Map 0->typeA, 1->typeB, ..., 25->typeZ, 26->typeAA, ..."""
    letters = string.ascii_uppercase
    if i < 0:
        raise ValueError("i must be non-negative")
    out = ""
    x = i
    while True:
        out = letters[x % 26] + out
        x = x // 26 - 1
        if x < 0:
            break
    return f"type{out}"


# -----------------------------
# Dynamics: proportions over time
# -----------------------------


def build_transition_matrix(
    K: int,
    transitions: Sequence[Tuple[int, int, float]],
    stable_types: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Create a row-stochastic KxK transition matrix with sparse off-diagonal moves."""
    T = np.eye(K, dtype=float)
    stable_set = set(stable_types or [])

    for src, dst, alpha in transitions:
        if src in stable_set or dst in stable_set:
            raise ValueError(
                f"Transition ({src}->{dst}) touches stable type; not allowed (keeps stability exact)."
            )
        if not (0 <= src < K and 0 <= dst < K):
            raise ValueError(f"Transition indices out of range: ({src}->{dst}) for K={K}")
        if src == dst:
            continue
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        T[src, dst] += alpha
        T[src, src] -= alpha

    row_sums = T.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        bad = np.where(row_sums.squeeze() <= 0)[0].tolist()
        raise ValueError(f"Invalid transition matrix; non-positive row sums for rows: {bad}")
    T = T / row_sums

    # Enforce stable identities (exact self-loop)
    for k in stable_set:
        T[k, :] = 0.0
        T[k, k] = 1.0
    return T


def simulate_proportions(
    K: int,
    n_timepoints: int,
    transitions: Sequence[Tuple[int, int, float]],
    stable_types: Optional[Sequence[int]] = None,
    stable_each: float = 0.02,
    p0_bias_sources: Optional[Dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate P[t, k] with p_{t+1} = p_t @ T, while keeping stable types fixed."""
    rng = rng or np.random.default_rng(0)
    stable_set = set(stable_types or [])
    dynamic = [k for k in range(K) if k not in stable_set]

    # stable mass
    p_stable = np.zeros(K, dtype=float)
    total_stable = 0.0
    if stable_set:
        total_stable = stable_each * len(stable_set)
        if total_stable >= 0.50:
            raise ValueError("Total stable mass too large; reduce stable_each or number of stable types.")
        for k in stable_set:
            p_stable[k] = stable_each

    # initial dynamic distribution
    base = rng.random(len(dynamic)) + 0.1
    p_dyn0 = base / base.sum() * (1.0 - total_stable)
    p0 = p_stable.copy()
    for idx, k in enumerate(dynamic):
        p0[k] = p_dyn0[idx]

    # initial bias (multiplicative on selected dynamic sources)
    if p0_bias_sources:
        weights = np.ones(K, dtype=float)
        for k, w in p0_bias_sources.items():
            if k in stable_set:
                continue
            weights[k] = float(w)
        dyn_mass = p0[dynamic].sum()
        v = p0[dynamic] * weights[dynamic]
        v = v / v.sum() * dyn_mass
        p0[dynamic] = v

    T = build_transition_matrix(K, transitions, stable_types=stable_types)

    P = np.zeros((n_timepoints, K), dtype=float)
    P[0] = p0 / p0.sum()
    for t in range(1, n_timepoints):
        P[t] = P[t - 1] @ T
        P[t] = np.clip(P[t], 0.0, None)
        P[t] = P[t] / P[t].sum()
    return P


# -----------------------------
# Expression: program overlap geometry
# -----------------------------


class _GeneAllocator:
    """Allocate non-overlapping gene index blocks from [0, end)."""

    def __init__(self, end: int):
        self._end = int(end)
        self._cursor = 0

    def alloc(self, n: int, name: str) -> List[int]:
        n = int(n)
        if n < 0:
            raise ValueError("program size must be non-negative")
        if self._cursor + n > self._end:
            raise ValueError(
                f"Not enough genes for program '{name}': need {n}, remaining {self._end - self._cursor}. "
                f"Increase n_genes or reduce program sizes."
            )
        idx = list(range(self._cursor, self._cursor + n))
        self._cursor += n
        return idx


def _default_transitions_for_K15() -> List[Tuple[int, int, float]]:
    # Module A chain
    # Module B branch + maturation
    # Module C convergence
    # Module D absorb
    return [
        (0, 1, 0.20),
        (1, 2, 0.20),
        (2, 3, 0.20),
        (4, 5, 0.12),
        (4, 6, 0.08),
        (5, 7, 0.15),
        (6, 8, 0.15),
        (9, 11, 0.15),
        (10, 11, 0.15),
        (12, 13, 0.18),
    ]


def _default_p0_bias_for_K15() -> Dict[int, float]:
    """Default multiplicative p0 bias factors for the baseline K=15 layout.

    These factors increase the initial probability mass on 'source' states so the
    dynamics are visually apparent within a small number of timepoints (e.g., 4).
    """
    return {0: 2.0, 4: 6.0, 9: 6.0, 10: 6.0, 12: 3.0}

@dataclass(frozen=True)
class ModuleInstance:
    """One replicate of a dynamical module template within the global K-state space."""
    module_type: str  # "A", "B", "C", "D", or "S"
    copy_id: int      # replicate index within the module_type (0-based)
    start: int        # global start index
    size: int         # number of types in this module instance


@dataclass(frozen=True)
class ModulePlan:
    """Resolved module replication plan for a given K."""
    K: int
    a: int  # number of replicates of the (A,C,D,S) group
    b: int  # number of replicates of module B
    modules: Tuple[ModuleInstance, ...]
    stable_types: Tuple[int, ...]


def _validate_K(K: int) -> None:
    if K < 15 or (K % 5) != 0:
        raise ValueError("n_true_celltypes must be >= 15 and a multiple of 5 (e.g., 15, 20, 25, ...).")


def _choose_replication_factors(K: int) -> Tuple[int, int]:
    """Choose (a, b) such that K = 10*a + 5*b with a>=1 and b>=1.

    If multiple solutions exist, choose the most 'balanced' one (minimize |a-b|),
    with deterministic tie-breaking.
    """
    _validate_K(K)
    sols: List[Tuple[int, int]] = []
    for a in range(1, K // 10 + 1):
        rem = K - 10 * a
        if rem < 5:
            continue
        if rem % 5 != 0:
            continue
        b = rem // 5
        if b >= 1:
            sols.append((a, b))
    if not sols:
        # This should not happen for K>=15 and K%5==0, but keep a hard failure.
        raise ValueError(f"Could not find replication factors for K={K} under K=10*a+5*b with a,b>=1.")

    # Balance criterion: minimize |a-b|; then maximize min(a,b); then maximize a.
    sols.sort(key=lambda ab: (abs(ab[0] - ab[1]), -min(ab[0], ab[1]), -ab[0]))
    return sols[0]


def build_module_plan(K: int) -> ModulePlan:
    """Build a deterministic module replication plan.

    The first 15 indices always match the original baseline layout:
      A0: 0-3, B0: 4-8, C0: 9-11, D0: 12-13, S0: 14
    Additional replicates are appended.
    """
    _validate_K(K)
    a, b = _choose_replication_factors(K)

    modules: List[ModuleInstance] = []
    idx = 0

    # Baseline layout (always present)
    modules.append(ModuleInstance("A", 0, idx, 4)); idx += 4
    modules.append(ModuleInstance("B", 0, idx, 5)); idx += 5
    modules.append(ModuleInstance("C", 0, idx, 3)); idx += 3
    modules.append(ModuleInstance("D", 0, idx, 2)); idx += 2
    modules.append(ModuleInstance("S", 0, idx, 1)); idx += 1

    # Extra (A,C,D,S) group replicates
    for i in range(1, a):
        modules.append(ModuleInstance("A", i, idx, 4)); idx += 4
        modules.append(ModuleInstance("C", i, idx, 3)); idx += 3
        modules.append(ModuleInstance("D", i, idx, 2)); idx += 2
        modules.append(ModuleInstance("S", i, idx, 1)); idx += 1

    # Extra B replicates
    for j in range(1, b):
        modules.append(ModuleInstance("B", j, idx, 5)); idx += 5

    if idx != K:
        raise RuntimeError(f"Internal error: module plan produced K={idx} but requested K={K}.")

    stable_types = tuple(m.start for m in modules if m.module_type == "S")
    return ModulePlan(K=K, a=a, b=b, modules=tuple(modules), stable_types=stable_types)


def _module_tag(m: ModuleInstance) -> str:
    return f"{m.module_type}{m.copy_id}"


def default_transitions_for_K(K: int) -> List[Tuple[int, int, float]]:
    """Default sparse transition list for arbitrary K via module-template replication."""
    plan = build_module_plan(K)
    transitions: List[Tuple[int, int, float]] = []
    for m in plan.modules:
        o = m.start
        if m.module_type == "A":
            transitions.extend([(o + 0, o + 1, 0.20), (o + 1, o + 2, 0.20), (o + 2, o + 3, 0.20)])
        elif m.module_type == "B":
            transitions.extend([(o + 0, o + 1, 0.12), (o + 0, o + 2, 0.08), (o + 1, o + 3, 0.15), (o + 2, o + 4, 0.15)])
        elif m.module_type == "C":
            transitions.extend([(o + 0, o + 2, 0.15), (o + 1, o + 2, 0.15)])
        elif m.module_type == "D":
            transitions.extend([(o + 0, o + 1, 0.18)])
        elif m.module_type == "S":
            continue
        else:
            raise RuntimeError(f"Unknown module_type: {m.module_type}")
    return transitions


def default_p0_bias_sources_for_K(K: int) -> Dict[int, float]:
    """Default p0 multiplicative bias factors via module-template replication."""
    plan = build_module_plan(K)
    bias: Dict[int, float] = {}
    for m in plan.modules:
        o = m.start
        if m.module_type == "A":
            bias[o + 0] = 2.0
        elif m.module_type == "B":
            bias[o + 0] = 6.0
        elif m.module_type == "C":
            bias[o + 0] = 4.0
            bias[o + 1] = 4.0
        elif m.module_type == "D":
            bias[o + 0] = 3.0
        elif m.module_type == "S":
            continue
        else:
            raise RuntimeError(f"Unknown module_type: {m.module_type}")
    return bias


def infer_n_genes(cfg: SimulationConfig, plan: Optional[ModulePlan] = None) -> int:
    """Infer the minimum number of genes required by the hierarchical program allocator.

    This keeps the same *program sizing logic* as the original K=15 generator, but scales the
    number of programs with the requested K via module replication.
    """
    K = int(cfg.n_true_celltypes)
    plan = plan or build_module_plan(K)

    hk_n = int(cfg.housekeeping_n)
    required = hk_n  # housekeeping reserved at the end

    # Module-level blocks + path-level programs per module replicate
    for m in plan.modules:
        if m.module_type == "A":
            required += int(cfg.module_A_block)
            required += int(cfg.A_stage_n_sets) * int(cfg.A_stage_set_size)
        elif m.module_type == "B":
            required += int(cfg.module_B_block)
            required += int(cfg.B_progenitor + cfg.B_early_common + cfg.B_branch1 + cfg.B_branch2 + cfg.B_mature1 + cfg.B_mature2)
        elif m.module_type == "C":
            required += int(cfg.module_C_block)
            required += int(cfg.C_source_common + cfg.C_source9 + cfg.C_source10 + cfg.C_sink11)
        elif m.module_type == "D":
            required += int(cfg.module_D_block)
            required += int(cfg.D_source12 + cfg.D_sink13)
        elif m.module_type == "S":
            required += int(cfg.module_S_block)
        else:
            raise RuntimeError(f"Unknown module_type: {m.module_type}")

    # Type-specific programs (one per type)
    required += int(K) * int(cfg.type_specific_size)

    # Safety margin
    required += int(cfg.n_genes_buffer)

    # A small extra cushion for future edits / notebook experimentation
    required = int(required * 1.05) + 8
    return required

def generate_expression_profiles_hierarchical(
    cfg: SimulationConfig,
    plan: Optional[ModulePlan] = None,
) -> Tuple[np.ndarray, List[str], Dict]:
    """Generate a KxG Poisson-mean matrix using a hierarchical factor design.

    The mean profile for each type is constructed as:
      base(low_mean) + module_block(module_level) + path_programs(path_level) + type_specific(type_level)

    Extension to K > 15:
      - The module templates (A/B/C/D/S) are replicated to reach K.
      - Each module replicate receives its *own* module-level gene block, so module replicates are
        separated primarily by module-level structure (treating replicates as distinct modules).
      - Replicates are otherwise structurally identical within each module template.
    """
    K = int(cfg.n_true_celltypes)
    plan = plan or build_module_plan(K)

    if cfg.n_genes is None:
        raise ValueError("cfg.n_genes is None; set it before calling generate_expression_profiles_hierarchical (or call infer_n_genes).")
    G = int(cfg.n_genes)

    hk_n = int(cfg.housekeeping_n)
    if hk_n < 0 or hk_n >= G:
        raise ValueError("housekeeping_n must satisfy 0 <= housekeeping_n < n_genes")

    # Reserve last hk_n genes for housekeeping (never used by any program)
    non_hk_end = G - hk_n
    alloc = _GeneAllocator(end=non_hk_end)

    programs: Dict[str, List[int]] = {}

    # Module-level blocks + path programs (allocated per module replicate)
    for m in plan.modules:
        tag = _module_tag(m)

        if m.module_type == "A":
            programs[f"{tag}_block"] = alloc.alloc(cfg.module_A_block, f"{tag}_block")
            for s in range(int(cfg.A_stage_n_sets)):
                programs[f"{tag}_stage{s}"] = alloc.alloc(cfg.A_stage_set_size, f"{tag}_stage{s}")

        elif m.module_type == "B":
            programs[f"{tag}_block"] = alloc.alloc(cfg.module_B_block, f"{tag}_block")
            programs[f"{tag}_progenitor"] = alloc.alloc(cfg.B_progenitor, f"{tag}_progenitor")
            programs[f"{tag}_early_common"] = alloc.alloc(cfg.B_early_common, f"{tag}_early_common")
            programs[f"{tag}_branch1"] = alloc.alloc(cfg.B_branch1, f"{tag}_branch1")
            programs[f"{tag}_branch2"] = alloc.alloc(cfg.B_branch2, f"{tag}_branch2")
            programs[f"{tag}_mature1"] = alloc.alloc(cfg.B_mature1, f"{tag}_mature1")
            programs[f"{tag}_mature2"] = alloc.alloc(cfg.B_mature2, f"{tag}_mature2")

        elif m.module_type == "C":
            programs[f"{tag}_block"] = alloc.alloc(cfg.module_C_block, f"{tag}_block")
            programs[f"{tag}_source_common"] = alloc.alloc(cfg.C_source_common, f"{tag}_source_common")
            programs[f"{tag}_source0"] = alloc.alloc(cfg.C_source9, f"{tag}_source0")
            programs[f"{tag}_source1"] = alloc.alloc(cfg.C_source10, f"{tag}_source1")
            programs[f"{tag}_sink"] = alloc.alloc(cfg.C_sink11, f"{tag}_sink")

        elif m.module_type == "D":
            programs[f"{tag}_block"] = alloc.alloc(cfg.module_D_block, f"{tag}_block")
            programs[f"{tag}_source"] = alloc.alloc(cfg.D_source12, f"{tag}_source")
            programs[f"{tag}_sink"] = alloc.alloc(cfg.D_sink13, f"{tag}_sink")

        elif m.module_type == "S":
            programs[f"{tag}_block"] = alloc.alloc(cfg.module_S_block, f"{tag}_block")

        else:
            raise RuntimeError(f"Unknown module_type: {m.module_type}")

    # Type-specific micro programs
    type_specific: Dict[int, List[int]] = {}
    for k in range(K):
        name = f"type{k}_spec"
        type_specific[k] = alloc.alloc(cfg.type_specific_size, name)
        programs[name] = type_specific[k]

    # Housekeeping genes (never used by programs)
    hk_idx = alloc.alloc(hk_n, "housekeeping") if hk_n > 0 else []
    programs["housekeeping"] = hk_idx

    gene_names = [f"gene{g+1}" for g in range(G)]

    # Initialize base lambdas
    lambdas = np.full((K, G), float(cfg.low_mean), dtype=float)
    if hk_idx:
        lambdas[:, hk_idx] = float(cfg.housekeeping_mean)

    def apply(k: int, prog: str, level: float) -> None:
        idx = programs.get(prog, [])
        if not idx:
            return
        lambdas[k, idx] = np.maximum(lambdas[k, idx], float(level))

    # Track per-type program activations for debugging/inspection
    type_programs: Dict[int, Dict[str, float]] = {k: {} for k in range(K)}

    def tag(k: int, prog: str, level: float) -> None:
        type_programs[k][prog] = float(level)

    # Assign module/path programs to types (replicate the original per-module logic)
    for m in plan.modules:
        tagm = _module_tag(m)
        o = m.start

        if m.module_type == "A":
            for local in range(4):
                k = o + local
                apply(k, f"{tagm}_block", cfg.module_level); tag(k, f"{tagm}_block", cfg.module_level)

            stage_pairs = {
                0: (f"{tagm}_stage0", f"{tagm}_stage1"),
                1: (f"{tagm}_stage1", f"{tagm}_stage2"),
                2: (f"{tagm}_stage2", f"{tagm}_stage3"),
                3: (f"{tagm}_stage3", f"{tagm}_stage4"),
            }
            for local, (p1, p2) in stage_pairs.items():
                k = o + local
                apply(k, p1, cfg.path_level); tag(k, p1, cfg.path_level)
                apply(k, p2, cfg.path_level); tag(k, p2, cfg.path_level)

        elif m.module_type == "B":
            for local in range(5):
                k = o + local
                apply(k, f"{tagm}_block", cfg.module_level); tag(k, f"{tagm}_block", cfg.module_level)

            # progenitor (local 0): progenitor + early_common
            k0 = o + 0
            apply(k0, f"{tagm}_progenitor", cfg.path_level); tag(k0, f"{tagm}_progenitor", cfg.path_level)
            apply(k0, f"{tagm}_early_common", cfg.path_level); tag(k0, f"{tagm}_early_common", cfg.path_level)

            # early branches (local 1/2): early_common + branch marker
            k1 = o + 1
            apply(k1, f"{tagm}_early_common", cfg.path_level); tag(k1, f"{tagm}_early_common", cfg.path_level)
            apply(k1, f"{tagm}_branch1", cfg.path_level); tag(k1, f"{tagm}_branch1", cfg.path_level)

            k2 = o + 2
            apply(k2, f"{tagm}_early_common", cfg.path_level); tag(k2, f"{tagm}_early_common", cfg.path_level)
            apply(k2, f"{tagm}_branch2", cfg.path_level); tag(k2, f"{tagm}_branch2", cfg.path_level)

            # mature (local 3/4): keep branch id + mature marker
            k3 = o + 3
            apply(k3, f"{tagm}_branch1", cfg.path_level); tag(k3, f"{tagm}_branch1", cfg.path_level)
            apply(k3, f"{tagm}_mature1", cfg.path_level); tag(k3, f"{tagm}_mature1", cfg.path_level)

            k4 = o + 4
            apply(k4, f"{tagm}_branch2", cfg.path_level); tag(k4, f"{tagm}_branch2", cfg.path_level)
            apply(k4, f"{tagm}_mature2", cfg.path_level); tag(k4, f"{tagm}_mature2", cfg.path_level)

        elif m.module_type == "C":
            for local in range(3):
                k = o + local
                apply(k, f"{tagm}_block", cfg.module_level); tag(k, f"{tagm}_block", cfg.module_level)

            # sources (local 0/1): source_common + source-specific
            k0 = o + 0
            apply(k0, f"{tagm}_source_common", cfg.path_level); tag(k0, f"{tagm}_source_common", cfg.path_level)
            apply(k0, f"{tagm}_source0", cfg.path_level); tag(k0, f"{tagm}_source0", cfg.path_level)

            k1 = o + 1
            apply(k1, f"{tagm}_source_common", cfg.path_level); tag(k1, f"{tagm}_source_common", cfg.path_level)
            apply(k1, f"{tagm}_source1", cfg.path_level); tag(k1, f"{tagm}_source1", cfg.path_level)

            # sink (local 2): sink program; keep source_common at lower level
            k2 = o + 2
            apply(k2, f"{tagm}_sink", cfg.path_level); tag(k2, f"{tagm}_sink", cfg.path_level)
            apply(k2, f"{tagm}_source_common", cfg.path_level * 0.75); tag(k2, f"{tagm}_source_common", cfg.path_level * 0.75)

        elif m.module_type == "D":
            for local in range(2):
                k = o + local
                apply(k, f"{tagm}_block", cfg.module_level); tag(k, f"{tagm}_block", cfg.module_level)

            k0 = o + 0
            apply(k0, f"{tagm}_source", cfg.path_level); tag(k0, f"{tagm}_source", cfg.path_level)

            k1 = o + 1
            apply(k1, f"{tagm}_sink", cfg.path_level); tag(k1, f"{tagm}_sink", cfg.path_level)

        elif m.module_type == "S":
            k = o
            apply(k, f"{tagm}_block", cfg.module_level); tag(k, f"{tagm}_block", cfg.module_level)

        else:
            raise RuntimeError(f"Unknown module_type: {m.module_type}")

    # Type-specific micro programs (small, to keep clustering stable)
    for k in range(K):
        idx_name = f"type{k}_spec"
        apply(k, idx_name, cfg.type_level)
        tag(k, idx_name, cfg.type_level)

    meta = {
        "gene_names": gene_names,
        "programs": programs,
        "type_programs": type_programs,
        "housekeeping_idx": hk_idx,
        "non_housekeeping_end": non_hk_end,
        "design": "hierarchical_factors",
        "module_plan": {
            "K": plan.K,
            "a": plan.a,
            "b": plan.b,
            "modules": [m.__dict__ for m in plan.modules],
            "stable_types": list(plan.stable_types),
        },
    }
    return lambdas, gene_names, meta
def _neighbors_from_transitions(K: int, transitions: Sequence[Tuple[int, int, float]]) -> List[List[int]]:
    """Build an undirected neighbor list from the directed transition list.

    This is used only for *expression bridging* (transitional mixing) to create local connectivity
    between adjacent states in embedding space.
    """
    nbrs: List[set] = [set() for _ in range(K)]
    for src, dst, _ in transitions:
        if 0 <= src < K and 0 <= dst < K and src != dst:
            nbrs[src].add(dst)
            nbrs[dst].add(src)
    return [sorted(list(s)) for s in nbrs]


# -----------------------------
# Batch effects: timepoint-specific gene bias + detection dropout
# -----------------------------


def _resolve_timepoint_values(
    value: Optional[Union[float, Sequence[float]]],
    n_timepoints: int,
    default: Union[Sequence[float], float],
    name: str,
) -> np.ndarray:
    """Resolve a scalar or length-n_timepoints sequence into a float vector."""
    raw = default if value is None else value
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        out = np.full(int(n_timepoints), float(arr), dtype=float)
    elif arr.ndim == 1 and arr.size == int(n_timepoints):
        out = arr.astype(float, copy=True)
    else:
        raise ValueError(
            f"{name} must be either a scalar or a length-n_timepoints sequence; "
            f"got shape {arr.shape} for n_timepoints={n_timepoints}."
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


def _default_dropout_pattern(n_timepoints: int) -> np.ndarray:
    """Mild default alpha_t pattern for time-dependent detection dropout."""
    n_timepoints = int(n_timepoints)
    if n_timepoints <= 0:
        raise ValueError("n_timepoints must be positive")
    if n_timepoints == 1:
        return np.array([-1.0], dtype=float)
    if n_timepoints == 4:
        # Mild differences: t2 has slightly stronger dropout; t0 is slightly cleaner.
        return np.array([-1.15, -1.00, -0.85, -0.95], dtype=float)
    offsets = 0.15 * np.sin(2.0 * np.pi * np.arange(n_timepoints) / float(n_timepoints))
    return -1.0 + offsets


def _make_gene_bias_log_factors(
    rng: np.random.Generator,
    n_timepoints: int,
    n_genes: int,
    gene_bias_sd: Optional[Union[float, Sequence[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample centered log-scale gene-bias factors B[t, g]."""
    sd = _resolve_timepoint_values(
        gene_bias_sd,
        n_timepoints=n_timepoints,
        default=0.10,
        name="gene_bias_sd",
    )
    if np.any(sd < 0):
        raise ValueError("gene_bias_sd must be non-negative.")
    B = rng.normal(loc=0.0, scale=sd.reshape(-1, 1), size=(int(n_timepoints), int(n_genes)))
    # Fixed design choice: center each timepoint-specific batch vector; no reference timepoint.
    B = B - B.mean(axis=1, keepdims=True)
    return B, sd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arrays."""
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))

# Main: generate per-cell dataframe
# -----------------------------


def generate_simulation_dataframe(cfg: Optional[SimulationConfig] = None) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Generate a clean time-series single-cell dataset.

    Returns
    -------
    df : pandas.DataFrame
        Columns: timepoint, cellname, gene1..geneG, true_celltype, true_typeName
    P : np.ndarray
        Shape (n_timepoints, K): true proportions used to sample true_celltype.
    meta : dict
        Metadata including gene columns and program assignments.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(int(cfg.rng_seed))

    K = int(cfg.n_true_celltypes)
    Tn = int(cfg.n_timepoints)
    M = int(cfg.cells_per_timepoint)

    plan = build_module_plan(K)

    # Resolve dynamics configuration (defaults replicate the baseline module templates).
    stable_types = cfg.stable_types if cfg.stable_types is not None else list(plan.stable_types)
    transitions = cfg.transitions if cfg.transitions is not None else default_transitions_for_K(K)
    p0_bias = cfg.p0_bias_sources if cfg.p0_bias_sources is not None else default_p0_bias_sources_for_K(K)

    # Basic validation for user overrides.
    for k in stable_types:
        if not (0 <= int(k) < K):
            raise ValueError(f"stable_types contains out-of-range index {k} for K={K}")
    stable_set = set(int(k) for k in stable_types)
    for k in (p0_bias or {}).keys():
        if not (0 <= int(k) < K):
            raise ValueError(f"p0_bias_sources contains out-of-range index {k} for K={K}")
        if int(k) in stable_set:
            raise ValueError(f"p0_bias_sources must not include stable type index {k}")

    # Stable mass is fixed in total and split evenly across stable types.
    stable_each = 0.0
    if stable_types:
        stable_each = float(cfg.stable_total) / float(len(stable_types))

    # Resolve gene count (auto-infer if not provided).
    if cfg.n_genes is None:
        cfg.n_genes = int(infer_n_genes(cfg, plan))
    G = int(cfg.n_genes)
    P = simulate_proportions(
        K=K,
        n_timepoints=Tn,
        transitions=transitions,
        stable_types=stable_types,
        stable_each=float(stable_each),
        p0_bias_sources=p0_bias,
        rng=rng,
    )

    lambdas, gene_names, expr_meta = generate_expression_profiles_hierarchical(cfg, plan)
    neighbors = _neighbors_from_transitions(K, transitions)

    # Resolve optional batch-effect configuration. This block is skipped when
    # batch_effect=False, preserving the original RNG stream and output exactly.
    apply_gene_bias = bool(cfg.batch_effect and cfg.gene_bias)
    apply_detection_dropout = bool(cfg.batch_effect and cfg.detection_dropout)
    gene_bias_log_fc = None
    resolved_gene_bias_sd = None
    dropout_alpha_by_time = None
    dropout_mask_fraction_by_time: List[float] = []

    if apply_gene_bias:
        gene_bias_log_fc, resolved_gene_bias_sd = _make_gene_bias_log_factors(
            rng=rng,
            n_timepoints=Tn,
            n_genes=G,
            gene_bias_sd=cfg.gene_bias_sd,
        )

    if apply_detection_dropout:
        dropout_alpha_by_time = _resolve_timepoint_values(
            cfg.dropout_pattern,
            n_timepoints=Tn,
            default=_default_dropout_pattern(Tn),
            name="dropout_pattern",
        )
        if not np.isfinite(float(cfg.dropout_beta)) or float(cfg.dropout_beta) < 0:
            raise ValueError("dropout_beta must be a finite non-negative value.")

    # Vectorized construction (much faster than per-cell dict assembly)
    frames: List[pd.DataFrame] = []
    cell_counter = 1
    for t in range(Tn):
        z = rng.choice(K, size=M, replace=True, p=P[t]).astype(int)

        # Per-cell mean profiles with optional transitional mixing.
        lam_cell = lambdas[z].copy()
        if cfg.bridge_frac > 0 and cfg.bridge_lambda > 0:
            lam = float(cfg.bridge_lambda)
            for k in np.unique(z):
                if k < 0 or k >= K:
                    continue
                if k in set(stable_types):
                    continue
                nbrs = neighbors[k]
                if not nbrs:
                    continue
                idx = np.where(z == k)[0]
                if idx.size == 0:
                    continue
                msk = rng.random(idx.size) < float(cfg.bridge_frac)
                sel = idx[msk]
                if sel.size == 0:
                    continue
                nbr_choice = rng.choice(nbrs, size=sel.size, replace=True)
                lam_cell[sel] = (1.0 - lam) * lambdas[k][None, :] + lam * lambdas[nbr_choice]

        # Optional: normalize expected library size to reduce global scale confounding.
        if cfg.target_total_mean is not None:
            target = float(cfg.target_total_mean)
            totals = lam_cell.sum(axis=1)
            totals = np.maximum(totals, 1e-8)
            scale = (target / totals).reshape(-1, 1)
            lam_cell = lam_cell * scale

        # Optional timepoint-specific gene bias on Poisson means:
        #   lambda_B[t, c, g] = lambda[t, c, g] * exp(B[t, g])
        # Detection dropout, if enabled, is based on the gene-bias-perturbed means.
        if apply_gene_bias:
            lam_cell = lam_cell * np.exp(gene_bias_log_fc[t][None, :])

        dropout_prob = None
        if apply_detection_dropout:
            alpha_t = float(dropout_alpha_by_time[t])
            beta = float(cfg.dropout_beta)
            dropout_prob = _sigmoid(alpha_t - beta * np.log1p(lam_cell))

        X = rng.poisson(lam=lam_cell, size=(M, G)).astype(float)

        if dropout_prob is not None:
            keep_mask = rng.random(size=(M, G)) >= dropout_prob
            dropout_mask_fraction_by_time.append(float(1.0 - keep_mask.mean()))
            X *= keep_mask

        df_block = pd.DataFrame(X, columns=gene_names)
        df_block.insert(0, "cellname", [f"cell{i}" for i in range(cell_counter, cell_counter + M)])
        df_block.insert(0, "timepoint", float(t))
        df_block["celltype"] = z.astype(int)
        df_block["typeName"] = [int_to_type_name(int(i)) for i in z]
        frames.append(df_block)
        cell_counter += M

    df = pd.concat(frames, axis=0, ignore_index=True)
    gene_cols = gene_names
    df = df[["timepoint", "cellname"] + gene_cols + ["celltype", "typeName"]]

    batch_meta = {
        "batch_effect": bool(cfg.batch_effect),
        "batch_effect_gene_bias": bool(apply_gene_bias),
        "batch_effect_detection_dropout": bool(apply_detection_dropout),
        "gene_bias_sd": None if resolved_gene_bias_sd is None else resolved_gene_bias_sd.tolist(),
        "gene_bias_log_fc": None if gene_bias_log_fc is None else gene_bias_log_fc,
        "gene_bias_centered_per_timepoint": bool(apply_gene_bias),
        "gene_bias_reference_timepoint": None,
        "dropout_pattern": None if dropout_alpha_by_time is None else dropout_alpha_by_time.tolist(),
        "dropout_beta": None if not apply_detection_dropout else float(cfg.dropout_beta),
        "dropout_model": None if not apply_detection_dropout else "sigmoid(alpha_t - beta * log1p(lambda_gene_bias))",
        "dropout_mask_fraction_by_time": dropout_mask_fraction_by_time if apply_detection_dropout else None,
    }

    meta = {
        "gene_names": gene_names,
        "gene_cols": gene_cols,
        "stable_types": list(stable_types),
        "transitions": list(transitions),
        "true_proportions": P,
        "batch_meta": batch_meta,
        **expr_meta,
    }
    return df, P, meta