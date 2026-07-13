"""
models.py

Discrete Flow Matching (DFM) denoiser model for discrete-state time series dynamics.
Target use case: cell type dynamics over timepoints (categorical state space of size K).
Compatible with PyTorch 2.6.0+cu126.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass(frozen=True)
class DFMModelConfig:
    """Configuration for the denoiser network.

    Args:
        num_states: K, number of discrete cell types.
        emb_dim: Dimension of the state embedding.
        s_condition: Whether to condition the denoiser on interval progress s in [0,1].
        t_condition: Whether to condition the denoiser on (t_start_norm, dt_norm).
        time_mlp_hidden: Hidden size for time-conditioning MLP.
        trunk_hidden: Hidden size for trunk MLP.
        trunk_layers: Number of hidden layers in the trunk MLP (>=1).
        dropout: Dropout probability.
    """
    num_states: int
    emb_dim: int = 64
    s_condition: bool = True
    t_condition: bool = True
    time_mlp_hidden: int = 64
    trunk_hidden: int = 128
    trunk_layers: int = 3
    dropout: float = 0.0


class DFMTimeDenoiser(nn.Module):
    """Global shared DFM denoiser for one-step interval dynamics.

    The model approximates p_theta(x1 | x_s, cond), where:
      - x_s is the current discrete state sampled along the replacement path at some
        normalized interval time s in [0, 1]
      - cond is a configurable set of conditioning features controlled by:
        - cfg.s_condition: include s
        - cfg.t_condition: include (t_start_norm, dt_norm)

    Output: logits over K states (cell types).

Architecture summary:
- State embedding: Embedding(K, emb_dim)
- Optional Time MLP on selected features: 2-layer MLP with SiLU
- Trunk MLP: `trunk_layers` blocks of Linear -> SiLU -> (optional Dropout),
  then final Linear to K logits.


    This aligns with the replacement-path DFM where the sampling update can be written as:
      T_s(z -> x) = (1 - alpha) * 1[x=z] + alpha * softmax(logits_theta(x | z, time_cond))
      with alpha = h * kappa_dot(s) / (1 - kappa(s)), and kappa(s)=s in the simplest case.
    """

    def __init__(self, cfg: DFMModelConfig):
        super().__init__()
        if cfg.num_states <= 1:
            raise ValueError("num_states must be >= 2")
        if cfg.trunk_layers < 1:
            raise ValueError("trunk_layers must be >= 1")

        self.cfg = cfg

        self.state_emb = nn.Embedding(cfg.num_states, cfg.emb_dim)

        time_in_dim = (1 if cfg.s_condition else 0) + (2 if cfg.t_condition else 0)
        if time_in_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_in_dim, cfg.time_mlp_hidden),
                nn.SiLU(),
                nn.Linear(cfg.time_mlp_hidden, cfg.time_mlp_hidden),
                nn.SiLU(),
            )
            time_out_dim = cfg.time_mlp_hidden
        else:
            self.time_mlp = None
            time_out_dim = 0

        layers = []
        in_dim = cfg.emb_dim + time_out_dim
        for li in range(cfg.trunk_layers):
            layers.append(nn.Linear(in_dim if li == 0 else cfg.trunk_hidden, cfg.trunk_hidden))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.trunk_hidden, cfg.num_states))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x_s: Tensor, s: Tensor, t_start: Tensor, dt: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_s: Long tensor of shape [B], values in [0, K-1].
            s: Float tensor of shape [B], normalized time within interval [0,1].
            t_start: Float tensor of shape [B], real start time (will be normalized upstream).
            dt: Float tensor of shape [B], real interval length (will be normalized upstream).

        Returns:
            logits: Float tensor of shape [B, K].
        """
        if x_s.dtype != torch.long:
            raise TypeError(f"x_s must be torch.long, got {x_s.dtype}")

        features = []
        batch = int(x_s.shape[0])
        if self.cfg.s_condition:
            if s.ndim != 1:
                raise ValueError("s must be a 1D tensor of shape [B]")
            if int(s.shape[0]) != batch:
                raise ValueError("Batch sizes must match for x_s and s")
            features.append(s)

        if self.cfg.t_condition:
            if t_start.ndim != 1 or dt.ndim != 1:
                raise ValueError("t_start, dt must be 1D tensors of shape [B]")
            if not (int(t_start.shape[0]) == batch == int(dt.shape[0])):
                raise ValueError("Batch sizes must match for x_s, t_start, dt")
            features.append(t_start)
            features.append(dt)

        x_emb = self.state_emb(x_s)  # [B, emb_dim]

        if len(features) > 0:
            if self.time_mlp is None:
                raise RuntimeError("time_mlp is None but conditioning features were provided.")
            time_feat = torch.stack(features, dim=-1)  # [B, time_in_dim]
            t_emb = self.time_mlp(time_feat)  # [B, time_hidden]
            h = torch.cat([x_emb, t_emb], dim=-1)
        else:
            h = x_emb

        logits = self.trunk(h)
        return logits
