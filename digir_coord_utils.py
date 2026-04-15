"""
Shared coordinate normalization for DIGIR + INTERACTION loaders.

- per_agent: subtract each vehicle's own last observed (x,y) — legacy behavior.
- scene: subtract one origin per batch = last (x,y) of the first vehicle with
  vehicle_masks[b, n] true (never assumes agent index 0 is valid).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

COORD_PER_AGENT = "per_agent"
COORD_SCENE = "scene"


def future_local_from_normed(future_traj_norm: torch.Tensor, trajectories_norm: torch.Tensor) -> torch.Tensor:
    """
    Per-agent relative future: future points minus each agent's last observed (x,y) in the **same
    normalized frame** as future_traj_norm / trajectories_norm.

    Equals global (future - last_pos_global) and matches the diffusion target used in per_agent
    mode. Use this for diffusion + eval when the encoder uses scene- or agent-normalized history.
    """
    last_norm = trajectories_norm[:, :, -1:, :2]
    return future_traj_norm - last_norm


def normalize_batch_for_digir(
    trajectories: torch.Tensor,
    future_traj: torch.Tensor,
    kg_data: Dict[str, torch.Tensor],
    vehicle_masks: torch.Tensor,
    mode: str = COORD_PER_AGENT,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Args:
        trajectories: (B, N, H, 4) global
        future_traj: (B, N, T, 2) global
        kg_data: dict with 'positions' (B,M,2) or (M,2), other keys unchanged
        vehicle_masks: (B, N)
        mode: 'per_agent' | 'scene'

    Returns:
        trajectories_norm, future_traj_norm, kg_data_out (shallow copy, positions may be new tensor),
        ref_point (B, N, 1, 2) — add to model (x,y) outputs to recover **global** coordinates.
    """
    if mode not in (COORD_PER_AGENT, COORD_SCENE):
        raise ValueError(f"mode must be '{COORD_PER_AGENT}' or '{COORD_SCENE}', got {mode!r}")

    device = trajectories.device
    B, N, _, _ = trajectories.shape

    kg_out = dict(kg_data)
    pos = kg_data["positions"]
    if pos.dim() == 2:
        pos_b = pos.unsqueeze(0).expand(B, -1, -1).clone()
    else:
        pos_b = pos.clone()

    if mode == COORD_PER_AGENT:
        ref_point = trajectories[:, :, -1:, :2].clone()
        ref_point = torch.nan_to_num(ref_point, nan=0.0, posinf=0.0, neginf=0.0)
        t_norm = trajectories.clone()
        t_norm[:, :, :, :2] -= ref_point
        f_norm = future_traj - ref_point
        kg_out["positions"] = pos_b
    else:
        # Scene anchor: per batch, first *valid* agent index from mask — not agent 0.
        # If a batch has no valid rows (should not happen with real data), fallback index 0;
        # origin may then be padding zeros (no-op subtract); caller should filter empty batches.
        agent_idx = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            valid_rows = torch.where(vehicle_masks[b].bool())[0]
            if valid_rows.numel() > 0:
                agent_idx[b] = valid_rows[0]
            else:
                agent_idx[b] = 0
        scene_xy = trajectories[torch.arange(B, device=device), agent_idx, -1, :2]
        scene_xy = torch.nan_to_num(scene_xy, nan=0.0, posinf=0.0, neginf=0.0)
        so = scene_xy[:, None, None, :]
        t_norm = trajectories.clone()
        t_norm[:, :, :, :2] -= so
        f_norm = future_traj - so
        pos_b[:, :, :2] = pos_b[:, :, :2] - scene_xy[:, None, :]
        kg_out["positions"] = pos_b
        ref_point = so.expand(B, N, 1, 2).contiguous()
        ref_point = torch.nan_to_num(ref_point, nan=0.0, posinf=0.0, neginf=0.0)

    t_norm = torch.nan_to_num(t_norm, nan=0.0, posinf=0.0, neginf=0.0)
    f_norm = torch.nan_to_num(f_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return t_norm, f_norm, kg_out, ref_point
