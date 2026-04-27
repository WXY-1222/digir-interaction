"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with
Homotopy-Corridor Builder + Selective State-Space Routing Decoder.

Pipeline:
    history + KG -> dual-granularity encoder -> Goal Decoder
    -> Homotopy/Corridor Builder -> Mamba-style Routing Decoder
    -> multi-modal trajectories.

This file does NOT overwrite existing model scripts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir_goal_cascade_homotopy_bifurcation import DIGIR as HomotopyDIGIR
from models.digir_goal_cascade_mamba_decoder import SelectiveSSMBlock


class DIGIR(HomotopyDIGIR):
    """
    Keep the strongest homotopy-corridor structure, but replace the final
    Transformer routing decoder with a linear-time selective SSM decoder.
    """

    def __init__(self, config):
        super().__init__(config)

        d = self.d_model
        dropout = float(config.get("dropout", 0.1))
        route_layers = int(config.get("mamba_route_layers", 4))
        expansion = int(config.get("mamba_expansion", 2))

        # Replace the Transformer route decoder created by HomotopyDIGIR.
        self.route_decoder = nn.ModuleList(
            [SelectiveSSMBlock(d, expansion=expansion, dropout=dropout) for _ in range(route_layers)]
        )
        self.route_traj_head = nn.Identity()

        self.route_seed_proj = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.step_delta_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 2),
        )

    def _route_decode_homotopy(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_h,
        goal_xy,
        goal_logits,
        kg_positions,
        kg_edge_index,
        num_points,
    ):
        """
        Stage 2 routing with homotopy corridors and Mamba-style SSM evolution.

        Returns are interface-compatible with the homotopy Transformer variant.
        """
        b, n, d = fused_conditions.shape
        q = goal_xy.shape[2]
        h_cls = self.homotopy_classes
        k_target = self.route_modes
        t = int(num_points)

        goal_slots = max(1, min(q, int(math.ceil(float(k_target) / float(h_cls)))))

        top = goal_logits.topk(goal_slots, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)

        corridor_feat, path_score = self._compute_homotopy_corridors(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
        )

        # Expand (goal, homotopy) combinations.
        g = goal_slots
        h = h_cls
        c = g * h

        mode_goals_exp = top_goals.unsqueeze(3).expand(b, n, g, h, 2).reshape(b, n, c, 2)
        mode_corr_exp = corridor_feat.reshape(b, n, c, d)
        goal_logits_exp = top_goal_logits.unsqueeze(3).expand(b, n, g, h).reshape(b, n, c)
        path_score_exp = path_score.reshape(b, n, c)

        hid = torch.arange(h, device=top_goals.device, dtype=torch.long).view(1, 1, 1, h).expand(b, n, g, h)
        gid = torch.arange(g, device=top_goals.device, dtype=torch.long).view(1, 1, g, 1).expand(b, n, g, h)
        mode_hid_exp = hid.reshape(b, n, c)
        mode_gid_exp = gid.reshape(b, n, c)

        (
            sel_goal_logits,
            sel_path_score,
            mode_goals,
            mode_corr,
            mode_hid,
            mode_gid,
        ) = self._select_mode_combinations(
            goal_logits_exp=goal_logits_exp,
            path_score_exp=path_score_exp,
            mode_goals_exp=mode_goals_exp,
            mode_corr_exp=mode_corr_exp,
            mode_hid_exp=mode_hid_exp,
            mode_gid_exp=mode_gid_exp,
            k_target=k_target,
        )

        # Seed each mode with goal, homotopy corridor, interaction and intent context.
        route_in = torch.cat(
            [fused_conditions.unsqueeze(2).expand(-1, -1, k_target, -1), mode_goals],
            dim=-1,
        )
        route_goal = self.goal_to_route(route_in)
        route_goal = route_goal + self.homotopy_embed(mode_hid)
        route_corr = self.corridor_proj(mode_corr)

        route_base = self.route_seed_proj(
            torch.cat(
                [
                    route_goal,
                    route_corr,
                    intent_priors.unsqueeze(2).expand(-1, -1, k_target, -1),
                    interaction_features.unsqueeze(2).expand(-1, -1, k_target, -1)
                    + local_contexts.unsqueeze(2).expand(-1, -1, k_target, -1),
                ],
                dim=-1,
            )
        )

        time = torch.linspace(0.0, 1.0, steps=t, device=route_base.device, dtype=route_base.dtype)
        time_feat = self.time_mlp(time.view(t, 1)).view(1, 1, 1, t, d)
        seq = route_base.unsqueeze(3) + time_feat
        seq = seq.reshape(b * n * k_target, t, d)

        for block in self.route_decoder:
            seq = block(seq)

        step_delta = self.step_delta_head(seq).view(b, n, k_target, t, 2)
        traj_modes = torch.cumsum(step_delta, dim=-2)

        # Endpoint alignment keeps the SSM evolution anchored to the selected goal.
        goal_expand = mode_goals.unsqueeze(-2)
        alpha = torch.linspace(
            0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, t, 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        route_h_view = seq[:, -1].view(b, n, k_target, d)
        route_logits = self.route_score_head(route_h_view).squeeze(-1)
        corridor_score = F.cosine_similarity(route_h_view, mode_corr, dim=-1)
        mode_logits = (
            sel_goal_logits
            + route_logits
            + self.homotopy_score_weight * sel_path_score
            + 0.2 * corridor_score
        )

        return traj_modes, mode_logits, mode_goals, mode_corr, route_h_view, mode_hid, mode_gid

