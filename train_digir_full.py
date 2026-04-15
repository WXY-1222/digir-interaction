"""
DIGIR Training on INTERACTION with Full Paper Metrics

Metrics:
- minADE_K, minFDE_K: Minimum ADE/FDE over K samples
- Miss Rate (MR): Ratio where minFDE_K > 2.0m
- Intent Accuracy (IA): Classification accuracy
- Intent-Trajectory Consistency (ITC): Alignment between intent and trajectory endpoint
- Collision Rate (CR): Bounding box overlap between agents
- Off-Road Rate (OR): Trajectory violating drivable area
"""
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import random
from collections import defaultdict

sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\DIGIR')

from models.digir import DIGIR
from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn
from digir_coord_utils import COORD_PER_AGENT, COORD_SCENE, future_local_from_normed, normalize_batch_for_digir


class LocationBatchSampler:
    """
    Scheme 2A: group samples by location_name so each batch uses a single map/KG.
    Works with Subset as well as the base dataset.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Resolve location list for indices in this dataset-like object
        locations = None
        if hasattr(dataset, "sample_locations"):
            locations = list(dataset.sample_locations)
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "sample_locations"):
            # torch.utils.data.Subset
            locations = [dataset.dataset.sample_locations[i] for i in dataset.indices]

        groups = defaultdict(list)
        if locations is not None:
            for idx, loc in enumerate(locations):
                groups[loc].append(idx)
        else:
            for idx in range(len(dataset)):
                groups[None].append(idx)

        self.groups = dict(groups)

    def __iter__(self):
        rng = random.Random(self.seed)
        loc_keys = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(loc_keys)

        for loc in loc_keys:
            indices = self.groups[loc][:]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for indices in self.groups.values():
            if self.drop_last:
                n += len(indices) // self.batch_size
            else:
                n += (len(indices) + self.batch_size - 1) // self.batch_size
        return n


def compute_min_ade_fde(pred_trajs, gt_traj):
    """
    Compute minADE_K and minFDE_K
    pred_trajs: (K, B, N, T, 2) or (K, B, T, 2)
    gt_traj: (B, N, T, 2)
    """
    # Handle dimensions
    if pred_trajs.dim() == 4:  # (K, B, T, 2)
        pred_trajs = pred_trajs.unsqueeze(2)  # (K, B, 1, T, 2)

    if gt_traj.dim() == 3:  # (B, T, 2)
        gt_traj = gt_traj.unsqueeze(1)  # (B, 1, T, 2)

    # pred_trajs: (K, B, N, T, 2) -> (B, K, N, T, 2)
    pred_trajs = pred_trajs.permute(1, 0, 2, 3, 4)
    gt_expanded = gt_traj.unsqueeze(1)  # (B, 1, N, T, 2)

    # Compute distances
    distances = torch.norm(pred_trajs - gt_expanded, dim=-1)  # (B, K, N, T)

    # ADE: average over time
    ade_per_sample = torch.mean(distances, dim=-1)  # (B, K, N)

    # FDE: final timestep
    fde_per_sample = distances[:, :, :, -1]  # (B, K, N)

    # Min over K samples
    min_ade = torch.min(ade_per_sample, dim=1)[0]  # (B, N)
    min_fde = torch.min(fde_per_sample, dim=1)[0]  # (B, N)

    return min_ade, min_fde


def compute_intent_trajectory_consistency(pred_trajs, intent_pred):
    """
    Compute Intent-Trajectory Consistency (ITC)
    Measures if trajectory endpoint matches predicted intent

    pred_trajs: (B, N, T, 2) - predicted trajectories
    intent_pred: (B, N) - predicted intent class
    """
    if pred_trajs.dim() == 3:  # (B, T, 2)
        pred_trajs = pred_trajs.unsqueeze(1)

    B, N, T, _ = pred_trajs.shape

    # Get displacement from start to end
    start_pos = pred_trajs[:, :, 0, :]  # (B, N, 2)
    end_pos = pred_trajs[:, :, -1, :]   # (B, N, 2)
    displacement = end_pos - start_pos  # (B, N, 2)

    # Compute heading change
    dx = displacement[:, :, 0]
    dy = displacement[:, :, 1]
    heading_change = torch.atan2(dy, dx) * 180 / np.pi  # degrees

    # Determine actual intent from trajectory
    # -15° ~ 15°: Straight (0)
    # 15° ~ 180°: Left (1)
    # -180° ~ -15°: Right (2)
    actual_intent = torch.zeros_like(intent_pred)
    actual_intent[torch.abs(heading_change) < 15] = 0
    actual_intent[heading_change >= 15] = 1
    actual_intent[heading_change <= -15] = 2
    actual_intent[(torch.abs(heading_change) >= 165) & (torch.abs(heading_change) <= 180)] = 3

    # Consistency: predicted intent matches actual trajectory
    consistent = (intent_pred == actual_intent).float()
    itc = torch.mean(consistent)

    return itc.item()


def compute_collision_rate(pred_trajs, vehicle_masks, vehicle_lengths, vehicle_widths):
    """
    Compute Collision Rate (CR)
    Percentage of trajectories where ego vehicle overlaps with others

    pred_trajs: (B, N, T, 2) - predicted trajectories
    vehicle_masks: (B, N) - valid vehicle mask
    vehicle_lengths: (N,) - vehicle lengths
    vehicle_widths: (N,) - vehicle widths
    """
    if pred_trajs.dim() == 3:
        pred_trajs = pred_trajs.unsqueeze(1)

    B, N, T, _ = pred_trajs.shape

    collision_count = 0
    total_valid = 0

    for b in range(B):
        valid_mask = vehicle_masks[b].bool()
        valid_indices = torch.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid < 2:
            continue

        total_valid += 1

        # Check each timestep
        has_collision = False
        for t in range(T):
            positions = pred_trajs[b, valid_indices, t, :]  # (n_valid, 2)

            # Simple circular collision check (can be improved with bounding boxes)
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    dist = torch.norm(positions[i] - positions[j])
                    # Approximate collision: distance < sum of half-lengths
                    threshold = (vehicle_lengths[valid_indices[i]] + vehicle_lengths[valid_indices[j]]) / 2
                    if dist < threshold:
                        has_collision = True
                        break
                if has_collision:
                    break

            if has_collision:
                break

        if has_collision:
            collision_count += 1

    return collision_count / max(total_valid, 1)


def _min_dist_points_to_segments(points_xy: torch.Tensor, nodes_xy: torch.Tensor, edge_index: torch.Tensor):
    """
    points_xy: (P, 2)
    nodes_xy: (M, 2)
    edge_index: (2, E) indices into nodes
    Returns: (P,) minimum distance to any segment.
    """
    if edge_index is None or edge_index.numel() == 0:
        # fallback: nearest node distance
        d = torch.cdist(points_xy, nodes_xy)
        return d.min(dim=1)[0]

    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {tuple(edge_index.shape)}")

    a = nodes_xy[edge_index[0].long()]  # (E, 2)
    b = nodes_xy[edge_index[1].long()]  # (E, 2)
    ab = b - a  # (E, 2)
    ab2 = (ab * ab).sum(dim=1).clamp_min(1e-8)  # (E,)

    # points (P,1,2), a/b (1,E,2)
    p = points_xy[:, None, :]  # (P,1,2)
    a_ = a[None, :, :]         # (1,E,2)
    ab_ = ab[None, :, :]       # (1,E,2)
    ap = p - a_                # (P,E,2)
    t = (ap * ab_).sum(dim=2) / ab2[None, :]  # (P,E)
    t = t.clamp(0.0, 1.0)
    proj = a_ + t[:, :, None] * ab_  # (P,E,2)
    d = torch.norm(p - proj, dim=2)  # (P,E)
    return d.min(dim=1)[0]           # (P,)


def compute_off_road_rate(
    pred_trajs,
    vehicle_masks,
    map_positions,
    edge_index=None,
    offroad_threshold=2.0,
    debug=False,
):
    """
    Compute Off-Road Rate (OR) using an approximation.

    Since the current dataset doesn't provide explicit drivable polygons, we approximate
    the drivable area as a neighborhood around the map graph nodes given by `map_positions`.

    A scene is counted as off-road if any valid agent's predicted trajectory has any timestep
    whose distance to the nearest map node is > `offroad_threshold`.
    """
    if pred_trajs.dim() == 3:
        pred_trajs = pred_trajs.unsqueeze(1)

    B, N, T, _ = pred_trajs.shape
    # map_positions: (B, M, D) or (M, D)
    if map_positions.dim() == 2:
        map_positions = map_positions.unsqueeze(0).expand(B, -1, -1)

    off_road_count = 0
    total_valid = 0

    for b in range(B):
        valid_mask = vehicle_masks[b].bool()
        if not valid_mask.any():
            continue
        total_valid += 1

        nodes_xy = map_positions[b, :, :2]  # (M, 2)
        positions = pred_trajs[b, valid_mask, :, :]  # (n_valid, T, 2)
        points_xy = positions.reshape(-1, 2)  # (n_valid*T, 2)

        # Min distance to road segments (or fallback to nodes)
        min_dists = _min_dist_points_to_segments(points_xy, nodes_xy, edge_index)  # (n_valid*T,)
        min_dists = min_dists.view(-1, T)  # (n_valid, T)

        has_offroad = (min_dists.max(dim=1)[0] > offroad_threshold).any().item()

        if debug and b == 0:
            # Print distribution once to help pick an appropriate threshold.
            md = min_dists.max(dim=1)[0].flatten().detach().cpu()
            # md: (n_valid,) distance-to-closest-node, taking the worst timestep per vehicle.
            qs = torch.quantile(md, torch.tensor([0.5, 0.9, 0.99, 1.0])).tolist()
            print(
                f"[OffRoad debug] nodes={nodes_xy.shape[0]}, edges={(int(edge_index.shape[1]) if edge_index is not None else 0)}, "
                f"valid_veh={int(valid_mask.sum().item())}, "
                f"threshold={offroad_threshold}, "
                f"veh-worst-dist quantiles: p50={qs[0]:.3f}, p90={qs[1]:.3f}, p99={qs[2]:.3f}, p100={qs[3]:.3f}"
            )
        if has_offroad:
            off_road_count += 1

    return off_road_count / max(total_valid, 1)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    num_samples=5,
    miss_threshold=2.0,
    max_batches=20,
    coord_frame=COORD_PER_AGENT,
    log_gate_stats=False,
):
    """
    Full evaluation with all paper metrics
    """
    model.eval()

    all_min_ade = []
    all_min_fde = []
    all_miss_rates = []
    all_intent_acc = []
    all_itc = []
    all_collision = []
    all_offroad = []
    gate_means = []
    gate_stds = []

    # Per-location aggregations (scheme 2A: each batch has a single location_name)
    per_loc = {}

    batch_count = 0

    def _normalize_pred_shape(pred: torch.Tensor) -> torch.Tensor:
        """
        Normalize output of model.generate to shape (B, N, T, 2).
        Avoids infinite loops if squeeze() does not change tensor rank.
        """
        if pred.dim() == 4:
            return pred

        if pred.dim() == 5:
            # Common cases:
            # (B, 1, N, T, 2) -> (B, N, T, 2)
            if pred.shape[1] == 1:
                return pred[:, 0]
            # (1, B, N, T, 2) -> (B, N, T, 2)
            if pred.shape[0] == 1:
                return pred[0]

        raise RuntimeError(
            f"Unexpected pred shape from model.generate: {tuple(pred.shape)} "
            f"(dim={pred.dim()}). Expected (B,N,T,2) or with a singleton sample dim."
        )

    # max_batches <= 0 means evaluate the full dataloader
    use_max = max_batches if (max_batches is not None and max_batches > 0) else len(dataloader)

    for batch in tqdm(dataloader, desc="Evaluating", total=min(use_max, len(dataloader))):
        if batch_count >= use_max:
            break
        batch_count += 1

        loc_name = None
        if 'location_names' in batch and batch['location_names']:
            # collate_fn guarantees all are the same when using scheme 2A sampler
            loc_name = batch['location_names'][0]
        if loc_name not in per_loc:
            per_loc[loc_name] = {
                'minADE_5': [],
                'minFDE_5': [],
                'MissRate': [],
                'IntentAcc': [],
                'ITC': [],
                'CollisionRate': [],
                'OffRoadRate': [],
            }

        trajectories = batch['trajectories'].to(device)  # (B, N, 8, 4)
        future_traj = batch['future_trajectory'].to(device)  # (B, N, 12, 2)
        intent_labels = batch['intent_labels'].to(device)  # (B, N)
        vehicle_masks = batch['vehicle_masks'].to(device)  # (B, N)

        kg_data = {
            'facility_types': batch['kg_data']['facility_types'].to(device),
            'positions': batch['kg_data']['positions'].to(device),
            'edge_index': batch['kg_data']['edge_index'].to(device),
            'edge_types': batch['kg_data']['edge_types'].to(device),
        }
        Bsz = trajectories.shape[0]
        kg_positions_global = kg_data["positions"].clone()
        if kg_positions_global.dim() == 2:
            kg_positions_global = kg_positions_global.unsqueeze(0).expand(Bsz, -1, -1)

        trajectories_norm, future_traj_norm, kg_data, _ = normalize_batch_for_digir(
            trajectories, future_traj, kg_data, vehicle_masks, mode=coord_frame
        )
        # Diffusion / generate 目标为「相对当前观测末帧的位移」，与 per_agent 尺度一致；地图仍用 scene 对齐。
        last_pos_global = torch.nan_to_num(
            trajectories[:, :, -1:, :2].clone(), nan=0.0, posinf=0.0, neginf=0.0
        )
        future_local = torch.nan_to_num(
            future_local_from_normed(future_traj_norm, trajectories_norm),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Default vehicle dimensions per-agent (N can be > 5).
        # If you have per-agent type in data, replace these with a proper lookup.
        N = trajectories.shape[1]
        vehicle_lengths = torch.full((N,), 4.5, device=device)  # meters
        vehicle_widths = torch.full((N,), 1.8, device=device)   # meters

        # ===== 1. Generate K samples for minADE/minFDE =====
        pred_trajs_k = []
        for _ in range(num_samples):
            pred = model.generate(
                trajectories_norm, kg_data,
                num_points=12,
                num_samples=1,
                sampling="ddim",
                step=10,
                bestof=False
            )
            pred_trajs_k.append(_normalize_pred_shape(pred))

        pred_trajs_k = torch.stack(pred_trajs_k, dim=0)  # (K, B, N, T, 2) or similar
        pred_trajs_k = torch.nan_to_num(pred_trajs_k, nan=0.0, posinf=0.0, neginf=0.0)

        # Compare pred vs GT in the same frame as training (local / displacement).
        min_ade, min_fde = compute_min_ade_fde(pred_trajs_k, future_local)

        # Apply mask
        valid_mask = vehicle_masks.bool()
        if valid_mask.any():
            min_ade_valid = min_ade[valid_mask]
            min_fde_valid = min_fde[valid_mask]

            ade_v = min_ade_valid.mean().item()
            fde_v = min_fde_valid.mean().item()
            all_min_ade.append(ade_v)
            all_min_fde.append(fde_v)
            per_loc[loc_name]['minADE_5'].append(ade_v)
            per_loc[loc_name]['minFDE_5'].append(fde_v)

            # Miss Rate
            miss_rate = (min_fde_valid > miss_threshold).float().mean().item()
            all_miss_rates.append(miss_rate)
            per_loc[loc_name]['MissRate'].append(miss_rate)

        # ===== 2. Intent Accuracy =====
        outputs = model(trajectories_norm, kg_data, mode='eval')
        intent_logits = outputs['intent_logits']  # (B, N, num_classes)
        intent_pred = intent_logits.argmax(dim=-1)  # (B, N)
        if log_gate_stats and 'gate_weights' in outputs:
            gw = outputs['gate_weights']
            if gw is not None and torch.is_tensor(gw):
                gw = torch.nan_to_num(gw, nan=0.0, posinf=0.0, neginf=0.0).float()
                gate_means.append(float(gw.mean().item()))
                gate_stds.append(float(gw.std().item()))

        valid_intent = (intent_labels >= 0) & valid_mask
        if valid_intent.any():
            intent_acc = ((intent_pred == intent_labels) & valid_intent).float().sum() / valid_intent.sum()
            all_intent_acc.append(intent_acc.item())
            per_loc[loc_name]['IntentAcc'].append(intent_acc.item())

        # ===== 3. Intent-Trajectory Consistency =====
        best_pred = pred_trajs_k[0]
        itc = compute_intent_trajectory_consistency(best_pred, intent_pred)
        all_itc.append(itc)
        per_loc[loc_name]['ITC'].append(itc)

        # ===== 4. Collision Rate =====
        # Predictions are local (future relative to last obs); add global last (x,y) per agent.
        best_pred_global = best_pred + last_pos_global
        cr = compute_collision_rate(best_pred_global, vehicle_masks, vehicle_lengths, vehicle_widths)
        all_collision.append(cr)
        per_loc[loc_name]['CollisionRate'].append(cr)

        # ===== 5. Off-Road Rate =====
        # Calibrate off-road threshold (meters). With the current "nodes as drivable neighborhood"
        # approximation, 2m is too strict (would lead to ~100% off-road).
        or_rate = compute_off_road_rate(
            best_pred_global,
            vehicle_masks,
            kg_positions_global,
            edge_index=kg_data['edge_index'],
            offroad_threshold=3.0,
            debug=False
        )
        all_offroad.append(or_rate)
        per_loc[loc_name]['OffRoadRate'].append(or_rate)

    # Aggregate metrics
    metrics = {
        'minADE_5': np.mean(all_min_ade) if all_min_ade else 0,
        'minFDE_5': np.mean(all_min_fde) if all_min_fde else 0,
        'MissRate': np.mean(all_miss_rates) if all_miss_rates else 0,
        'IntentAcc': np.mean(all_intent_acc) if all_intent_acc else 0,
        'ITC': np.mean(all_itc) if all_itc else 0,
        'CollisionRate': np.mean(all_collision) if all_collision else 0,
        'OffRoadRate': np.mean(all_offroad) if all_offroad else 0,
    }
    if log_gate_stats:
        metrics['GateMean'] = float(np.mean(gate_means)) if gate_means else 0.0
        metrics['GateStd'] = float(np.mean(gate_stds)) if gate_stds else 0.0

    per_location_metrics = {}
    for loc, vals in per_loc.items():
        per_location_metrics[loc] = {
            'minADE_5': float(np.mean(vals['minADE_5'])) if vals['minADE_5'] else 0.0,
            'minFDE_5': float(np.mean(vals['minFDE_5'])) if vals['minFDE_5'] else 0.0,
            'MissRate': float(np.mean(vals['MissRate'])) if vals['MissRate'] else 0.0,
            'IntentAcc': float(np.mean(vals['IntentAcc'])) if vals['IntentAcc'] else 0.0,
            'ITC': float(np.mean(vals['ITC'])) if vals['ITC'] else 0.0,
            'CollisionRate': float(np.mean(vals['CollisionRate'])) if vals['CollisionRate'] else 0.0,
            'OffRoadRate': float(np.mean(vals['OffRoadRate'])) if vals['OffRoadRate'] else 0.0,
            'batches': len(vals['ITC']),
        }

    metrics['per_location'] = per_location_metrics

    return metrics


def train_epoch(model, dataloader, optimizer, device, coord_frame=COORD_PER_AGENT, log_gate_stats=False):
    """Training epoch"""
    model.train()
    total_loss = 0
    effective_batches = 0
    printed_nan = 0
    gate_means = []
    gate_stds = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        trajectories = batch['trajectories'].to(device)
        future_traj = batch['future_trajectory'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        vehicle_masks = batch['vehicle_masks'].to(device)

        kg_data = {
            'facility_types': batch['kg_data']['facility_types'].to(device),
            'positions': batch['kg_data']['positions'].to(device),
            'edge_index': batch['kg_data']['edge_index'].to(device),
            'edge_types': batch['kg_data']['edge_types'].to(device),
        }
        Bsz = trajectories.shape[0]
        kg_positions_global = kg_data["positions"].clone()
        if kg_positions_global.dim() == 2:
            kg_positions_global = kg_positions_global.unsqueeze(0).expand(Bsz, -1, -1)

        trajectories_norm, future_traj_norm, kg_data, _ = normalize_batch_for_digir(
            trajectories, future_traj, kg_data, vehicle_masks, mode=coord_frame
        )
        if printed_nan < 1 and (not torch.isfinite(trajectories_norm).all() or not torch.isfinite(future_traj_norm).all()):
            printed_nan += 1
            print("[NaN/Inf] found in input normalization; applying nan_to_num")

        last_pos_global = torch.nan_to_num(
            trajectories[:, :, -1:, :2].clone(), nan=0.0, posinf=0.0, neginf=0.0
        )
        future_local = torch.nan_to_num(
            future_local_from_normed(future_traj_norm, trajectories_norm),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        optimizer.zero_grad()
        outputs = model(
            trajectories_norm,
            kg_data,
            future_traj=future_local,
            mode='train',
            vehicle_masks=vehicle_masks,
        )
        if log_gate_stats and 'gate_weights' in outputs:
            gw = outputs['gate_weights']
            if gw is not None and torch.is_tensor(gw):
                gw = torch.nan_to_num(gw, nan=0.0, posinf=0.0, neginf=0.0).float()
                gate_means.append(float(gw.mean().item()))
                gate_stds.append(float(gw.std().item()))
        # Rule losses: Y_pred + ref_point → global; ref_point must be per-agent last xy in **global** meters.
        outputs['kg_positions'] = kg_positions_global
        outputs['kg_edge_index'] = kg_data['edge_index']
        outputs['ref_point'] = last_pos_global
        diff_loss = outputs.get('diffusion_loss', None)
        intent_logits = outputs.get('intent_logits', None)
        if diff_loss is not None and (not torch.isfinite(diff_loss).item()):
            if printed_nan < 3:
                printed_nan += 1
                print("[NaN/Inf] diffusion_loss is not finite; skipping batch")
            optimizer.zero_grad()
            continue
        if intent_logits is not None and (not torch.isfinite(intent_logits).all().item()):
            if printed_nan < 3:
                printed_nan += 1
                print("[NaN/Inf] intent_logits contains NaN/Inf; skipping batch")
            optimizer.zero_grad()
            continue

        losses, loss = model.compute_losses(outputs, future_local, intent_labels, vehicle_masks)
        if not torch.isfinite(loss).item():
            if printed_nan < 10:
                printed_nan += 1
                # losses 中有些是 python float，有些是 tensor；这里只做可读性输出
                safe_losses = {}
                for k, v in losses.items():
                    try:
                        safe_losses[k] = float(v) if isinstance(v, (int, float)) else (v.item() if torch.is_tensor(v) else v)
                    except Exception:
                        safe_losses[k] = str(v)
                print("[NaN/Inf] total loss is not finite; losses=", safe_losses)
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        # Show key loss components when available
        postfix = {'loss': f"{loss.item():.4f}"}
        if isinstance(losses, dict) and 'loss_rule' in losses:
            postfix.update({
                'diff': f"{losses.get('loss_diff', 0.0):.3f}",
                'rule': f"{losses.get('loss_rule', 0.0):.1f}",
                'col': f"{losses.get('loss_col', 0.0):.2f}",
                'map': f"{losses.get('loss_map', 0.0):.1f}",
                'λr': f"{losses.get('lambda_rule', 0.0):.0e}",
            })
        if log_gate_stats and gate_means:
            postfix['gμ'] = f"{gate_means[-1]:.2f}"
        pbar.set_postfix(postfix)
        effective_batches += 1

    gate_stats = None
    if log_gate_stats:
        gate_stats = {
            'GateMean': float(np.mean(gate_means)) if gate_means else 0.0,
            'GateStd': float(np.mean(gate_stds)) if gate_stds else 0.0,
        }
    return total_loss / max(effective_batches, 1), gate_stats


def main():
    parser = argparse.ArgumentParser(description="Train DIGIR (scheme 2A multi-map ready)")
    parser.add_argument("--data", type=str, default="./digir_data/interaction_digir.pkl")
    parser.add_argument("--save", type=str, default="./digir_interaction_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_by_location", action="store_true", help="Scheme 2A: group batches by location_name")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_subset", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_rule", type=float, default=1e-3, help="Weight for (L_col + L_map). Set 0 for baseline.")
    parser.add_argument("--map_margin", type=float, default=3.0, help="Meters. L_map penalizes distance beyond this.")
    parser.add_argument(
        "--ablate_cross_attn",
        action="store_true",
        help="Ablation: disable trajectory<->map cross-attention (local_context <- motion_summaries).",
    )
    parser.add_argument(
        "--ablate_gate",
        type=str,
        default="none",
        choices=["none", "fixed_half", "force_intent", "force_interaction"],
        help="Gate ablation: none=normal learned gate; fixed_half=0.5; force_intent=gate=0; force_interaction=gate=1.",
    )
    parser.add_argument(
        "--log_gate_stats",
        action="store_true",
        help="Print gate mean/std during train and eval for interpretability.",
    )
    parser.add_argument(
        "--gate_fixed_ratio",
        type=float,
        default=None,
        help="Optional continuous gate override in [0,1]: fused = r*interaction + (1-r)*intent. "
        "If set, this overrides --ablate_gate.",
    )
    parser.add_argument(
        "--coord_frame",
        type=str,
        default=COORD_PER_AGENT,
        choices=[COORD_PER_AGENT, COORD_SCENE],
        help="per_agent: each vehicle centered at its last obs (default). "
        "scene: one origin per batch (first valid vehicle); KG shifted to match trajectories.",
    )
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(args.save))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        'd_model': 128,
        'd_prior': 128,
        'hist_len': 8,
        'prediction_horizon': 12,
        'num_intent_classes': 4,
        'num_facility_types': 10,
        'traj_enc_layers': 3,
        'graph_enc_layers': 3,
        'scene_tf_layers': 3,
        'v2v_layers': 3,
        'diffusion_tf_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'elementwise_gate': True,
        'diffusion_steps': 50,
        'beta_1': 1e-4,
        'beta_T': 5e-2,
        'lambda_fine': 1.0,
        'lambda_coarse': 0.5,
        'lambda_cross': 0.1,
        # Rule loss weight (L_col + L_map). Distances are in meters.
        'lambda_rule': float(args.lambda_rule),
        # Map margin (meters) for L_map (distance to road segment beyond this is penalized)
        'map_margin': float(args.map_margin),
        'coord_frame': str(args.coord_frame),
        'ablate_cross_attn': bool(args.ablate_cross_attn),
        'ablate_gate': str(args.ablate_gate),
        'gate_fixed_ratio': (None if args.gate_fixed_ratio is None else float(args.gate_fixed_ratio)),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model config: d_model={config['d_model']}, diffusion_steps={config['diffusion_steps']}")
    print(f"Coordinate frame: {args.coord_frame}")
    print(f"Ablate cross-attn: {args.ablate_cross_attn}")
    print(f"Ablate gate: {args.ablate_gate}")
    if args.gate_fixed_ratio is not None:
        r = float(max(0.0, min(1.0, float(args.gate_fixed_ratio))))
        print(f"Gate fixed ratio: {r:.3f} (interaction={r:.3f}, intent={1.0-r:.3f})")

    # Load data
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    train_dataset = InteractionDatasetForDIGIR(data_path, split='train', max_vehicles=10)
    val_dataset = InteractionDatasetForDIGIR(data_path, split='val', max_vehicles=10)

    # Use subset for faster training
    train_subset = torch.utils.data.Subset(train_dataset, range(min(args.train_subset, len(train_dataset))))

    if args.batch_by_location:
        train_loader = DataLoader(
            train_subset,
            batch_sampler=LocationBatchSampler(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False),
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.batch_by_location:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=LocationBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False),
            collate_fn=collate_fn,
        )
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_subset)}, Val: {len(val_dataset)}")

    # Create model
    model = DIGIR(config).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training
    num_epochs = args.epochs
    best_ade = float('inf')

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_gate_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            coord_frame=args.coord_frame,
            log_gate_stats=args.log_gate_stats,
        )
        print(f"Train Loss: {train_loss:.4f}")
        if args.log_gate_stats and train_gate_stats is not None:
            print(f"  Gate(train): mean={train_gate_stats['GateMean']:.4f}, std={train_gate_stats['GateStd']:.4f}")

        # Evaluate
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_samples=args.k,
            max_batches=args.eval_batches,
            coord_frame=args.coord_frame,
            log_gate_stats=args.log_gate_stats,
        )

        print("\nMetrics:")
        print(f"  Kinematic:")
        print(f"    minADE_5:  {metrics['minADE_5']:.3f} m")
        print(f"    minFDE_5:  {metrics['minFDE_5']:.3f} m")
        print(f"    MissRate:  {metrics['MissRate']:.2%}")
        print(f"  Semantic:")
        print(f"    IntentAcc: {metrics['IntentAcc']:.2%}")
        print(f"    ITC:       {metrics['ITC']:.2%}")
        print(f"  Safety:")
        print(f"    Collision: {metrics['CollisionRate']:.2%}")
        print(f"    Off-Road:  {metrics['OffRoadRate']:.2%}")
        if args.log_gate_stats:
            print(f"  Gate:")
            print(f"    Mean:      {metrics.get('GateMean', 0.0):.4f}")
            print(f"    Std:       {metrics.get('GateStd', 0.0):.4f}")

        # Per-location summary (only when location info is present)
        per_loc = metrics.get('per_location', {}) or {}
        if per_loc:
            print("\nPer-location (avg over evaluated batches):")
            for loc_name in sorted(per_loc.keys(), key=lambda x: str(x)):
                m = per_loc[loc_name]
                print(
                    f"  {loc_name} | batches={m['batches']}"
                    f" | minADE_5={m['minADE_5']:.3f}"
                    f" | minFDE_5={m['minFDE_5']:.3f}"
                    f" | MR={m['MissRate']:.2%}"
                    f" | IA={m['IntentAcc']:.2%}"
                    f" | ITC={m['ITC']:.2%}"
                    f" | Col={m['CollisionRate']:.2%}"
                    f" | OR={m['OffRoadRate']:.2%}"
                )

        scheduler.step()

        # Save best model
        if metrics['minADE_5'] < best_ade:
            best_ade = metrics['minADE_5']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'config': config,
                'metrics': metrics,
            }, args.save)
            print(f"  [*] Best model saved (minADE: {best_ade:.3f})")

    print("\n" + "="*70)
    print("Training Completed!")
    print(f"Best minADE_5: {best_ade:.3f} m")
    print("="*70)


if __name__ == "__main__":
    main()
