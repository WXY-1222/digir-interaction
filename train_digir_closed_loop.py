"""
Closed-loop (receding-horizon) training script for DIGIR.

Goal:
- Mimic the paper's closed-loop rollout idea during training:
  maintain a sliding observation window of length H (hist_len),
  repeatedly (1) train on the current window, then (2) advance the window
  using the model's own 1-step prediction (scheduled sampling).

Important:
- This script does NOT modify existing code; it is a standalone trainer.
- The model's diffusion training objective remains the same (differentiable),
  but the *inputs* are progressively replaced by model predictions to reduce
  exposure bias (covariate shift).

Usage example:
python train_digir_closed_loop.py ^
  --data .\\digir_data\\interaction_digir_all_12loc_h8_f12.pkl ^
  --save .\\best_closed_loop.pt ^
  --batch_by_location ^
  --epochs 20 ^
  --batch_size 8 ^
  --lr 1e-4 ^
  --rollout_steps 12 ^
  --tf_schedule linear --tf_start 0.9 --tf_end 0.2 ^
  --one_step_k 5 --fixed_noise_seed 12345

Teacher forcing: use --tf_schedule constant --teacher_forcing 0.2 to match the old fixed-TF behavior.
"""

import os
import sys
import math
import argparse
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Default DIGIR root: <interaction>/digir (or DIGIR_ROOT env override).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIGIR_ROOT = os.environ.get("DIGIR_ROOT", os.path.join(PROJECT_ROOT, "digir"))
if not os.path.exists(DEFAULT_DIGIR_ROOT):
    raise FileNotFoundError(
        f"DIGIR root not found: {DEFAULT_DIGIR_ROOT}. "
        "Put digir.py at <interaction>/digir/models/digir.py or set DIGIR_ROOT."
    )
if DEFAULT_DIGIR_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_DIGIR_ROOT)

from models.digir import DIGIR  # noqa: E402
from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn  # noqa: E402
from digir_coord_utils import COORD_PER_AGENT, COORD_SCENE, future_local_from_normed, normalize_batch_for_digir  # noqa: E402


class LocationBatchSampler:
    """Group indices by location_name so each batch uses a single KG."""

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        locations = None
        if hasattr(dataset, "sample_locations"):
            locations = list(dataset.sample_locations)
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "sample_locations"):
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
        keys = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(keys)
        for loc in keys:
            idxs = self.groups[loc][:]
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for idxs in self.groups.values():
            if self.drop_last:
                n += len(idxs) // self.batch_size
            else:
                n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n


def compute_min_ade_fde(pred_trajs, gt_traj):
    """
    pred_trajs: (K, B, N, T, 2) or (K, B, T, 2)
    gt_traj: (B, N, T, 2)
    """
    if pred_trajs.dim() == 4:  # (K,B,T,2)
        pred_trajs = pred_trajs.unsqueeze(2)  # (K,B,1,T,2)
    if gt_traj.dim() == 3:  # (B,T,2)
        gt_traj = gt_traj.unsqueeze(1)
    pred_trajs = pred_trajs.permute(1, 0, 2, 3, 4)  # (B,K,N,T,2)
    gt_expanded = gt_traj.unsqueeze(1)  # (B,1,N,T,2)
    distances = torch.norm(pred_trajs - gt_expanded, dim=-1)  # (B,K,N,T)
    ade_per_sample = torch.mean(distances, dim=-1)  # (B,K,N)
    fde_per_sample = distances[:, :, :, -1]  # (B,K,N)
    min_ade = torch.min(ade_per_sample, dim=1)[0]  # (B,N)
    min_fde = torch.min(fde_per_sample, dim=1)[0]  # (B,N)
    return min_ade, min_fde


@torch.no_grad()
def evaluate_open_loop(
    model,
    dataloader,
    device,
    num_samples=5,
    miss_threshold=2.0,
    max_batches=0,
    coord_frame: str = COORD_PER_AGENT,
):
    """
    Open-loop evaluation (same style as train_digir_full.py):
    use 8-step history to generate 12-step future, compute metrics.
    """
    model.eval()
    use_max = max_batches if (max_batches is not None and max_batches > 0) else len(dataloader)

    all_min_ade = []
    all_min_fde = []
    all_miss_rates = []
    all_intent_acc = []
    per_loc = {}

    batch_count = 0
    for batch in tqdm(dataloader, desc="Evaluating", total=min(use_max, len(dataloader))):
        if batch_count >= use_max:
            break
        batch_count += 1

        loc_name = None
        if "location_names" in batch and batch["location_names"]:
            loc_name = batch["location_names"][0]
        if loc_name not in per_loc:
            per_loc[loc_name] = {"minADE_5": [], "minFDE_5": [], "MissRate": [], "IntentAcc": [], "batches": 0}

        trajectories = batch["trajectories"].to(device)  # (B,N,H,4)
        future_traj = batch["future_trajectory"].to(device)  # (B,N,T,2)
        intent_labels = batch["intent_labels"].to(device)
        vehicle_masks = batch["vehicle_masks"].to(device)

        kg_data = {
            "facility_types": batch["kg_data"]["facility_types"].to(device),
            "positions": batch["kg_data"]["positions"].to(device),
            "edge_index": batch["kg_data"]["edge_index"].to(device),
            "edge_types": batch["kg_data"]["edge_types"].to(device),
        }

        trajectories_norm, future_traj_norm, kg_data, _ = normalize_batch_for_digir(
            trajectories, future_traj, kg_data, vehicle_masks, mode=coord_frame
        )
        future_local = torch.nan_to_num(
            future_local_from_normed(future_traj_norm, trajectories_norm),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # K samples
        pred_list = []
        for _ in range(num_samples):
            pred = model.generate(
                trajectories_norm,
                kg_data,
                num_points=future_traj.shape[2],
                num_samples=1,
                sampling="ddim",
                step=10,
                bestof=False,
            )  # (1,B,N,T,2)
            pred = pred[0]  # (B,N,T,2)
            pred_list.append(pred)
        pred_k = torch.stack(pred_list, dim=0)  # (K,B,N,T,2)

        min_ade, min_fde = compute_min_ade_fde(pred_k, future_local)
        valid_mask = vehicle_masks.bool()
        if valid_mask.any():
            ade_v = min_ade[valid_mask].mean().item()
            fde_v = min_fde[valid_mask].mean().item()
            miss_rate = (min_fde[valid_mask] > miss_threshold).float().mean().item()
            all_min_ade.append(ade_v)
            all_min_fde.append(fde_v)
            all_miss_rates.append(miss_rate)
            per_loc[loc_name]["minADE_5"].append(ade_v)
            per_loc[loc_name]["minFDE_5"].append(fde_v)
            per_loc[loc_name]["MissRate"].append(miss_rate)

        outputs = model(trajectories_norm, kg_data, mode="eval")
        intent_logits = outputs["intent_logits"]
        intent_pred = intent_logits.argmax(dim=-1)
        valid_intent = (intent_labels >= 0) & valid_mask
        if valid_intent.any():
            intent_acc = ((intent_pred == intent_labels) & valid_intent).float().sum() / valid_intent.sum()
            all_intent_acc.append(intent_acc.item())
            per_loc[loc_name]["IntentAcc"].append(intent_acc.item())

        per_loc[loc_name]["batches"] += 1

    metrics = {
        "minADE_5": float(np.mean(all_min_ade)) if all_min_ade else 0.0,
        "minFDE_5": float(np.mean(all_min_fde)) if all_min_fde else 0.0,
        "MissRate": float(np.mean(all_miss_rates)) if all_miss_rates else 0.0,
        "IntentAcc": float(np.mean(all_intent_acc)) if all_intent_acc else 0.0,
    }
    metrics["per_location"] = {}
    for loc, vals in per_loc.items():
        metrics["per_location"][loc] = {
            "batches": vals["batches"],
            "minADE_5": float(np.mean(vals["minADE_5"])) if vals["minADE_5"] else 0.0,
            "minFDE_5": float(np.mean(vals["minFDE_5"])) if vals["minFDE_5"] else 0.0,
            "MissRate": float(np.mean(vals["MissRate"])) if vals["MissRate"] else 0.0,
            "IntentAcc": float(np.mean(vals["IntentAcc"])) if vals["IntentAcc"] else 0.0,
        }
    return metrics


def _infer_heading_speed(prev_xy: torch.Tensor, cur_xy: torch.Tensor, dt: float = 1.0):
    """
    prev_xy, cur_xy: (..., 2)
    Returns heading (rad), speed (m/s) with same leading shape.
    """
    vel = (cur_xy - prev_xy) / dt
    speed = torch.norm(vel, dim=-1)
    heading = torch.atan2(vel[..., 1], vel[..., 0])
    return heading, speed


def _teacher_forcing_at_epoch(schedule: str, epoch: int, epochs: int, tf_start: float, tf_end: float, tf_constant: float) -> float:
    """Returns teacher-forcing probability in [0, 1] for this epoch (1-based epoch index)."""
    if schedule == "constant":
        return float(max(0.0, min(1.0, tf_constant)))
    if epochs <= 1:
        t = 0.0
    else:
        t = (epoch - 1) / (epochs - 1)
    if schedule == "linear":
        v = tf_start + (tf_end - tf_start) * t
    elif schedule == "cosine":
        v = tf_end + (tf_start - tf_end) * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        raise ValueError(f"Unknown tf_schedule: {schedule}")
    return float(max(0.0, min(1.0, v)))


@torch.no_grad()
def predict_one_step_global(
    model: DIGIR,
    obs_window: torch.Tensor,
    kg_data: dict,
    step: int = 10,
    num_candidates: int = 1,
    gt_next_one: Optional[torch.Tensor] = None,
    vehicle_masks: Optional[torch.Tensor] = None,
    fixed_noise_seed: Optional[int] = None,
    scene_origin: Optional[torch.Tensor] = None,
):
    """
    obs_window: (B, N, H, 4) in global coords.
    Returns next_xy: (B, N, 2) in global coords.

    num_candidates: if >1, draw K diffusion samples and pick the one closest to gt_next_one (global),
        when gt_next_one is provided; otherwise use the first sample.
    fixed_noise_seed: if set, temporarily fixes torch RNG so the same obs yields reproducible noise
        (restores RNG state after the call).
    """
    if scene_origin is not None:
        ref_point = scene_origin
    else:
        ref_point = obs_window[:, :, -1:, :2].clone()
    obs_norm = obs_window.clone()
    obs_norm[:, :, :, :2] -= ref_point
    obs_norm = torch.nan_to_num(obs_norm, nan=0.0, posinf=0.0, neginf=0.0)

    k = max(1, int(num_candidates))
    rng_cpu = None
    rng_cuda = None
    if fixed_noise_seed is not None:
        rng_cpu = torch.get_rng_state()
        if torch.cuda.is_available() and obs_norm.is_cuda:
            rng_cuda = torch.cuda.get_rng_state_all()
        torch.manual_seed(int(fixed_noise_seed))
        if torch.cuda.is_available() and obs_norm.is_cuda:
            torch.cuda.manual_seed_all(int(fixed_noise_seed))

    try:
        pred = model.generate(
            obs_norm,
            kg_data,
            num_points=1,
            num_samples=k,
            sampling="ddim",
            step=step,
            bestof=False,
        )  # (K, B, N, 1, 2)
    finally:
        if fixed_noise_seed is not None:
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)

    if pred.dim() != 5:
        raise RuntimeError(f"Unexpected pred shape from generate(num_points=1): {tuple(pred.shape)}")

    if k > 1 and gt_next_one is not None:
        pred_global = pred + ref_point.unsqueeze(0)  # (K,B,N,1,2)
        gt_exp = gt_next_one.unsqueeze(0).unsqueeze(3)  # (1,B,N,1,2)
        dist = torch.norm(pred_global - gt_exp, dim=-1).squeeze(-1)  # (K,B,N)
        if vehicle_masks is not None:
            vm = vehicle_masks.unsqueeze(0).bool()
            dist = dist.masked_fill(~vm, float("inf"))
        best_k = dist.argmin(dim=0)  # (B,N)
        b, n = best_k.shape
        gather_idx = best_k.long().view(1, b, n, 1, 1).expand(1, b, n, 1, 2)
        pred = pred.gather(0, gather_idx).squeeze(0)  # (B,N,1,2)
    else:
        pred = pred[0]  # (B, N, 1, 2)

    next_xy = pred[:, :, 0, :] + ref_point[:, :, 0, :]
    next_xy = torch.nan_to_num(next_xy, nan=0.0, posinf=0.0, neginf=0.0)
    return next_xy


def closed_loop_train_epoch(
    model: DIGIR,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    rollout_steps: int,
    teacher_forcing: float,
    generate_step: int,
    one_step_k: int = 1,
    fixed_noise_seed: Optional[int] = None,
    coord_frame: str = COORD_PER_AGENT,
):
    """
    Closed-loop epoch:
    For each batch:
      - Maintain obs window W (B,N,H,4)
      - For tau in [0..rollout_steps-1]:
          (a) compute differentiable training loss on current W using remaining GT horizon
          (b) advance W by one step using either GT (with prob teacher_forcing) or model prediction
      - Backprop once per batch on the sum of losses across tau
    """
    model.train()
    total_loss = 0.0
    effective_batches = 0
    rng = random.Random(1234)

    pbar = tqdm(dataloader, desc="Training(CL)")
    for batch in pbar:
        trajectories = batch["trajectories"].to(device)  # (B,N,H,4)
        future_traj = batch["future_trajectory"].to(device)  # (B,N,T,2)
        intent_labels = batch["intent_labels"].to(device)
        vehicle_masks = batch["vehicle_masks"].to(device)

        kg_data = {
            "facility_types": batch["kg_data"]["facility_types"].to(device),
            "positions": batch["kg_data"]["positions"].to(device),
            "edge_index": batch["kg_data"]["edge_index"].to(device),
            "edge_types": batch["kg_data"]["edge_types"].to(device),
        }
        Bsz = trajectories.shape[0]
        kg_positions_global = kg_data["positions"].clone()
        if kg_positions_global.dim() == 2:
            kg_positions_global = kg_positions_global.unsqueeze(0).expand(Bsz, -1, -1)

        # Working observation window in global coords
        W = torch.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0).clone()
        GT = torch.nan_to_num(future_traj, nan=0.0, posinf=0.0, neginf=0.0)

        scene_origin_411 = None
        ref_scene_bn12 = None
        kg_rollout = kg_data
        if coord_frame == COORD_SCENE:
            _, _, kg_rollout, ref_scene_bn12 = normalize_batch_for_digir(
                trajectories, future_traj, kg_data, vehicle_masks, mode=COORD_SCENE
            )
            scene_origin_411 = ref_scene_bn12[:, 0:1, :, :].contiguous()

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        T = GT.shape[2]
        steps = min(rollout_steps, T)

        for tau in range(steps):
            # Fixed-horizon GT target from current time: predict next H steps (H=prediction_horizon=12)
            H = 12
            gt_slice = GT[:, :, tau: tau + H, :]  # (B,N,<=H,2)
            if gt_slice.shape[2] == 0:
                break
            if gt_slice.shape[2] < H:
                # Pad by repeating the last available GT point (keeps continuity, avoids shape mismatch)
                pad_len = H - gt_slice.shape[2]
                last = gt_slice[:, :, -1:, :].expand(-1, -1, pad_len, -1)
                gt_next = torch.cat([gt_slice, last], dim=2)  # (B,N,H,2)
            else:
                gt_next = gt_slice  # (B,N,H,2)

            if coord_frame == COORD_SCENE:
                so = scene_origin_411
                W_norm = W.clone()
                W_norm[:, :, :, :2] -= so
                gt_next_norm = gt_next - so
                kg_in = kg_rollout
            else:
                ref_pa = W[:, :, -1:, :2].clone()
                W_norm = W.clone()
                W_norm[:, :, :, :2] -= ref_pa
                gt_next_norm = gt_next - ref_pa
                kg_in = kg_data
            W_norm = torch.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)
            gt_next_norm = torch.nan_to_num(gt_next_norm, nan=0.0, posinf=0.0, neginf=0.0)

            last_pos_global = torch.nan_to_num(W[:, :, -1:, :2].clone(), nan=0.0, posinf=0.0, neginf=0.0)
            future_local = torch.nan_to_num(
                future_local_from_normed(gt_next_norm, W_norm),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Model forward (differentiable diffusion loss); target = per-agent relative future
            outputs = model(
                W_norm,
                kg_in,
                future_traj=future_local,
                mode="train",
                vehicle_masks=vehicle_masks,
            )
            # Rule losses use global trajectories; always pass global map nodes (not scene-shifted).
            outputs["kg_positions"] = kg_positions_global
            outputs["kg_edge_index"] = kg_in["edge_index"]
            outputs["ref_point"] = last_pos_global

            losses, loss = model.compute_losses(outputs, future_local, intent_labels, vehicle_masks)
            if not torch.isfinite(loss).item():
                # Skip unstable batch
                batch_loss = None
                break

            # Accumulate
            batch_loss = batch_loss + loss

            # Advance window by one step (teacher forcing or prediction)
            use_gt = rng.random() < teacher_forcing
            if use_gt:
                next_xy = GT[:, :, tau, :]  # (B,N,2) global
            else:
                next_xy = predict_one_step_global(
                    model,
                    W,
                    kg_rollout if coord_frame == COORD_SCENE else kg_data,
                    step=generate_step,
                    num_candidates=one_step_k,
                    gt_next_one=GT[:, :, tau, :],
                    vehicle_masks=vehicle_masks,
                    fixed_noise_seed=fixed_noise_seed,
                    scene_origin=scene_origin_411,
                )

            # Update kinematics (heading/speed) from last position
            prev_xy = W[:, :, -1, :2]
            heading, speed = _infer_heading_speed(prev_xy, next_xy, dt=1.0)
            next_state = torch.zeros(W.shape[0], W.shape[1], 4, device=device, dtype=W.dtype)
            next_state[:, :, 0:2] = next_xy
            next_state[:, :, 2] = heading
            next_state[:, :, 3] = speed

            # Slide window
            W = torch.cat([W[:, :, 1:, :], next_state[:, :, None, :]], dim=2)

        if batch_loss is None:
            optimizer.zero_grad()
            continue

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(batch_loss.detach().item())
        effective_batches += 1
        pbar.set_postfix(
            {
                "loss": f"{float(batch_loss.detach().item()):.4f}",
                "tf": f"{teacher_forcing:.2f}",
                "k1": one_step_k,
                "steps": steps,
            }
        )

    return total_loss / max(effective_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Closed-loop trainer for DIGIR (standalone script)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, default="./digir_closed_loop_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_by_location", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_subset", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout_steps", type=int, default=12, help="How many 1-step advances per batch")
    parser.add_argument(
        "--teacher_forcing",
        type=float,
        default=0.2,
        help="When --tf_schedule=constant: probability of GT for window advance. Ignored for linear/cosine (use --tf_start/--tf_end).",
    )
    parser.add_argument(
        "--tf_schedule",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine"],
        help="How teacher-forcing probability changes across epochs.",
    )
    parser.add_argument("--tf_start", type=float, default=0.9, help="TF at first epoch (linear/cosine).")
    parser.add_argument("--tf_end", type=float, default=0.2, help="TF at last epoch (linear/cosine).")
    parser.add_argument(
        "--one_step_k",
        type=int,
        default=5,
        help="Diffusion samples for each 1-step rollout advance; picks closest to next GT (reduces variance). Use 1 to disable.",
    )
    parser.add_argument(
        "--fixed_noise_seed",
        type=int,
        default=None,
        help="If set, fixes torch RNG for each 1-step generate call (restores state after). Optional with one_step_k.",
    )
    parser.add_argument("--generate_step", type=int, default=10, help="DDIM stride used for 1-step prediction")
    parser.add_argument("--eval_batches", type=int, default=0, help="Max eval batches (0=full val)")
    parser.add_argument("--k", type=int, default=5, help="Num samples for minADE/minFDE in eval")
    parser.add_argument(
        "--coord_frame",
        type=str,
        default=COORD_PER_AGENT,
        choices=[COORD_PER_AGENT, COORD_SCENE],
        help="scene: fixed batch origin + shifted KG (match train_digir_full). per_agent: rolling per-agent ref each step.",
    )
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(args.save))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Coordinate frame: {args.coord_frame}")

    # Load dataset
    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)
    train_dataset = InteractionDatasetForDIGIR(args.data, split="train", max_vehicles=10)
    val_dataset = InteractionDatasetForDIGIR(args.data, split="val", max_vehicles=10)

    train_subset = torch.utils.data.Subset(train_dataset, range(min(args.train_subset, len(train_dataset))))
    if args.batch_by_location:
        train_loader = DataLoader(
            train_subset,
            batch_sampler=LocationBatchSampler(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, seed=args.seed),
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.batch_by_location:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=LocationBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, seed=args.seed),
            collate_fn=collate_fn,
        )
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model config must match your dataset horizon
    # We reuse the same defaults as train_digir_full.py
    config = {
        "d_model": 128,
        "d_prior": 128,
        "hist_len": 8,
        "prediction_horizon": 12,
        "num_intent_classes": 4,
        "num_facility_types": 10,
        "traj_enc_layers": 3,
        "graph_enc_layers": 3,
        "scene_tf_layers": 3,
        "v2v_layers": 3,
        "diffusion_tf_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "elementwise_gate": True,
        "diffusion_steps": 50,
        "beta_1": 1e-4,
        "beta_T": 5e-2,
        "lambda_fine": 1.0,
        "lambda_coarse": 0.5,
        "lambda_cross": 0.1,
        # keep rule knobs (if present in your model code)
        "lambda_rule": 0.0,
        "map_margin": 3.0,
        "coord_frame": str(args.coord_frame),
    }

    model = DIGIR(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_ade = float("inf")
    for epoch in range(1, args.epochs + 1):
        tf_epoch = _teacher_forcing_at_epoch(
            args.tf_schedule,
            epoch,
            args.epochs,
            args.tf_start,
            args.tf_end,
            args.teacher_forcing,
        )
        print(f"\nEpoch {epoch}/{args.epochs}  (teacher_forcing={tf_epoch:.3f}, schedule={args.tf_schedule})")
        train_loss = closed_loop_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rollout_steps=args.rollout_steps,
            teacher_forcing=tf_epoch,
            generate_step=args.generate_step,
            one_step_k=args.one_step_k,
            fixed_noise_seed=args.fixed_noise_seed,
            coord_frame=args.coord_frame,
        )
        print(f"Train Loss(CL): {train_loss:.4f}")
        scheduler.step()

        # Open-loop evaluation to match existing metrics for comparison
        metrics = evaluate_open_loop(
            model,
            val_loader,
            device,
            num_samples=args.k,
            max_batches=args.eval_batches,
            coord_frame=args.coord_frame,
        )
        print("\nMetrics (open-loop):")
        print(f"  minADE_5:  {metrics['minADE_5']:.3f} m")
        print(f"  minFDE_5:  {metrics['minFDE_5']:.3f} m")
        print(f"  MissRate:  {metrics['MissRate']:.2%}")
        print(f"  IntentAcc: {metrics['IntentAcc']:.2%}")

        per_loc = metrics.get("per_location", {}) or {}
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
                )

        # Save best by open-loop minADE_5
        if metrics["minADE_5"] < best_ade:
            best_ade = metrics["minADE_5"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "metrics": metrics}, args.save)
            print(f"  [*] Saved best (minADE_5={best_ade:.3f}) -> {args.save}")

    print("Done.")


if __name__ == "__main__":
    main()
