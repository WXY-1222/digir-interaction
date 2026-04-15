"""
Evaluate QCNet adapter on INTERACTION pkl with DIGIR-like print format.

Prints:
- Kinematic: minADE_K, minFDE_K, MissRate
- Semantic: N/A (QCNet adapter does not predict intent labels)
- Safety: optional Collision / Off-Road (requires --with_safety; Off-Road meaningful when KG is enabled)
- Per-location table
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\Admin\Desktop\interaction")
from interaction_dataset_for_digir import InteractionDatasetForDIGIR  # noqa: E402
from QCNet.train_qcnet_interaction_adapter import to_qcnet_data  # noqa: E402

QCNET_SRC = r"C:\Users\Admin\Desktop\QCNet-main\QCNet-main"
if QCNET_SRC not in sys.path:
    sys.path.insert(0, QCNET_SRC)
from predictors.qcnet import QCNet  # noqa: E402


def compute_collision_rate(pred_trajs, vehicle_masks, vehicle_lengths, vehicle_widths):
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
        has_collision = False
        for t in range(T):
            positions = pred_trajs[b, valid_indices, t, :]
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    dist = torch.norm(positions[i] - positions[j])
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
    if edge_index is None or edge_index.numel() == 0:
        d = torch.cdist(points_xy, nodes_xy)
        return d.min(dim=1)[0]
    a = nodes_xy[edge_index[0].long()]
    b = nodes_xy[edge_index[1].long()]
    ab = b - a
    ab2 = (ab * ab).sum(dim=1).clamp_min(1e-8)
    p = points_xy[:, None, :]
    a_ = a[None, :, :]
    ab_ = ab[None, :, :]
    ap = p - a_
    t = (ap * ab_).sum(dim=2) / ab2[None, :]
    t = t.clamp(0.0, 1.0)
    proj = a_ + t[:, :, None] * ab_
    d = torch.norm(p - proj, dim=2)
    return d.min(dim=1)[0]


def compute_off_road_rate(pred_trajs, vehicle_masks, map_positions, edge_index=None, offroad_threshold=3.0):
    if pred_trajs.dim() == 3:
        pred_trajs = pred_trajs.unsqueeze(1)
    B, N, T, _ = pred_trajs.shape
    if map_positions.dim() == 2:
        map_positions = map_positions.unsqueeze(0).expand(B, -1, -1)
    off_road_count = 0
    total_valid = 0
    for b in range(B):
        valid_mask = vehicle_masks[b].bool()
        if not valid_mask.any():
            continue
        total_valid += 1
        nodes_xy = map_positions[b, :, :2]
        positions = pred_trajs[b, valid_mask, :, :]
        points_xy = positions.reshape(-1, 2)
        min_dists = _min_dist_points_to_segments(points_xy, nodes_xy, edge_index).view(-1, T)
        has_offroad = (min_dists.max(dim=1)[0] > offroad_threshold).any().item()
        if has_offroad:
            off_road_count += 1
    return off_road_count / max(total_valid, 1)


def build_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise KeyError("Checkpoint missing 'config'")
    model = QCNet(
        dataset="argoverse_v2",
        input_dim=2,
        hidden_dim=cfg.get("hidden_dim", 128),
        output_dim=2,
        output_head=False,
        num_historical_steps=cfg.get("hist_len", 8),
        num_future_steps=cfg.get("future_len", 12),
        num_modes=cfg.get("num_modes", 6),
        num_recurrent_steps=cfg.get("num_recurrent_steps", 3),
        num_freq_bands=64,
        num_map_layers=1,
        num_agent_layers=2,
        num_dec_layers=2,
        num_heads=8,
        head_dim=16,
        dropout=0.1,
        pl2pl_radius=cfg.get("pl2pl_radius", 80.0),
        time_span=cfg.get("time_span", None),
        pl2a_radius=cfg.get("pl2a_radius", 50.0),
        a2a_radius=cfg.get("a2a_radius", 50.0),
        num_t2m_steps=cfg.get("num_t2m_steps", 8),
        pl2m_radius=cfg.get("pl2m_radius", 80.0),
        a2m_radius=cfg.get("a2m_radius", 80.0),
        lr=cfg.get("lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        T_max=cfg.get("epochs", 20),
        submission_dir="./",
        submission_file_name="unused",
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate(model, dataset, device, k=6, max_samples=0, use_kg=True, with_safety=False):
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    all_ade = []
    all_fde = []
    all_mr = []
    all_col = []
    all_or = []
    per_loc = defaultdict(lambda: {"minADE_K": [], "minFDE_K": [], "MissRate": [], "CollisionRate": [], "OffRoadRate": [], "batches": 0})

    for idx in tqdm(range(n), desc="Evaluating(QCNet-INT)"):
        sample = dataset[idx]
        loc = sample.get("location_name", None)
        data = to_qcnet_data(sample, hist_len=model.num_historical_steps, fut_len=model.num_future_steps, use_kg=use_kg).to(device)
        pred = model(data)

        traj = pred["loc_refine_pos"][..., : model.output_dim]  # (N,K,T,2) local displacement
        pi = F.softmax(pred["pi"], dim=-1)
        gt = data["agent"]["target"][..., : model.output_dim]   # (N,T,2) local displacement
        pm = data["agent"]["predict_mask"][:, model.num_historical_steps :]
        valid_agents = pm[:, -1].bool()
        if not valid_agents.any():
            continue

        topk = min(k, traj.shape[1])
        top_idx = torch.topk(pi, k=topk, dim=-1).indices
        gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj.shape[2], traj.shape[3])
        traj_top = traj.gather(1, gather_idx)  # (N,topk,T,2)
        dxy = torch.norm(traj_top - gt.unsqueeze(1), dim=-1)  # (N,topk,T)
        ade = dxy.mean(dim=-1)
        fde = dxy[:, :, -1]
        min_ade = ade.min(dim=1)[0][valid_agents]
        min_fde = fde.min(dim=1)[0][valid_agents]
        ade_v = float(min_ade.mean().item())
        fde_v = float(min_fde.mean().item())
        mr_v = float((min_fde > 2.0).float().mean().item())

        all_ade.append(ade_v)
        all_fde.append(fde_v)
        all_mr.append(mr_v)
        per_loc[loc]["minADE_K"].append(ade_v)
        per_loc[loc]["minFDE_K"].append(fde_v)
        per_loc[loc]["MissRate"].append(mr_v)
        per_loc[loc]["batches"] += 1

        if with_safety:
            # Convert local displacement prediction to global for safety metrics
            last_pos = data["agent"]["position"][:, model.num_historical_steps - 1 : model.num_historical_steps, :2]  # (N,1,2)
            best_k_idx = fde.argmin(dim=1)  # (N,)
            best = traj_top[torch.arange(traj_top.size(0), device=device), best_k_idx]  # (N,T,2)
            best_global = best + last_pos  # (N,T,2)
            best_global = best_global.unsqueeze(0)
            vm = valid_agents.unsqueeze(0)
            N = best_global.shape[1]
            vehicle_lengths = torch.full((N,), 4.5, device=device)
            vehicle_widths = torch.full((N,), 1.8, device=device)
            col = compute_collision_rate(best_global, vm, vehicle_lengths, vehicle_widths)
            all_col.append(col)
            per_loc[loc]["CollisionRate"].append(col)
            if use_kg:
                kg_pos = sample["kg_data"]["positions"].float().to(device).unsqueeze(0)
                ei = sample["kg_data"]["edge_index"].long().to(device)
                orv = compute_off_road_rate(best_global, vm, kg_pos, edge_index=ei, offroad_threshold=3.0)
                all_or.append(orv)
                per_loc[loc]["OffRoadRate"].append(orv)

    metrics = {
        "minADE_K": float(np.mean(all_ade)) if all_ade else 0.0,
        "minFDE_K": float(np.mean(all_fde)) if all_fde else 0.0,
        "MissRate": float(np.mean(all_mr)) if all_mr else 0.0,
        "IntentAcc": None,
        "ITC": None,
        "CollisionRate": float(np.mean(all_col)) if all_col else None,
        "OffRoadRate": float(np.mean(all_or)) if all_or else None,
        "per_location": {},
    }
    for loc, vals in per_loc.items():
        metrics["per_location"][loc] = {
            "batches": vals["batches"],
            "minADE_K": float(np.mean(vals["minADE_K"])) if vals["minADE_K"] else 0.0,
            "minFDE_K": float(np.mean(vals["minFDE_K"])) if vals["minFDE_K"] else 0.0,
            "MissRate": float(np.mean(vals["MissRate"])) if vals["MissRate"] else 0.0,
            "IntentAcc": None,
            "ITC": None,
            "CollisionRate": float(np.mean(vals["CollisionRate"])) if vals["CollisionRate"] else None,
            "OffRoadRate": float(np.mean(vals["OffRoadRate"])) if vals["OffRoadRate"] else None,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate QCNet adapter on INTERACTION pkl")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--eval_subset", type=int, default=0, help="0=full split")
    parser.add_argument("--no_kg", action="store_true", help="Evaluate in no-KG mode (dummy map).")
    parser.add_argument("--with_safety", action="store_true", help="Compute collision/off-road (slower).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg = build_model_from_checkpoint(args.ckpt, device)
    dataset = InteractionDatasetForDIGIR(args.data, split=args.split, max_vehicles=10)

    metrics = evaluate(
        model=model,
        dataset=dataset,
        device=device,
        k=args.k,
        max_samples=args.eval_subset,
        use_kg=(not args.no_kg),
        with_safety=args.with_safety,
    )

    print("\nMetrics:")
    print("  Kinematic:")
    print(f"    minADE_{args.k}:  {metrics['minADE_K']:.3f} m")
    print(f"    minFDE_{args.k}:  {metrics['minFDE_K']:.3f} m")
    print(f"    MissRate:  {metrics['MissRate']:.2%}")
    print("  Semantic:")
    print("    IntentAcc: N/A (QCNet adapter has no intent head)")
    print("    ITC:       N/A (requires intent labels)")
    print("  Safety:")
    if args.with_safety:
        col = metrics["CollisionRate"]
        off = metrics["OffRoadRate"]
        print(f"    Collision: {col:.2%}" if col is not None else "    Collision: N/A")
        print(f"    Off-Road:  {off:.2%}" if off is not None else "    Off-Road:  N/A")
    else:
        print("    Collision: N/A (--with_safety to enable)")
        print("    Off-Road:  N/A (--with_safety to enable)")

    per_loc = metrics.get("per_location", {}) or {}
    if per_loc:
        print("\nPer-location (avg over evaluated batches):")
        for loc in sorted(per_loc.keys(), key=lambda x: str(x)):
            m = per_loc[loc]
            line = (
                f"  {loc} | batches={m['batches']}"
                f" | minADE_{args.k}={m['minADE_K']:.3f}"
                f" | minFDE_{args.k}={m['minFDE_K']:.3f}"
                f" | MR={m['MissRate']:.2%}"
                f" | IA=N/A | ITC=N/A"
            )
            if args.with_safety:
                line += (
                    f" | Col={m['CollisionRate']:.2%}" if m["CollisionRate"] is not None else " | Col=N/A"
                )
                line += (
                    f" | OR={m['OffRoadRate']:.2%}" if m["OffRoadRate"] is not None else " | OR=N/A"
                )
            print(line)


if __name__ == "__main__":
    main()
