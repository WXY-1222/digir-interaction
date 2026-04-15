"""
Train/eval QCNet on INTERACTION pkl (DIGIR preprocessed format) for comparison.

This adapter converts each INTERACTION sample into a minimal QCNet-compatible
HeteroData instance (agent/map nodes + required edges/fields), then trains a
QCNet model with QCNet's original losses.

Goal: provide a fair, runnable QCNet baseline on your current task setup.
"""
import os
import sys
import math
import argparse
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import HeteroData

# Local dataset (your current task data format)
sys.path.insert(0, r"C:\Users\Admin\Desktop\interaction")
from interaction_dataset_for_digir import InteractionDatasetForDIGIR  # noqa: E402

# QCNet source (downloaded repo)
QCNET_SRC = r"C:\Users\Admin\Desktop\QCNet-main\QCNet-main"
if QCNET_SRC not in sys.path:
    sys.path.insert(0, QCNET_SRC)
from predictors.qcnet import QCNet  # noqa: E402
from losses import MixtureNLLLoss, NLLLoss  # noqa: E402


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_map_orientation(positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    positions: (M,2)
    edge_index: (2,E)
    Returns orientation per map polygon node: (M,)
    """
    M = positions.shape[0]
    orient = torch.zeros(M, dtype=torch.float32)
    if edge_index is None or edge_index.numel() == 0:
        return orient
    src = edge_index[0].long().cpu()
    dst = edge_index[1].long().cpu()
    pos = positions.cpu()
    used = torch.zeros(M, dtype=torch.bool)
    for s, d in zip(src.tolist(), dst.tolist()):
        if 0 <= s < M and 0 <= d < M and (not used[s]):
            vec = pos[d] - pos[s]
            orient[s] = math.atan2(float(vec[1]), float(vec[0]))
            used[s] = True
    return orient


def to_qcnet_data(sample: Dict, hist_len: int = 8, fut_len: int = 12, use_kg: bool = True) -> HeteroData:
    """
    Convert one InteractionDatasetForDIGIR sample to QCNet HeteroData.
    """
    traj = sample["trajectories"].float()  # (N,H,4): x,y,heading,speed
    fut = sample["future_trajectory"].float()  # (N,F,2): global xy
    mask = sample["vehicle_mask"].bool()  # (N,)
    kg = sample["kg_data"]

    N = traj.shape[0]
    M = kg["positions"].shape[0] if use_kg else 1

    # Agent features
    pos_hist = traj[:, :hist_len, :2].clone()  # (N,H,2)
    heading_hist = traj[:, :hist_len, 2].clone()  # (N,H)
    speed_hist = traj[:, :hist_len, 3].clone()  # (N,H)
    vel_hist = torch.zeros(N, hist_len, 2, dtype=torch.float32)
    vel_hist[..., 0] = speed_hist * torch.cos(heading_hist)
    vel_hist[..., 1] = speed_hist * torch.sin(heading_hist)

    last_pos = pos_hist[:, hist_len - 1 : hist_len, :].clone()  # (N,1,2)
    target_xy = fut[:, :fut_len, :2] - last_pos  # QCNet target in local displacement frame

    # For QCNet (output_head=False), only [:, :, :2] is used for regression.
    # Keep a third channel placeholder to match expected target[..., -1:] access.
    target = torch.zeros(N, fut_len, 3, dtype=torch.float32)
    target[:, :, :2] = target_xy

    valid_mask = torch.zeros(N, hist_len, dtype=torch.bool)
    valid_mask[mask] = True
    predict_mask = torch.zeros(N, hist_len + fut_len, dtype=torch.bool)
    predict_mask[mask, hist_len:] = True

    agent_type = torch.zeros(N, dtype=torch.uint8)  # vehicle=0
    agent_type[~mask] = 5  # static for padded rows
    agent_category = torch.ones(N, dtype=torch.uint8)  # default uns cored
    agent_category[mask] = 3  # focal/scored style for eval mask in QCNet

    valid_idx = torch.where(mask)[0]
    av_index = int(valid_idx[0].item()) if valid_idx.numel() > 0 else 0

    # Map features from KG (or dummy map for no-KG baseline)
    if use_kg:
        map_pos = kg["positions"].float()[:, :2].clone()  # (M,2)
        facility = kg["facility_types"].long().clone()
        edge_index = kg["edge_index"].long().clone()
        edge_type = kg["edge_types"].long().clone()
        if edge_type.numel() == 0:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.uint8)
        else:
            edge_type = torch.remainder(edge_type, 5).to(torch.uint8)
        map_orient = _build_map_orientation(map_pos, edge_index)
    else:
        # Minimal dummy node to keep QCNet graph modules valid while removing map semantics.
        map_pos = torch.zeros(1, 2, dtype=torch.float32)
        facility = torch.zeros(1, dtype=torch.long)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.uint8)
        map_orient = torch.zeros(1, dtype=torch.float32)

    # map_point = same nodes as map_polygon (minimal valid construction)
    point_pos = map_pos.clone()
    point_orient = map_orient.clone()
    point_magnitude = torch.zeros(M, dtype=torch.float32)
    point_type = torch.clamp(facility, min=0, max=16).to(torch.uint8)  # embedding size 17
    point_side = torch.zeros(M, dtype=torch.uint8)
    poly_type = torch.clamp(facility, min=0, max=3).to(torch.uint8)  # embedding size 4
    poly_is_intersection = torch.zeros(M, dtype=torch.uint8)  # 0/1/2

    id_edge = torch.stack([torch.arange(M, dtype=torch.long), torch.arange(M, dtype=torch.long)], dim=0)

    data = HeteroData()
    data["agent"]["num_nodes"] = N
    data["agent"]["position"] = pos_hist  # (N,H,2)
    data["agent"]["heading"] = heading_hist
    data["agent"]["velocity"] = vel_hist
    data["agent"]["valid_mask"] = valid_mask
    data["agent"]["predict_mask"] = predict_mask
    data["agent"]["target"] = target
    data["agent"]["type"] = agent_type
    data["agent"]["category"] = agent_category
    data["agent"]["av_index"] = torch.tensor(av_index, dtype=torch.long)

    data["map_point"]["num_nodes"] = M
    data["map_point"]["position"] = point_pos
    data["map_point"]["orientation"] = point_orient
    data["map_point"]["magnitude"] = point_magnitude
    data["map_point"]["type"] = point_type
    data["map_point"]["side"] = point_side

    data["map_polygon"]["num_nodes"] = M
    data["map_polygon"]["position"] = map_pos
    data["map_polygon"]["orientation"] = map_orient
    data["map_polygon"]["type"] = poly_type
    data["map_polygon"]["is_intersection"] = poly_is_intersection

    data["map_point", "to", "map_polygon"]["edge_index"] = id_edge
    data["map_polygon", "to", "map_polygon"]["edge_index"] = edge_index
    data["map_polygon", "to", "map_polygon"]["type"] = edge_type
    return data


def compute_losses_qcnet(model: QCNet, data: HeteroData, reg_loss_fn, cls_loss_fn):
    hist = model.num_historical_steps
    out_dim = model.output_dim
    out_head = 1 if model.output_head else 0

    pred = model(data)
    reg_mask = data["agent"]["predict_mask"][:, hist:]
    cls_mask = data["agent"]["predict_mask"][:, -1]

    if model.output_head:
        traj_propose = torch.cat(
            [
                pred["loc_propose_pos"][..., :out_dim],
                pred["loc_propose_head"],
                pred["scale_propose_pos"][..., :out_dim],
                pred["conc_propose_head"],
            ],
            dim=-1,
        )
        traj_refine = torch.cat(
            [
                pred["loc_refine_pos"][..., :out_dim],
                pred["loc_refine_head"],
                pred["scale_refine_pos"][..., :out_dim],
                pred["conc_refine_head"],
            ],
            dim=-1,
        )
    else:
        traj_propose = torch.cat(
            [pred["loc_propose_pos"][..., :out_dim], pred["scale_propose_pos"][..., :out_dim]],
            dim=-1,
        )
        traj_refine = torch.cat(
            [pred["loc_refine_pos"][..., :out_dim], pred["scale_refine_pos"][..., :out_dim]],
            dim=-1,
        )

    pi = pred["pi"]
    gt = torch.cat([data["agent"]["target"][..., :out_dim], data["agent"]["target"][..., -1:]], dim=-1)

    l2_norm = (torch.norm(traj_propose[..., :out_dim] - gt[..., :out_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
    best_mode = l2_norm.argmin(dim=-1)
    traj_propose_best = traj_propose[torch.arange(traj_propose.size(0), device=traj_propose.device), best_mode]
    traj_refine_best = traj_refine[torch.arange(traj_refine.size(0), device=traj_refine.device), best_mode]

    reg_loss_propose = reg_loss_fn(traj_propose_best, gt[..., : out_dim + out_head]).sum(dim=-1) * reg_mask
    reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    reg_loss_propose = reg_loss_propose.mean()

    reg_loss_refine = reg_loss_fn(traj_refine_best, gt[..., : out_dim + out_head]).sum(dim=-1) * reg_mask
    reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    reg_loss_refine = reg_loss_refine.mean()

    cls_loss = (
        cls_loss_fn(
            pred=traj_refine[:, :, -1:].detach(),
            target=gt[:, -1:, : out_dim + out_head],
            prob=pi,
            mask=reg_mask[:, -1:],
        )
        * cls_mask
    )
    cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)

    total = reg_loss_propose + reg_loss_refine + cls_loss
    return total, {
        "reg_prop": float(reg_loss_propose.detach().item()),
        "reg_ref": float(reg_loss_refine.detach().item()),
        "cls": float(cls_loss.detach().item()),
    }


@torch.no_grad()
def evaluate_qcnet(model: QCNet, dataset, device, max_samples=0, use_kg: bool = True):
    model.eval()
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    ade_all = []
    fde_all = []
    mr_all = []

    for i in tqdm(range(n), desc="Eval(QCNet-INT)"):
        d = to_qcnet_data(
            dataset[i],
            hist_len=model.num_historical_steps,
            fut_len=model.num_future_steps,
            use_kg=use_kg,
        ).to(device)
        pred = model(d)
        traj = pred["loc_refine_pos"][..., : model.output_dim]  # (N,K,T,2)
        pi = F.softmax(pred["pi"], dim=-1)
        gt = d["agent"]["target"][..., : model.output_dim]      # (N,T,2)
        pm = d["agent"]["predict_mask"][:, model.num_historical_steps :]

        # Use top-6 modes by probability
        topk = min(6, traj.shape[1])
        prob_idx = torch.topk(pi, k=topk, dim=-1).indices  # (N,topk)
        gather_idx = prob_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj.shape[2], traj.shape[3])
        traj_top = traj.gather(1, gather_idx)  # (N,topk,T,2)

        valid_agents = pm[:, -1].bool()
        if not valid_agents.any():
            continue
        dxy = torch.norm(traj_top - gt.unsqueeze(1), dim=-1)  # (N,topk,T)
        ade = dxy.mean(dim=-1)                                # (N,topk)
        fde = dxy[:, :, -1]                                   # (N,topk)
        min_ade = ade.min(dim=1)[0][valid_agents]
        min_fde = fde.min(dim=1)[0][valid_agents]
        ade_all.append(min_ade.mean().item())
        fde_all.append(min_fde.mean().item())
        mr_all.append((min_fde > 2.0).float().mean().item())

    return {
        "minADE_6": float(np.mean(ade_all)) if ade_all else 0.0,
        "minFDE_6": float(np.mean(fde_all)) if fde_all else 0.0,
        "MissRate": float(np.mean(mr_all)) if mr_all else 0.0,
    }


def build_qcnet_model(args) -> QCNet:
    # Keep comparable horizon/settings to your DIGIR runs.
    model = QCNet(
        dataset="argoverse_v2",
        input_dim=2,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        output_head=False,
        num_historical_steps=args.hist_len,
        num_future_steps=args.future_len,
        num_modes=args.num_modes,
        num_recurrent_steps=args.num_recurrent_steps,
        num_freq_bands=64,
        num_map_layers=1,
        num_agent_layers=2,
        num_dec_layers=2,
        num_heads=8,
        head_dim=16,
        dropout=0.1,
        pl2pl_radius=args.pl2pl_radius,
        time_span=args.time_span,
        pl2a_radius=args.pl2a_radius,
        a2a_radius=args.a2a_radius,
        num_t2m_steps=args.num_t2m_steps,
        pl2m_radius=args.pl2m_radius,
        a2m_radius=args.a2m_radius,
        lr=args.lr,
        weight_decay=args.weight_decay,
        T_max=args.epochs,
        submission_dir="./",
        submission_file_name="unused",
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="QCNet adapter for INTERACTION pkl (DIGIR format)")
    parser.add_argument("--data", type=str, required=True, help="Path to interaction_digir*.pkl")
    parser.add_argument("--save", type=str, default=r"C:\Users\Admin\Desktop\interaction\QCNet\qcnet_interaction_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_subset", type=int, default=5000)
    parser.add_argument("--eval_subset", type=int, default=0, help="0=full val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hist_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--num_recurrent_steps", type=int, default=3)
    # QCNet spatial radii (meters): keep moderate for INTERACTION scale
    parser.add_argument("--pl2pl_radius", type=float, default=80.0)
    parser.add_argument("--time_span", type=int, default=None)
    parser.add_argument("--pl2a_radius", type=float, default=50.0)
    parser.add_argument("--a2a_radius", type=float, default=50.0)
    parser.add_argument("--num_t2m_steps", type=int, default=8)
    parser.add_argument("--pl2m_radius", type=float, default=80.0)
    parser.add_argument("--a2m_radius", type=float, default=80.0)
    parser.add_argument("--no_kg", action="store_true", help="QCNet baseline without map/KG (dummy map node).")
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(args.save))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using QCNet source: {QCNET_SRC}")

    train_ds = InteractionDatasetForDIGIR(args.data, split="train", max_vehicles=10)
    val_ds = InteractionDatasetForDIGIR(args.data, split="val", max_vehicles=10)
    n_train = min(args.train_subset, len(train_ds))
    train_indices = list(range(n_train))
    print(f"Train samples: {n_train}, Val samples: {len(val_ds)}")

    model = build_qcnet_model(args).to(device)
    reg_loss_fn = NLLLoss(component_distribution=["laplace"] * model.output_dim + ["von_mises"] * int(model.output_head), reduction="none")
    cls_loss_fn = MixtureNLLLoss(component_distribution=["laplace"] * model.output_dim + ["von_mises"] * int(model.output_head), reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    best_ade = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_indices)
        total_loss = 0.0
        pbar = tqdm(train_indices, desc=f"Train(QCNet-INT) E{epoch}/{args.epochs}")
        for idx in pbar:
            data = to_qcnet_data(
                train_ds[idx],
                hist_len=args.hist_len,
                fut_len=args.future_len,
                use_kg=(not args.no_kg),
            ).to(device)
            optimizer.zero_grad()
            loss, comp = compute_losses_qcnet(model, data, reg_loss_fn, cls_loss_fn)
            if not torch.isfinite(loss).item():
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "reg": f"{comp['reg_ref']:.3f}", "cls": f"{comp['cls']:.3f}"})

        scheduler.step()
        train_loss = total_loss / max(len(train_indices), 1)
        print(f"Train Loss: {train_loss:.4f}")

        metrics = evaluate_qcnet(
            model,
            val_ds,
            device,
            max_samples=args.eval_subset,
            use_kg=(not args.no_kg),
        )
        print(
            f"Val | minADE_6={metrics['minADE_6']:.3f}  "
            f"minFDE_6={metrics['minFDE_6']:.3f}  MR={metrics['MissRate']:.2%}"
        )

        if metrics["minADE_6"] < best_ade:
            best_ade = metrics["minADE_6"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "config": vars(args),
                    "qcnet_src": QCNET_SRC,
                },
                args.save,
            )
            print(f"  [*] Saved best -> {args.save} (minADE_6={best_ade:.3f})")

    print("\nDone.")
    print(f"Best minADE_6: {best_ade:.3f}")


if __name__ == "__main__":
    main()
