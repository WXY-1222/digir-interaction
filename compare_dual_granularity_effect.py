"""
Side-by-side visualization for dual-granularity effect.

Compare two checkpoints on the same sample:
- left: full model (dual-granularity)
- right: ablation model (e.g., cross-attn removed)

Shows history / GT / K sampled predictions to inspect:
- trajectory spread (variance collapse intuition)
- road-following quality and endpoint concentration
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.patches import Ellipse

from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn
from digir_coord_utils import COORD_PER_AGENT, COORD_SCENE, normalize_batch_for_digir


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def parse_osm_xy(osm_path: str):
    if osm_path is None or (not os.path.exists(osm_path)):
        return None, None
    try:
        tree = ET.parse(osm_path)
        root = tree.getroot()
        nodes = {}
        ways = []
        for elem in root:
            if elem.tag == "node":
                node_id = int(elem.get("id"))
                x = float(elem.get("x", 0))
                y = float(elem.get("y", 0))
                nodes[node_id] = (x, y)
            elif elem.tag == "way":
                way_nodes = [int(nd.get("ref")) for nd in elem if nd.tag == "nd"]
                ways.append({"nodes": way_nodes})
        return nodes, ways
    except Exception:
        return None, None


def draw_osm_background(ax, nodes, ways, color="gray", alpha=0.45, linewidth=1.0):
    if not nodes or not ways:
        return
    for way in ways:
        ids = way["nodes"]
        xs = [nodes[i][0] for i in ids if i in nodes]
        ys = [nodes[i][1] for i in ids if i in nodes]
        if xs and ys:
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=0)


@torch.no_grad()
def predict_k_global(model, batch, device, k=20, step=10, coord_frame=COORD_PER_AGENT):
    trajectories = batch["trajectories"].to(device)  # (1,N,H,4)
    future_traj = batch["future_trajectory"].to(device)  # (1,N,T,2)
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
    last_pos_global = trajectories[:, :, -1:, :2].clone()
    last_pos_global = torch.nan_to_num(last_pos_global, nan=0.0, posinf=0.0, neginf=0.0)

    pred_list = []
    for _ in range(k):
        pred = model.generate(
            trajectories_norm,
            kg_data,
            num_points=future_traj.shape[2],
            num_samples=1,
            sampling="ddim",
            step=step,
            bestof=False,
        )  # (1,1,N,T,2)
        pred = pred[0]  # (1,N,T,2)
        pred_list.append(pred)
    pred_k = torch.stack(pred_list, dim=0)  # (K,1,N,T,2), local/displacement
    pred_k = torch.nan_to_num(pred_k, nan=0.0, posinf=0.0, neginf=0.0)
    pred_k_global = pred_k + last_pos_global.unsqueeze(0)

    return {
        "hist": trajectories[:, :, :, :2],  # (1,N,H,2) global
        "gt": future_traj,                  # (1,N,T,2) global
        "pred_k": pred_k_global,            # (K,1,N,T,2) global
        "mask": vehicle_masks,              # (1,N)
    }


def sample_min_ade_fde(pred_k, gt, mask):
    # pred_k: (K,1,N,T,2), gt: (1,N,T,2), mask: (1,N)
    pred = pred_k[:, 0]  # (K,N,T,2)
    gt0 = gt[0]          # (N,T,2)
    m = mask[0].bool()   # (N,)
    if not m.any():
        return 0.0, 0.0
    pred = pred[:, m]    # (K,n,T,2)
    gt0 = gt0[m]         # (n,T,2)
    d = torch.norm(pred - gt0.unsqueeze(0), dim=-1)  # (K,n,T)
    ade = d.mean(dim=-1)                               # (K,n)
    fde = d[:, :, -1]                                  # (K,n)
    min_ade = ade.min(dim=0)[0].mean().item()
    min_fde = fde.min(dim=0)[0].mean().item()
    return float(min_ade), float(min_fde)


def draw_panel(ax, pred_dict, title, max_agents=10, osm_nodes=None, osm_ways=None):
    hist = pred_dict["hist"][0]
    gt = pred_dict["gt"][0]
    pred_k = pred_dict["pred_k"][:, 0]
    mask = pred_dict["mask"][0].bool()
    valid_idx = torch.where(mask)[0][:max_agents]

    if osm_nodes and osm_ways:
        draw_osm_background(ax, osm_nodes, osm_ways)

    cmap = plt.get_cmap("tab10")
    for c, i in enumerate(valid_idx.tolist()):
        col = cmap(c % 10)
        h = hist[i]
        g = gt[i]
        ax.plot(_to_numpy(h[:, 0]), _to_numpy(h[:, 1]), color=col, linewidth=2.0, alpha=0.95)
        ax.plot(_to_numpy(g[:, 0]), _to_numpy(g[:, 1]), color=col, linestyle="--", linewidth=2.0, alpha=0.95)
        for k in range(pred_k.shape[0]):
            p = pred_k[k, i]
            ax.plot(_to_numpy(p[:, 0]), _to_numpy(p[:, 1]), color=col, linewidth=1.0, alpha=0.22)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _draw_end_scatter_and_ellipse(
    ax,
    pred_k_agent: torch.Tensor,
    color,
    alpha_scatter=0.80,
    alpha_ellipse=0.95,
    min_axis_len=0.9,
):
    """
    pred_k_agent: (K, T, 2) in global coordinates for one agent.
    Draw:
      - endpoint scatter (x_T, y_T) for K samples
      - 95% covariance ellipse for endpoints
    """
    if pred_k_agent.dim() != 3 or pred_k_agent.shape[0] < 2:
        return
    end_xy = pred_k_agent[:, -1, :]  # (K, 2)
    pts = _to_numpy(end_xy)
    if pts.shape[0] < 2:
        return

    # Endpoint scatter (high-visibility style)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=24,
        c=[color],
        alpha=alpha_scatter,
        marker="o",
        edgecolors="black",
        linewidths=0.35,
        zorder=6,
    )

    # Covariance ellipse (95% for 2D Gaussian: chi2_{2,0.95} ~= 5.991)
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if cov.shape != (2, 2):
        return
    # Guard against degenerate covariance
    cov = cov + np.eye(2) * 1e-6
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if np.any(evals <= 0):
        return

    chi2_95 = 5.991
    width = 2.0 * np.sqrt(evals[0] * chi2_95)
    height = 2.0 * np.sqrt(evals[1] * chi2_95)
    # Avoid near-zero ellipse becoming invisible when predictions are tightly collapsed.
    width = max(width, float(min_axis_len))
    height = max(height, float(min_axis_len))
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

    ell = Ellipse(
        xy=(mean[0], mean[1]),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linestyle="--",
        linewidth=1.8,
        alpha=alpha_ellipse,
        zorder=5,
    )
    ax.add_patch(ell)
    # Ellipse center marker
    ax.scatter(
        [mean[0]],
        [mean[1]],
        s=28,
        c=[color],
        marker="x",
        linewidths=1.3,
        zorder=7,
    )


def render_compare_figure(
    dataset,
    idx,
    model_full,
    model_ablate,
    device,
    k,
    step,
    max_agents,
    frame_full,
    frame_ablate,
    auto_osm=False,
    maps_dir=None,
    osm_override=None,
    save_path="./compare.png",
    split_name="val",
    show_endpoint_stats=True,
):
    item = dataset[idx]
    batch = collate_fn([item])

    osm_path = osm_override
    loc_name = item.get("location_name", None)
    if auto_osm and loc_name and maps_dir:
        p = os.path.join(maps_dir, f"{loc_name}.osm_xy")
        if os.path.isfile(p):
            osm_path = p
    osm_nodes, osm_ways = parse_osm_xy(osm_path) if osm_path else (None, None)

    # VRAM-friendly sequential inference:
    # keep only one model on GPU at a time to avoid OOM.
    if device.type == "cuda":
        model_ablate.cpu()
        torch.cuda.empty_cache()
    pred_full = predict_k_global(model_full, batch, device, k=k, step=step, coord_frame=frame_full)

    if device.type == "cuda":
        model_full.cpu()
        torch.cuda.empty_cache()
        model_ablate.to(device)
    pred_abl = predict_k_global(model_ablate, batch, device, k=k, step=step, coord_frame=frame_ablate)

    # restore for next call in batch mode
    if device.type == "cuda":
        model_full.to(device)

    ade_f, fde_f = sample_min_ade_fde(pred_full["pred_k"], pred_full["gt"], pred_full["mask"])
    ade_a, fde_a = sample_min_ade_fde(pred_abl["pred_k"], pred_abl["gt"], pred_abl["mask"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_panel(
        axes[0],
        pred_full,
        title=f"Full model | minADE@K={ade_f:.2f}, minFDE@K={fde_f:.2f}",
        max_agents=max_agents,
        osm_nodes=osm_nodes,
        osm_ways=osm_ways,
    )
    draw_panel(
        axes[1],
        pred_abl,
        title=f"Ablation | minADE@K={ade_a:.2f}, minFDE@K={fde_a:.2f}",
        max_agents=max_agents,
        osm_nodes=osm_nodes,
        osm_ways=osm_ways,
    )

    if show_endpoint_stats:
        # Overlay endpoint scatter + 95% ellipse for each valid agent on both panels.
        mask = pred_full["mask"][0].bool()
        valid_idx = torch.where(mask)[0][:max_agents]
        cmap = plt.get_cmap("tab10")
        for c, i in enumerate(valid_idx.tolist()):
            col = cmap(c % 10)
            _draw_end_scatter_and_ellipse(axes[0], pred_full["pred_k"][:, 0, i], color=col)
            _draw_end_scatter_and_ellipse(axes[1], pred_abl["pred_k"][:, 0, i], color=col)
    fig.suptitle(
        f"{split_name} idx={idx} loc={loc_name} | K={k}, step={step}",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_digir_root = os.environ.get("DIGIR_ROOT", os.path.join(script_dir, "digir"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt_full", type=str, required=True, help="Checkpoint with full dual-granularity model.")
    parser.add_argument("--ckpt_ablate", type=str, required=True, help="Checkpoint with ablation model.")
    parser.add_argument(
        "--digir_root",
        type=str,
        default=default_digir_root,
        help="Path to DIGIR code root. Defaults to <interaction>/digir or DIGIR_ROOT env.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument(
        "--idx_list",
        type=str,
        default=None,
        help="Comma-separated indices for batch export, e.g. 0,10,25",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=0,
        help="Batch export first N indices from split when idx_list is not set (0=disabled).",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max_agents", type=int, default=10)
    parser.add_argument("--coord_frame_full", type=str, default=None, choices=[COORD_PER_AGENT, COORD_SCENE, None])
    parser.add_argument("--coord_frame_ablate", type=str, default=None, choices=[COORD_PER_AGENT, COORD_SCENE, None])
    parser.add_argument("--maps_dir", type=str, default=r"C:\Users\Admin\Desktop\interaction\INTERACTION-Dataset-DR-multi-v1_2\maps")
    parser.add_argument("--auto_osm", action="store_true")
    parser.add_argument("--osm", type=str, default=None)
    parser.add_argument("--save", type=str, default="./compare_dual_granularity.png")
    parser.add_argument(
        "--no_endpoint_stats",
        action="store_true",
        help="Disable endpoint scatter + 95% ellipse overlay.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./compare_dual_granularity_batch",
        help="Output dir for batch export mode.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.digir_root):
        raise FileNotFoundError(args.digir_root)
    sys.path.insert(0, args.digir_root)
    from models.digir import DIGIR  # noqa: E402

    dataset = InteractionDatasetForDIGIR(args.data, split=args.split, max_vehicles=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full checkpoint
    ckpt_full = torch.load(args.ckpt_full, map_location="cpu")
    cfg_full = ckpt_full["config"]
    model_full = DIGIR(cfg_full).to(device)
    model_full.load_state_dict(ckpt_full["model"], strict=True)
    model_full.eval()
    frame_full = args.coord_frame_full or cfg_full.get("coord_frame", COORD_PER_AGENT)

    # Load ablation checkpoint
    ckpt_ablate = torch.load(args.ckpt_ablate, map_location="cpu")
    cfg_ablate = ckpt_ablate["config"]
    model_ablate = DIGIR(cfg_ablate).to(device)
    model_ablate.load_state_dict(ckpt_ablate["model"], strict=True)
    model_ablate.eval()
    frame_ablate = args.coord_frame_ablate or cfg_ablate.get("coord_frame", COORD_PER_AGENT)

    # Batch export mode: --idx_list or --max_count
    if args.idx_list is not None or args.max_count > 0:
        if args.idx_list:
            idxs = [int(x.strip()) for x in args.idx_list.split(",") if x.strip()]
        else:
            idxs = list(range(min(args.max_count, len(dataset))))
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[batch] exporting {len(idxs)} comparisons -> {args.out_dir}")
        for idx in idxs:
            if idx < 0 or idx >= len(dataset):
                print(f"  skip idx={idx}: out of range")
                continue
            out_path = os.path.join(args.out_dir, f"compare_idx{idx}.png")
            render_compare_figure(
                dataset=dataset,
                idx=idx,
                model_full=model_full,
                model_ablate=model_ablate,
                device=device,
                k=args.k,
                step=args.step,
                max_agents=args.max_agents,
                frame_full=frame_full,
                frame_ablate=frame_ablate,
                auto_osm=args.auto_osm,
                maps_dir=args.maps_dir,
                osm_override=args.osm,
                save_path=out_path,
                split_name=args.split,
                show_endpoint_stats=(not args.no_endpoint_stats),
            )
    else:
        if args.idx < 0 or args.idx >= len(dataset):
            raise IndexError(f"idx out of range: {args.idx} (len={len(dataset)})")
        render_compare_figure(
            dataset=dataset,
            idx=args.idx,
            model_full=model_full,
            model_ablate=model_ablate,
            device=device,
            k=args.k,
            step=args.step,
            max_agents=args.max_agents,
            frame_full=frame_full,
            frame_ablate=frame_ablate,
            auto_osm=args.auto_osm,
            maps_dir=args.maps_dir,
            osm_override=args.osm,
            save_path=args.save,
            split_name=args.split,
            show_endpoint_stats=(not args.no_endpoint_stats),
        )


if __name__ == "__main__":
    main()
