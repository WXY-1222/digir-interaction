"""
Visualize DIGIR predictions on INTERACTION processed dataset.

This script plots:
- Historical trajectories (8 steps)
- Ground-truth future trajectories (12 steps)
- K sampled predicted futures from DIGIR.generate()

It uses the same normalization/un-normalization logic as training/evaluation in train_digir_full.py.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import xml.etree.ElementTree as ET
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from torch.utils.data import DataLoader

from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn
from digir_coord_utils import COORD_PER_AGENT, COORD_SCENE, normalize_batch_for_digir


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def parse_osm_xy(osm_path: str):
    """Parse INTERACTION .osm_xy for background rendering."""
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
                tags = {tag.get("k"): tag.get("v") for tag in elem if tag.tag == "tag"}
                ways.append({"nodes": way_nodes, "tags": tags})

        return nodes, ways
    except Exception:
        return None, None


def draw_osm_background(ax, nodes, ways, color="gray", alpha=0.5, linewidth=1.0):
    if not nodes or not ways:
        return
    for way in ways:
        way_nodes = way["nodes"]
        xs = [nodes.get(nid, (None, None))[0] for nid in way_nodes if nid in nodes]
        ys = [nodes.get(nid, (None, None))[1] for nid in way_nodes if nid in nodes]
        if xs and ys:
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=0)


def osm_bbox(nodes):
    """Compute (minx, maxx, miny, maxy) from parsed osm nodes dict."""
    if not nodes:
        return None
    xs = [v[0] for v in nodes.values()]
    ys = [v[1] for v in nodes.values()]
    return (min(xs), max(xs), min(ys), max(ys))


def point_in_bbox(x, y, bbox, margin=0.0):
    minx, maxx, miny, maxy = bbox
    return (minx - margin) <= x <= (maxx + margin) and (miny - margin) <= y <= (maxy + margin)


def find_indices_for_osm(dataset, osm_path, split_name="val", max_results=30, margin=5.0):
    """
    Heuristic: find dataset indices whose trajectories fall inside the OSM bbox.
    Since processed pkl samples may not store scenario/map name, we use coordinate overlap.
    """
    nodes, _ways = parse_osm_xy(osm_path)
    if not nodes:
        raise FileNotFoundError(f"Failed to parse osm: {osm_path}")
    bbox = osm_bbox(nodes)

    hits = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        traj = item["trajectories"]  # (N, 8, 4)
        mask = item["vehicle_mask"].bool()
        if not mask.any():
            continue
        # Use last observed point of the first valid agent as a quick representative.
        i0 = int(torch.where(mask)[0][0].item())
        x = float(traj[i0, -1, 0].item())
        y = float(traj[i0, -1, 1].item())
        if point_in_bbox(x, y, bbox, margin=margin):
            hits.append((idx, item.get("case_id", None), x, y))
            if len(hits) >= max_results:
                break
    return hits, bbox


@torch.no_grad()
def predict_k(model, batch, device, k=5, sampling="ddim", step=10, coord_frame=COORD_PER_AGENT):
    trajectories = batch["trajectories"].to(device)  # (B, N, 8, 4)
    future_traj = batch["future_trajectory"].to(device)  # (B, N, 12, 2)
    vehicle_masks = batch["vehicle_masks"].to(device)  # (B, N)

    kg_data = {
        "facility_types": batch["kg_data"]["facility_types"].to(device),
        "positions": batch["kg_data"]["positions"].to(device),
        "edge_index": batch["kg_data"]["edge_index"].to(device),
        "edge_types": batch["kg_data"]["edge_types"].to(device),
    }
    # Plot overlay uses global map; model may see scene-shifted positions.
    kg_pos_plot = kg_data["positions"].clone()
    if kg_pos_plot.dim() == 2:
        kg_pos_plot = kg_pos_plot.unsqueeze(0).expand(trajectories.shape[0], -1, -1)

    trajectories_norm, future_traj_norm, kg_data, _ref_unused = normalize_batch_for_digir(
        trajectories, future_traj, kg_data, vehicle_masks, mode=coord_frame
    )
    last_pos_global = torch.nan_to_num(
        trajectories[:, :, -1:, :2].clone(), nan=0.0, posinf=0.0, neginf=0.0
    )

    pred_list = []
    for _ in range(k):
        pred = model.generate(
            trajectories_norm,
            kg_data,
            num_points=future_traj.shape[2],
            num_samples=1,
            sampling=sampling,
            step=step,
            bestof=False,
        )
        # DIGIR.generate returns (num_samples, B, N, T, 2)
        if pred.dim() == 5 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.dim() == 5 and pred.shape[1] == 1:
            pred = pred[:, 0]
        elif pred.dim() != 4:
            raise RuntimeError(f"Unexpected pred shape: {tuple(pred.shape)}")

        pred_list.append(pred)

    pred_k_norm = torch.stack(pred_list, dim=0)  # (K, B, N, T, 2) local / displacement
    pred_k_norm = torch.nan_to_num(pred_k_norm, nan=0.0, posinf=0.0, neginf=0.0)
    pred_k_global = pred_k_norm + last_pos_global.unsqueeze(0)

    return {
        "traj_hist": trajectories[:, :, :, :2],  # global
        "traj_future_gt": future_traj,  # global
        "pred_k": pred_k_global,  # global
        "vehicle_masks": vehicle_masks,
        "kg_positions": kg_pos_plot,
        "case_ids": batch.get("case_ids", None),
        "location_names": batch.get("location_names", None),
    }


def plot_scene(
    pred_dict,
    scene_idx=0,
    max_agents=10,
    show_kg=True,
    osm_path=None,
    save_path=None,
    title=None,
    show=True,
):
    hist = pred_dict["traj_hist"][scene_idx]  # (N, 8, 2)
    gt = pred_dict["traj_future_gt"][scene_idx]  # (N, 12, 2)
    pred_k = pred_dict["pred_k"][:, scene_idx]  # (K, N, 12, 2)
    mask = pred_dict["vehicle_masks"][scene_idx].bool()  # (N,)

    valid_idx = torch.where(mask)[0][:max_agents]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Optional: real map background from INTERACTION .osm_xy
    osm_nodes, osm_ways = parse_osm_xy(osm_path) if osm_path else (None, None)
    if osm_nodes and osm_ways:
        draw_osm_background(ax, osm_nodes, osm_ways, color="gray", alpha=0.5, linewidth=1.0)

    if show_kg:
        kg_pos = pred_dict["kg_positions"][scene_idx, :, :2]
        kg_pos = torch.nan_to_num(kg_pos, nan=0.0, posinf=0.0, neginf=0.0)
        ax.scatter(_to_numpy(kg_pos[:, 0]), _to_numpy(kg_pos[:, 1]), s=10, c="lightgray", alpha=0.6, label="KG nodes")

    cmap = plt.get_cmap("tab10")

    for c, i in enumerate(valid_idx.tolist()):
        color = cmap(c % 10)

        h = hist[i]
        g = gt[i]

        ax.plot(_to_numpy(h[:, 0]), _to_numpy(h[:, 1]), "-", color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(_to_numpy(h[0, 0]), _to_numpy(h[0, 1]), s=30, color=color, marker="o", alpha=0.9)
        ax.scatter(_to_numpy(h[-1, 0]), _to_numpy(h[-1, 1]), s=40, color=color, marker="s", alpha=0.9)

        # GT future
        ax.plot(_to_numpy(g[:, 0]), _to_numpy(g[:, 1]), "--", color="black", linewidth=1.5, alpha=0.9)

        # Predicted futures (K samples)
        pk = pred_k[:, i]  # (K, T, 2)
        for k in range(pk.shape[0]):
            ax.plot(_to_numpy(pk[k, :, 0]), _to_numpy(pk[k, :, 1]), color="red", alpha=0.25, linewidth=1.2)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    if title is None:
        title = "DIGIR predictions (hist/GT/pred-K)"
    ax.set_title(title)

    # Legend (minimal)
    handles = []
    labels = []
    if show_kg:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=6))
        labels.append("KG nodes")
    handles.append(plt.Line2D([0], [0], linestyle="--", color="black"))
    labels.append("GT future")
    handles.append(plt.Line2D([0], [0], linestyle="-", color="red"))
    labels.append("Pred samples")
    ax.legend(handles, labels, loc="best")

    plt.tight_layout()
    if save_path:
        d = os.path.dirname(os.path.abspath(save_path))
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def first_val_idx_per_location(dataset):
    """First index in split for each non-null location_name (for multi-map pkl)."""
    out = {}
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return out
    for idx, sample in enumerate(samples):
        loc = sample.get("location_name", None)
        if loc is not None and loc not in out:
            out[loc] = idx
    return out


def animate_scene(
    pred_dict,
    scene_idx=0,
    max_agents=10,
    show_kg=True,
    osm_path=None,
    save_path="prediction.mp4",
    title=None,
    fps=4,
    video_frames=0,
):
    """
    Export dynamic visualization where future GT/pred trajectories unfold over time.
    """
    hist = pred_dict["traj_hist"][scene_idx]           # (N, H, 2)
    gt = pred_dict["traj_future_gt"][scene_idx]        # (N, T, 2)
    pred_k = pred_dict["pred_k"][:, scene_idx]         # (K, N, T, 2)
    mask = pred_dict["vehicle_masks"][scene_idx].bool()
    valid_idx = torch.where(mask)[0][:max_agents]
    T = gt.shape[1]
    F = int(video_frames) if (video_frames is not None and int(video_frames) > 0) else T

    fig, ax = plt.subplots(figsize=(10, 8))

    # Static background
    osm_nodes, osm_ways = parse_osm_xy(osm_path) if osm_path else (None, None)
    if osm_nodes and osm_ways:
        draw_osm_background(ax, osm_nodes, osm_ways, color="gray", alpha=0.5, linewidth=1.0)
    if show_kg:
        kg_pos = pred_dict["kg_positions"][scene_idx, :, :2]
        kg_pos = torch.nan_to_num(kg_pos, nan=0.0, posinf=0.0, neginf=0.0)
        ax.scatter(_to_numpy(kg_pos[:, 0]), _to_numpy(kg_pos[:, 1]), s=10, c="lightgray", alpha=0.55, label="KG nodes")

    cmap = plt.get_cmap("tab10")
    for c, i in enumerate(valid_idx.tolist()):
        color = cmap(c % 10)
        h = hist[i]
        # history as static context
        ax.plot(_to_numpy(h[:, 0]), _to_numpy(h[:, 1]), "-", color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(_to_numpy(h[0, 0]), _to_numpy(h[0, 1]), s=22, color=color, marker="o", alpha=0.9)
        ax.scatter(_to_numpy(h[-1, 0]), _to_numpy(h[-1, 1]), s=32, color=color, marker="s", alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title is None:
        title = "DIGIR dynamic prediction"
    ttl = ax.set_title(title)

    # Dynamic artists
    gt_lines = []
    pred_lines = []
    gt_dots = []
    pred_dots = []
    for c, i in enumerate(valid_idx.tolist()):
        _ = i
        gl, = ax.plot([], [], "--", color="black", linewidth=1.8, alpha=0.95)
        gd = ax.scatter([], [], s=30, color="black", marker="x", alpha=0.95)
        gt_lines.append(gl)
        gt_dots.append(gd)

        p_lines_agent = []
        p_dots_agent = []
        for _k in range(pred_k.shape[0]):
            pl, = ax.plot([], [], "-", color="red", linewidth=1.2, alpha=0.25)
            pd = ax.scatter([], [], s=18, color="red", marker=".", alpha=0.45)
            p_lines_agent.append(pl)
            p_dots_agent.append(pd)
        pred_lines.append(p_lines_agent)
        pred_dots.append(p_dots_agent)

    # Legend
    handles = []
    labels = []
    if show_kg:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=6))
        labels.append("KG nodes")
    handles.append(plt.Line2D([0], [0], linestyle="--", color="black"))
    labels.append("GT future")
    handles.append(plt.Line2D([0], [0], linestyle="-", color="red"))
    labels.append("Pred samples")
    ax.legend(handles, labels, loc="best")

    def init():
        for gl in gt_lines:
            gl.set_data([], [])
        for lines in pred_lines:
            for pl in lines:
                pl.set_data([], [])
        return []

    def update(frame_idx):
        # Map output frame idx (0..F-1) to trajectory progress (0..T-1)
        if F <= 1:
            prog = 0.0
        else:
            prog = frame_idx * (T - 1) / (F - 1)
        i0 = int(np.floor(prog))
        i1 = min(i0 + 1, T - 1)
        a = float(prog - i0)

        for a, i in enumerate(valid_idx.tolist()):
            g = gt[i]
            gx = _to_numpy(g[: i0 + 1, 0])
            gy = _to_numpy(g[: i0 + 1, 1])
            # append interpolated current point for smoother animation
            if i1 > i0:
                gcur = (1.0 - a) * g[i0] + a * g[i1]
                gx = np.append(gx, float(gcur[0].item()))
                gy = np.append(gy, float(gcur[1].item()))
            else:
                gcur = g[i0]
            gt_lines[a].set_data(gx, gy)
            gt_dots[a].set_offsets(_to_numpy(gcur[None, :]))

            for k in range(pred_k.shape[0]):
                p = pred_k[k, i]
                px = _to_numpy(p[: i0 + 1, 0])
                py = _to_numpy(p[: i0 + 1, 1])
                if i1 > i0:
                    pcur = (1.0 - a) * p[i0] + a * p[i1]
                    px = np.append(px, float(pcur[0].item()))
                    py = np.append(py, float(pcur[1].item()))
                else:
                    pcur = p[i0]
                pred_lines[a][k].set_data(px, py)
                pred_dots[a][k].set_offsets(_to_numpy(pcur[None, :]))

        ttl.set_text(f"{title} | frame={frame_idx + 1}/{F} (traj_t~{prog + 1:.2f}/{T})")
        return []

    ani = FuncAnimation(fig, update, init_func=init, frames=F, interval=max(1, int(1000 / fps)), blit=False, repeat=False)

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".gif":
        ani.save(save_path, writer=PillowWriter(fps=fps))
    else:
        try:
            ani.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
        except Exception as e:
            fallback = os.path.splitext(save_path)[0] + ".gif"
            print(f"[animate] ffmpeg unavailable ({e}); fallback to gif: {fallback}")
            ani.save(fallback, writer=PillowWriter(fps=fps))
            save_path = fallback

    plt.close(fig)
    print(f"Saved animation: {save_path}")
    return save_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_digir_root = os.environ.get("DIGIR_ROOT", os.path.join(script_dir, "digir"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./digir_data/interaction_digir.pkl")
    parser.add_argument("--ckpt", type=str, default="./digir_interaction_best.pt")
    parser.add_argument(
        "--digir_root",
        type=str,
        default=default_digir_root,
        help="Path to DIGIR repo root (contains models/). "
        "Defaults to <interaction>/digir, or DIGIR_ROOT env.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--coord_frame",
        type=str,
        default=COORD_PER_AGENT,
        choices=[COORD_PER_AGENT, COORD_SCENE],
        help="Must match training: scene = batch anchor + shifted KG.",
    )
    parser.add_argument("--idx", type=int, default=0, help="Scene index in split")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_agents", type=int, default=10)
    parser.add_argument("--sampling", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--no_kg", action="store_true")
    parser.add_argument(
        "--osm",
        type=str,
        default=None,
        help="Optional INTERACTION .osm_xy path for real map background (e.g. .../maps/DR_USA_Intersection_EP0.osm_xy).",
    )
    parser.add_argument(
        "--find_osm",
        type=str,
        default=None,
        help="If set, scan dataset and print candidate indices whose coordinates overlap this .osm_xy bbox.",
    )
    parser.add_argument("--find_max", type=int, default=30)
    parser.add_argument("--find_margin", type=float, default=5.0)
    parser.add_argument(
        "--find_only",
        action="store_true",
        help="With --find_osm: only print candidate idx, then exit (no model load / no plot).",
    )
    parser.add_argument("--save", type=str, default=None, help="Save image path (e.g., out.png)")
    parser.add_argument("--animate", action="store_true", help="Export dynamic video/gif for the selected idx")
    parser.add_argument("--video", type=str, default="./prediction.mp4", help="Animation output path (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=4, help="Animation fps")
    parser.add_argument("--video_frames", type=int, default=0, help="Output video frame count (0 means use trajectory horizon)")
    parser.add_argument("--auto_osm", action="store_true", help="Auto-select maps/<location_name>.osm_xy for selected sample")
    parser.add_argument(
        "--export_all_locations",
        action="store_true",
        help="Save one PNG per location in the pkl, each with matching maps/<location>.osm_xy background.",
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default=r"C:\Users\Admin\Desktop\interaction\INTERACTION-Dataset-DR-multi-v1_2\maps",
        help="Directory containing INTERACTION *.osm_xy map files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./viz_all_locations",
        help="Output folder for --export_all_locations",
    )
    args = parser.parse_args()

    if not os.path.exists(args.digir_root):
        raise FileNotFoundError(f"DIGIR root not found: {args.digir_root}")
    sys.path.insert(0, args.digir_root)
    from models.digir import DIGIR  # noqa: E402

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data not found: {args.data}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = InteractionDatasetForDIGIR(args.data, split=args.split, max_vehicles=10)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Optional: find candidate indices for a given map
    if args.find_osm:
        hits, bbox = find_indices_for_osm(
            dataset,
            args.find_osm,
            split_name=args.split,
            max_results=args.find_max,
            margin=args.find_margin,
        )
        print(f"[find_osm] split={args.split} osm={args.find_osm}")
        print(f"[find_osm] bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}) margin={args.find_margin}")
        if not hits:
            print("[find_osm] no hits (try increasing --find_margin or check osm choice)")
        else:
            for (idx, case_id, x, y) in hits:
                print(f"  idx={idx} case_id={case_id} last_xy=({x:.3f},{y:.3f})")
        if args.find_only:
            return

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", None)
    if config is None:
        raise KeyError("Checkpoint missing 'config'")

    model = DIGIR(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.export_all_locations:
        loc_to_idx = first_val_idx_per_location(dataset)
        if not loc_to_idx:
            raise RuntimeError(
                "No location_name in samples; regenerate pkl with prepare_interaction_for_digir.py (multi-map)."
            )
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[export_all_locations] maps_dir={args.maps_dir} out_dir={args.out_dir} n_loc={len(loc_to_idx)}")
        for loc in sorted(loc_to_idx.keys()):
            idx = loc_to_idx[loc]
            osm_path = os.path.join(args.maps_dir, f"{loc}.osm_xy")
            if not os.path.isfile(osm_path):
                print(f"  skip {loc}: map not found {osm_path}")
                continue
            batch = collate_fn([dataset[idx]])
            pred = predict_k(
                model, batch, device, k=args.k, sampling=args.sampling, step=args.step, coord_frame=args.coord_frame
            )
            case_ids = pred.get("case_ids", None)
            case_str = str(case_ids[0]) if case_ids is not None else "unknown"
            title = f"{args.split} loc={loc} idx={idx} case={case_str} | K={args.k} {args.sampling} step={args.step}"
            out_png = os.path.join(args.out_dir, f"{loc}.png")
            plot_scene(
                pred,
                scene_idx=0,
                max_agents=args.max_agents,
                show_kg=(not args.no_kg),
                osm_path=osm_path,
                save_path=out_png,
                title=title,
                show=False,
            )
        print(f"Done. Images in: {args.out_dir}")
        return

    # Fetch the idx-th sample (DataLoader style)
    batch = None
    for j, b in enumerate(loader):
        if j == args.idx:
            batch = b
            break
    if batch is None:
        raise IndexError(f"idx out of range: {args.idx} (len={len(dataset)})")

    pred = predict_k(model, batch, device, k=args.k, sampling=args.sampling, step=args.step, coord_frame=args.coord_frame)

    case_ids = pred.get("case_ids", None)
    location_names = pred.get("location_names", None)
    case_str = str(case_ids[0]) if case_ids is not None else "unknown"
    loc_str = str(location_names[0]) if location_names is not None else "unknown"

    # Optionally auto-select matching map by location_name
    if args.auto_osm:
        auto_osm = os.path.join(args.maps_dir, f"{loc_str}.osm_xy")
        if os.path.isfile(auto_osm):
            args.osm = auto_osm
            print(f"[auto_osm] using: {args.osm}")
        else:
            print(f"[auto_osm] map not found for location={loc_str}: {auto_osm}")

    title = f"{args.split} loc={loc_str} idx={args.idx} case={case_str} | K={args.k} {args.sampling} step={args.step}"

    plot_scene(
        pred,
        scene_idx=0,
        max_agents=args.max_agents,
        show_kg=(not args.no_kg),
        osm_path=args.osm,
        save_path=args.save,
        title=title,
        show=(not args.animate),
    )

    if args.animate:
        animate_scene(
            pred,
            scene_idx=0,
            max_agents=args.max_agents,
            show_kg=(not args.no_kg),
            osm_path=args.osm,
            save_path=args.video,
            title=title,
            fps=args.fps,
            video_frames=args.video_frames,
        )


if __name__ == "__main__":
    main()
