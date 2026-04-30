#!/usr/bin/env python3
"""Convert FJMP INTERACTION preprocess files into the pkl format used by DIGIR.

FJMP stores one ``*.p`` per scene with trajectories and lane graph already in
the scene-centered SE(2) frame.  DIGIR expects one pickle with ``train``/``val``
sample lists plus a KG per location.  This converter keeps FJMP's 10-frame
history and 30-frame future, so the resulting data is 1s -> 3s at 10 Hz.
"""
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_fjmp_scene(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict):
        return data
    return dict(data)


def infer_intent_from_heading(psirad: np.ndarray, has_obs: np.ndarray) -> int:
    valid_hist = has_obs[:10].astype(bool)
    valid_fut = has_obs[10:40].astype(bool)
    if not valid_hist.any() or not valid_fut.any():
        return 3

    hist_heading = float(np.mean(psirad[:10][valid_hist]))
    fut_heading = float(np.mean(psirad[10:40][valid_fut]))
    heading_change = (fut_heading - hist_heading) * 180.0 / math.pi
    while heading_change > 180.0:
        heading_change -= 360.0
    while heading_change < -180.0:
        heading_change += 360.0

    if abs(heading_change) < 15.0:
        return 0
    if heading_change > 15.0:
        return 1
    if heading_change < -15.0:
        return 2
    return 3


def lane_graph_to_kg(graph: dict, max_nodes: int) -> dict:
    if not isinstance(graph, dict) or "ctrs" not in graph:
        return default_kg(max_nodes)

    positions = np.asarray(graph["ctrs"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] == 0:
        return default_kg(max_nodes)
    positions = positions[:, :2]

    if positions.shape[0] > max_nodes:
        positions = positions[:max_nodes]
    num_nodes = int(positions.shape[0])
    facility_types = np.ones(num_nodes, dtype=np.int64)

    edges: List[Tuple[int, int]] = []
    for rel_name in ("pre", "suc"):
        rels = graph.get(rel_name, [])
        if not rels:
            continue
        rel0 = rels[0]
        u = np.asarray(rel0.get("u", []), dtype=np.int64)
        v = np.asarray(rel0.get("v", []), dtype=np.int64)
        for a, b in zip(u.tolist(), v.tolist()):
            if 0 <= a < num_nodes and 0 <= b < num_nodes:
                edges.append((a, b))

    if not edges and num_nodes > 1:
        edges = [(i, i + 1) for i in range(num_nodes - 1)]
    if not edges:
        edge_index = np.zeros((2, 1), dtype=np.int64)
        edge_types = np.zeros(1, dtype=np.int64)
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_types = np.zeros(len(edges), dtype=np.int64)

    return {
        "facility_types": facility_types,
        "positions": positions,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "facility_type_map": {"other": 0, "lane": 1},
    }


def default_kg(num_nodes: int = 20) -> dict:
    xs = np.linspace(-20.0, 20.0, num_nodes, dtype=np.float32)
    positions = np.stack([xs, np.zeros_like(xs)], axis=1)
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    return {
        "facility_types": np.ones(num_nodes, dtype=np.int64),
        "positions": positions,
        "edge_index": np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 1), dtype=np.int64),
        "edge_types": np.zeros(max(1, len(edges)), dtype=np.int64),
        "num_nodes": num_nodes,
        "num_edges": max(1, len(edges)),
        "facility_type_map": {"other": 0, "lane": 1},
    }


def iter_scene_files(preprocess_dir: Path) -> Iterable[Path]:
    def key(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem

    return sorted(preprocess_dir.glob("*.p"), key=key)


def convert_split(preprocess_dir: Path, mapping: Dict[int, str], max_vehicles: int, max_kg_nodes: int):
    samples = []
    kg_per_location = {}

    for p in iter_scene_files(preprocess_dir):
        scene = load_fjmp_scene(p)
        idx = int(scene.get("idx", p.stem))
        location = str(scene.get("city") or mapping.get(idx, "unknown").split("_train_")[0].split("_val_")[0])

        feat_locs = np.asarray(scene["feat_locs"], dtype=np.float32)
        feat_vels = np.asarray(scene.get("feat_vels", np.zeros_like(feat_locs)), dtype=np.float32)
        feat_psirads = np.asarray(scene.get("feat_psirads", np.zeros((*feat_locs.shape[:2], 1))), dtype=np.float32)
        has_obss = np.asarray(scene.get("has_obss", np.ones(feat_locs.shape[:2])), dtype=bool)

        n = min(int(feat_locs.shape[0]), max_vehicles)
        if n <= 0:
            continue
        hist_xy = feat_locs[:n, :10, :2]
        fut_xy = feat_locs[:n, 10:40, :2]
        speeds = np.linalg.norm(feat_vels[:n, :10, :2], axis=-1, keepdims=True)
        headings = feat_psirads[:n, :10, :1]
        trajectory = np.concatenate([hist_xy, headings, speeds], axis=-1).astype(np.float32)

        intents = np.asarray(
            [infer_intent_from_heading(feat_psirads[i, :, 0], has_obss[i]) for i in range(n)],
            dtype=np.int64,
        )
        sample = {
            "location_name": location,
            "case_id": idx,
            "start_frame": 0,
            "trajectory": trajectory,
            "future_trajectory": fut_xy.astype(np.float32),
            "intent_labels": intents,
            "vehicle_types": np.ones(n, dtype=np.int64),
        }
        samples.append(sample)

        if location not in kg_per_location:
            kg_per_location[location] = lane_graph_to_kg(scene.get("graph", {}), max_kg_nodes)

    return samples, kg_per_location


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FJMP INTERACTION preprocess output to DIGIR pkl")
    parser.add_argument("--fjmp_root", type=Path, default=Path("/data/sdb/bitwxy/interaction_raw"))
    parser.add_argument("--output", type=Path, default=Path("/data/sdb/bitwxy/interaction_data/interaction_fjmp_digir_h10_f30.pkl"))
    parser.add_argument("--max_vehicles", type=int, default=10)
    parser.add_argument("--max_kg_nodes", type=int, default=200)
    args = parser.parse_args()

    preprocess = args.fjmp_root / "preprocess"
    train_map = load_pickle(args.fjmp_root / "mapping_train.pkl")
    val_map = load_pickle(args.fjmp_root / "mapping_val.pkl")

    train_samples, train_kg = convert_split(preprocess / "train_interaction", train_map, args.max_vehicles, args.max_kg_nodes)
    val_samples, val_kg = convert_split(preprocess / "val_interaction", val_map, args.max_vehicles, args.max_kg_nodes)

    kg_per_location = dict(train_kg)
    kg_per_location.update({k: v for k, v in val_kg.items() if k not in kg_per_location})
    default = next(iter(kg_per_location.values())) if kg_per_location else default_kg(args.max_kg_nodes)

    dataset = {
        "train": train_samples,
        "val": val_samples,
        "kg": default,
        "kg_per_location": kg_per_location,
        "config": {
            "hist_len": 10,
            "future_len": 30,
            "window_stride": "fjmp_scene",
            "num_intent_classes": 4,
            "input_dim": 4,
            "output_dim": 2,
            "source": "FJMP INTERACTION preprocessing",
            "coordinate_frame": "FJMP scene-centered SE(2)",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved: {args.output}")
    print(f"train={len(train_samples)} val={len(val_samples)} locations={len(kg_per_location)}")


if __name__ == "__main__":
    main()
