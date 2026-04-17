"""
INTERACTION Dataset Preprocessing for DIGIR

Converts INTERACTION CSV files to DIGIR-compatible format with:
- Multi-agent trajectory extraction
- OSM map parsing for Knowledge Graph
- Intent label inference
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET


def parse_osm_map(osm_path, max_nodes=50, facility_mode='coarse'):
    """
    解析 OSM 地图文件，构建知识图

    Returns:
        kg_data: 包含节点和边的知识图字典
    """
    if not os.path.exists(osm_path):
        print(f"Warning: Map file not found: {osm_path}")
        return build_default_kg()

    try:
        tree = ET.parse(osm_path)
        root = tree.getroot()

        # 提取道路节点和连接关系
        nodes = {}
        ways = []

        for elem in root:
            if elem.tag == 'node':
                node_id = int(elem.get('id'))
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                nodes[node_id] = {'x': x, 'y': y}

            elif elem.tag == 'way':
                way_nodes = [int(nd.get('ref')) for nd in elem if nd.tag == 'nd']
                tags = {tag.get('k'): tag.get('v') for tag in elem if tag.tag == 'tag'}
                ways.append({'nodes': way_nodes, 'tags': tags})

        # ===== Facility type mapping =====
        # coarse mode: 0=other, 1=lane, 2=crosswalk, 3=stop_line
        # binary mode: 0=non_lane, 1=lane_like
        # Priority when a node belongs to multiple way types:
        # crosswalk > stop_line > lane > other
        if facility_mode == 'binary':
            facility_type_map = {
                'other': 0,
                'lane': 1,
                'crosswalk': 1,
                'stop_line': 1,
            }
        else:
            facility_type_map = {
                'other': 0,
                'lane': 1,
                'crosswalk': 2,
                'stop_line': 3,
            }
        priority = {
            'other': 0,
            'lane': 1,
            'stop_line': 2,
            'crosswalk': 3,
        }

        def classify_way(tags):
            """
            Coarse mapping from OSM tags to paper-friendly categories.
            INTERACTION .osm_xy commonly has:
            - type=pedestrian_marking
            - type=line_thin (subtype=dashed/solid)
            - type=virtual
            - type=curbstone
            - type=traffic_sign
            """
            t = tags.get('type', '')
            st = tags.get('subtype', '')

            if t == 'pedestrian_marking':
                return 'crosswalk'
            if t == 'line_thin' and st == 'solid':
                # Heuristic: solid line_thin is treated as stop_line-like boundary marker
                return 'stop_line'
            if t in ('line_thin', 'virtual'):
                return 'lane'
            return 'other'

        # Accumulate candidate categories for each node from incident ways
        node_cat = {nid: 'other' for nid in nodes.keys()}
        for way in ways:
            cat = classify_way(way['tags'])
            for nid in way['nodes']:
                if nid not in node_cat:
                    continue
                if priority[cat] > priority[node_cat[nid]]:
                    node_cat[nid] = cat

        # 构建简化的知识图
        # 取最多 max_nodes 个节点，避免过大；优先保留语义更强的节点
        by_cat = {
            'crosswalk': [],
            'stop_line': [],
            'lane': [],
            'other': [],
        }
        for nid in nodes.keys():
            by_cat[node_cat.get(nid, 'other')].append(nid)

        # Balanced quota to avoid a single category occupying all nodes.
        # Keep lane-like structure dominant, while preserving semantic anchors.
        quotas = {
            'lane': int(max_nodes * 0.50),       # e.g. 25
            'crosswalk': int(max_nodes * 0.20),  # e.g. 10
            'stop_line': int(max_nodes * 0.10),  # e.g. 5
        }

        node_ids = []
        used = set()

        for cat in ['lane', 'crosswalk', 'stop_line']:
            q = quotas[cat]
            cnt = 0
            for nid in by_cat[cat]:
                if nid in used:
                    continue
                node_ids.append(nid)
                used.add(nid)
                cnt += 1
                if cnt >= q or len(node_ids) >= max_nodes:
                    break
            if len(node_ids) >= max_nodes:
                break

        # Fill remaining budget with other categories by priority
        if len(node_ids) < max_nodes:
            for cat in ['lane', 'crosswalk', 'stop_line', 'other']:
                for nid in by_cat[cat]:
                    if nid in used:
                        continue
                    node_ids.append(nid)
                    used.add(nid)
                    if len(node_ids) >= max_nodes:
                        break
                if len(node_ids) >= max_nodes:
                    break
        num_nodes = len(node_ids)

        positions = np.array([[nodes[nid]['x'], nodes[nid]['y']] for nid in node_ids], dtype=np.float32)

        # 节点类型：0=other, 1=lane, 2=crosswalk, 3=stop_line
        facility_types = np.array(
            [facility_type_map.get(node_cat.get(nid, 'other'), 0) for nid in node_ids],
            dtype=np.int64
        )

        # 构建边：连接相邻的道路节点
        edges = []
        edge_types = []

        for way in ways:
            way_node_ids = way['nodes']
            for i in range(len(way_node_ids) - 1):
                if way_node_ids[i] in node_ids and way_node_ids[i+1] in node_ids:
                    idx1 = node_ids.index(way_node_ids[i])
                    idx2 = node_ids.index(way_node_ids[i+1])
                    edges.append([idx1, idx2])
                    edge_types.append(0)  # connected_to

        if len(edges) == 0:
            # 如果没有边，创建默认连接
            for i in range(num_nodes - 1):
                edges.append([i, i+1])
                edge_types.append(0)

        edge_index = np.array(edges, dtype=np.int64).T
        edge_types = np.array(edge_types, dtype=np.int64)

        kg_data = {
            'facility_types': facility_types,
            'positions': positions,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'num_nodes': num_nodes,
            'num_edges': len(edges),
            'facility_type_map': facility_type_map,
        }

        return kg_data

    except Exception as e:
        print(f"Error parsing OSM: {e}")
        return build_default_kg()


def build_default_kg(num_nodes=20):
    """构建默认知识图（当地图解析失败时使用）"""
    # 创建网格状结构
    positions = np.random.randn(num_nodes, 2).astype(np.float32) * 50
    facility_types = np.zeros(num_nodes, dtype=np.int64)

    # 创建连接边
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i+1])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.array([[0], [0]], dtype=np.int64)
    edge_types = np.zeros(len(edges), dtype=np.int64) if edges else np.array([0], dtype=np.int64)

    return {
        'facility_types': facility_types,
        'positions': positions,
        'edge_index': edge_index,
        'edge_types': edge_types,
        'num_nodes': num_nodes,
        'num_edges': len(edges),
        'facility_type_map': {
            'other': 0,
            'lane': 1,
            'crosswalk': 2,
            'stop_line': 3,
        },
    }


def compute_intent_label(track_data, num_history=8, num_future=12):
    """
    根据轨迹的航向角变化推断意图
    Returns: 0=直行, 1=左转, 2=右转, 3=其他
    """
    if len(track_data) < num_history + num_future:
        return -1

    hist_heading = track_data['psi_rad'].iloc[:num_history].mean()
    future_heading = track_data['psi_rad'].iloc[num_history:num_history+num_future].mean()

    heading_change = (future_heading - hist_heading) * 180 / np.pi

    # 归一化到 [-180, 180]
    while heading_change > 180:
        heading_change -= 360
    while heading_change < -180:
        heading_change += 360

    if abs(heading_change) < 15:
        return 0  # Straight
    elif heading_change > 15:
        return 1  # Left
    elif heading_change < -15:
        return 2  # Right
    else:
        return 3  # Other


def process_interaction_file(
    csv_path,
    kg_data,
    location_name=None,
    hist_len=8,
    future_len=12,
    min_speed=0.3,
    window_stride=5,
):
    """
    处理单个 INTERACTION CSV 文件

    INTERACTION 是多人场景，需要按 case_id 分组处理
    """
    df = pd.read_csv(csv_path)

    samples = []

    # 按 case_id 分组（同一场景）
    for case_id, case_data in tqdm(df.groupby('case_id'), desc=f"Processing {os.path.basename(csv_path)}"):
        # 按 track_id 分组（每个车辆）
        tracks = {tid: tdata for tid, tdata in case_data.groupby('track_id')}

        # 找到所有车辆都存在的帧范围
        common_frames = None
        for tid, tdata in tracks.items():
            frames = set(tdata['frame_id'].values)
            if common_frames is None:
                common_frames = frames
            else:
                common_frames = common_frames & frames

        common_frames = sorted(list(common_frames))
        if len(common_frames) < hist_len + future_len:
            continue

        # 滑动窗口提取样本
        stride = max(1, int(window_stride))
        for start_idx in range(0, len(common_frames) - hist_len - future_len + 1, stride):
            hist_frames = common_frames[start_idx:start_idx + hist_len]
            future_frames = common_frames[start_idx + hist_len:start_idx + hist_len + future_len]

            # 收集所有车辆的轨迹
            trajectories = []
            future_trajectories = []
            intent_labels = []
            vehicle_types = []

            for tid, tdata in tracks.items():
                hist_data = tdata[tdata['frame_id'].isin(hist_frames)].sort_values('frame_id')
                future_data = tdata[tdata['frame_id'].isin(future_frames)].sort_values('frame_id')

                if len(hist_data) != hist_len or len(future_data) != future_len:
                    continue

                # 提取特征 [x, y, heading, speed]
                hist_features = np.column_stack([
                    hist_data['x'].values,
                    hist_data['y'].values,
                    hist_data['psi_rad'].values,
                    np.sqrt(hist_data['vx']**2 + hist_data['vy']**2).values
                ])

                future_pos = np.column_stack([
                    future_data['x'].values,
                    future_data['y'].values
                ])

                # 推断意图
                combined_data = pd.concat([hist_data, future_data])
                intent = compute_intent_label(combined_data, hist_len, future_len)
                if intent == -1:
                    continue

                # 车辆类型编码
                agent_type = hist_data['agent_type'].iloc[0]
                type_map = {'car': 0, 'truck': 1, 'bus': 1, 'pedestrian': 2, 'bicycle': 3, 'motorcycle': 4}
                vtype = type_map.get(agent_type, 0)

                trajectories.append(hist_features)
                future_trajectories.append(future_pos)
                intent_labels.append(intent)
                vehicle_types.append(vtype)

            # 至少需要2辆车才有交互意义
            if len(trajectories) < 1:
                continue

            # 限制最多10辆车（避免过大）
            max_vehicles = 10
            if len(trajectories) > max_vehicles:
                trajectories = trajectories[:max_vehicles]
                future_trajectories = future_trajectories[:max_vehicles]
                intent_labels = intent_labels[:max_vehicles]
                vehicle_types = vehicle_types[:max_vehicles]

            # 填充到固定大小（方便 batch）
            num_vehicles = len(trajectories)

            sample = {
                'location_name': location_name,
                'case_id': case_id,
                'start_frame': hist_frames[0],
                'num_vehicles': num_vehicles,
                'trajectory': np.array(trajectories, dtype=np.float32),  # (N, 8, 4)
                'future_trajectory': np.array(future_trajectories, dtype=np.float32),  # (N, 12, 2)
                'intent_labels': np.array(intent_labels, dtype=np.int64),  # (N,)
                'vehicle_types': np.array(vehicle_types, dtype=np.int64),  # (N,)
            }

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Preprocess INTERACTION dataset for DIGIR')
    parser.add_argument('--data_dir', type=str,
                        default='C:\\Users\\Admin\\Desktop\\interaction\\INTERACTION-Dataset-DR-multi-v1_2',
                        help='INTERACTION data directory')
    parser.add_argument('--output_dir', type=str, default='./digir_data', help='Output directory')
    parser.add_argument('--hist_len', type=int, default=8, help='History length')
    parser.add_argument('--future_len', type=int, default=12, help='Prediction horizon')
    parser.add_argument('--window_stride', type=int, default=5, help='Sliding window stride')
    parser.add_argument('--max_nodes', type=int, default=50, help='Max map nodes kept per location KG')
    parser.add_argument(
        '--facility_mode',
        type=str,
        default='coarse',
        choices=['coarse', 'binary'],
        help='Facility type mapping mode: coarse(4 classes) or binary( lane_like / non_lane )',
    )
    parser.add_argument(
        '--locations',
        type=str,
        default=None,
        help='Comma-separated location names to process (e.g., DR_CHN_Roundabout_LN). Default: all.',
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output pkl filename (default auto from locations/hist/future).',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    map_dir = data_dir / 'maps'

    # 获取所有训练文件
    train_files = list(train_dir.glob('*_train.csv'))

    print(f"Found {len(train_files)} training files")

    location_filter = None
    if args.locations:
        location_filter = {x.strip() for x in args.locations.split(',') if x.strip()}
        print(f"Location filter enabled: {sorted(location_filter)}")

    all_train_samples = []
    all_val_samples = []
    kg_data_per_location = {}

    # 处理每个地点的训练文件
    for train_file in train_files:
        location_name = train_file.stem.replace('_train', '')
        if location_filter is not None and location_name not in location_filter:
            continue
        print(f"\nProcessing {location_name}...")

        # 查找对应的验证文件
        val_file = val_dir / f"{location_name}_val.csv"

        # 解析地图
        osm_file = map_dir / f"{location_name}.osm_xy"
        if not osm_file.exists():
            osm_file = map_dir / f"{location_name}.osm"

        print(f"  Loading map: {osm_file}")
        kg_data = parse_osm_map(str(osm_file), max_nodes=args.max_nodes, facility_mode=args.facility_mode)
        kg_data_per_location[location_name] = kg_data

        # 处理训练数据
        print(f"  Processing training data...")
        train_samples = process_interaction_file(
            str(train_file), kg_data, location_name, args.hist_len, args.future_len, window_stride=args.window_stride
        )
        all_train_samples.extend(train_samples)
        print(f"    Generated {len(train_samples)} samples")

        # 处理验证数据
        if val_file.exists():
            print(f"  Processing validation data...")
            val_samples = process_interaction_file(
                str(val_file), kg_data, location_name, args.hist_len, args.future_len, window_stride=args.window_stride
            )
            all_val_samples.extend(val_samples)
            print(f"    Generated {len(val_samples)} samples")

    print(f"\n{'='*50}")
    print(f"Total training samples: {len(all_train_samples)}")
    print(f"Total validation samples: {len(all_val_samples)}")

    # 统计意图分布
    intent_counts = defaultdict(int)
    for sample in all_train_samples:
        for intent in sample['intent_labels']:
            intent_counts[intent] += 1

    print("\nIntent distribution:")
    intent_names = {0: 'Straight', 1: 'Left', 2: 'Right', 3: 'Other'}
    total_intents = sum(intent_counts.values())
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent_names.get(intent, 'Unknown')}: {count} ({count/total_intents*100:.1f}%)")

    # For single-location preprocessing, use that location's KG.
    # If multiple locations are processed, default to the first (still not ideal for mixed-map training).
    if kg_data_per_location:
        default_kg = next(iter(kg_data_per_location.values()))
    else:
        default_kg = build_default_kg()

    # 保存数据
    dataset = {
        'train': all_train_samples,
        'val': all_val_samples,
        'kg': default_kg,
        'kg_per_location': kg_data_per_location,
        'config': {
            'hist_len': args.hist_len,
            'future_len': args.future_len,
            'window_stride': args.window_stride,
            'num_intent_classes': 4,
            'input_dim': 4,
            'output_dim': 2,
            'facility_type_map': default_kg.get('facility_type_map', {}),
            'facility_mode': args.facility_mode,
            'max_nodes': args.max_nodes,
        }
    }

    if args.output_name:
        out_name = args.output_name
    else:
        stride_suffix = f"_s{args.window_stride}"
        if location_filter and len(location_filter) == 1:
            loc = next(iter(location_filter))
            out_name = f"interaction_digir_{loc}_h{args.hist_len}_f{args.future_len}{stride_suffix}.pkl"
        elif location_filter and len(location_filter) > 1:
            out_name = f"interaction_digir_{len(location_filter)}loc_h{args.hist_len}_f{args.future_len}{stride_suffix}.pkl"
        else:
            out_name = f"interaction_digir{stride_suffix}.pkl"

    output_path = os.path.join(args.output_dir, out_name)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'='*50}")
    print(f"Dataset saved to: {output_path}")
    print(f"Window stride: {args.window_stride}")
    print(f"KG nodes: {default_kg['num_nodes']}")
    print(f"KG edges: {default_kg['num_edges']}")
    print("✓ Ready for DIGIR training!")


if __name__ == "__main__":
    main()
