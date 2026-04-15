"""
INTERACTION Dataset Loader for DIGIR

Usage:
    from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn

    dataset = InteractionDatasetForDIGIR('./digir_data/interaction_digir.pkl', split='train')
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class InteractionDatasetForDIGIR(Dataset):
    """INTERACTION Dataset compatible with DIGIR model"""

    def __init__(self, data_path, split='train', max_vehicles=10):
        """
        Args:
            data_path: path to interaction_digir.pkl
            split: 'train' or 'val'
            max_vehicles: maximum number of vehicles to consider
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.samples = data[split]
        self.kg = data.get('kg', None)
        self.kg_per_location = data.get('kg_per_location', None)
        self.config = data['config']
        self.max_vehicles = max_vehicles
        self.sample_locations = [s.get('location_name', None) for s in self.samples]

        print(f"Loaded INTERACTION {split} set: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        location_name = sample.get('location_name', None)

        # 提取数据（已经是 numpy 数组）
        trajectory = torch.from_numpy(sample['trajectory']).float()  # (N, 8, 4)
        future_traj = torch.from_numpy(sample['future_trajectory']).float()  # (N, 12, 2)
        intent_labels = torch.from_numpy(sample['intent_labels']).long()  # (N,)

        N = trajectory.shape[0]

        # 填充或截断到固定车辆数
        if N < self.max_vehicles:
            # 填充
            pad_size = self.max_vehicles - N
            trajectory = F.pad(trajectory, (0, 0, 0, 0, 0, pad_size), value=0)
            future_traj = F.pad(future_traj, (0, 0, 0, 0, 0, pad_size), value=0)
            intent_labels = F.pad(intent_labels, (0, pad_size), value=-1)  # -1 表示无效
            vehicle_mask = torch.cat([torch.ones(N), torch.zeros(pad_size)], dim=0)
        else:
            # 截断
            trajectory = trajectory[:self.max_vehicles]
            future_traj = future_traj[:self.max_vehicles]
            intent_labels = intent_labels[:self.max_vehicles]
            vehicle_mask = torch.ones(self.max_vehicles)
            N = self.max_vehicles

        # Select knowledge graph for this sample
        kg_src = None
        if self.kg_per_location is not None and location_name in self.kg_per_location:
            kg_src = self.kg_per_location[location_name]
        else:
            kg_src = self.kg
        if kg_src is None:
            raise KeyError("No KG found in dataset (expected 'kg' or 'kg_per_location')")

        # 准备知识图数据
        kg_data = {
            'facility_types': torch.from_numpy(kg_src['facility_types']).long(),
            'positions': torch.from_numpy(kg_src['positions']).float(),
            'edge_index': torch.from_numpy(kg_src['edge_index']).long(),
            'edge_types': torch.from_numpy(kg_src['edge_types']).long(),
        }

        return {
            'trajectories': trajectory,  # (max_vehicles, 8, 4)
            'kg_data': kg_data,
            'future_trajectory': future_traj,  # (max_vehicles, 12, 2)
            'intent_labels': intent_labels,  # (max_vehicles,)
            'vehicle_mask': vehicle_mask,  # (max_vehicles,)
            'num_vehicles': N,
            'case_id': sample['case_id'],
            'location_name': location_name,
        }


def collate_fn(batch):
    """自定义 collate 函数，处理变长车辆数"""
    # 获取 batch size
    batch_size = len(batch)
    max_vehicles = batch[0]['trajectories'].shape[0]

    # 堆叠常规数据
    trajectories = torch.stack([item['trajectories'] for item in batch], dim=0)  # (B, N, 8, 4)
    future_trajectory = torch.stack([item['future_trajectory'] for item in batch], dim=0)
    intent_labels = torch.stack([item['intent_labels'] for item in batch], dim=0)
    vehicle_masks = torch.stack([item['vehicle_mask'] for item in batch], dim=0)

    # 方案2A假设同一个 batch 内所有样本属于同一个 location（同一张地图）
    loc0 = batch[0].get('location_name', None)
    for item in batch[1:]:
        if item.get('location_name', None) != loc0:
            raise ValueError(
                "Mixed locations in a single batch. Use a location-aware sampler (scheme 2A) "
                "or implement KG padding/merging for mixed-map batches."
            )

    # 知识图数据：扩展以匹配 batch size
    kg_template = batch[0]['kg_data']
    kg_data = {
        'facility_types': kg_template['facility_types'].unsqueeze(0).expand(batch_size, -1),
        'positions': kg_template['positions'].unsqueeze(0).expand(batch_size, -1, -1),
        'edge_index': kg_template['edge_index'],
        'edge_types': kg_template['edge_types'],
    }

    return {
        'trajectories': trajectories,
        'kg_data': kg_data,
        'future_trajectory': future_trajectory,
        'intent_labels': intent_labels,
        'vehicle_masks': vehicle_masks,
        'num_vehicles': [item['num_vehicles'] for item in batch],
        'case_ids': [item['case_id'] for item in batch],
        'location_names': [item.get('location_name', None) for item in batch],
    }


# 测试代码
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    data_path = './digir_data/interaction_digir.pkl'

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        print("Please run: python prepare_interaction_for_digir.py")
    else:
        # 测试数据集
        dataset = InteractionDatasetForDIGIR(data_path, split='train')

        print(f"\nConfig: {dataset.config}")
        print(f"KG nodes: {dataset.kg['num_nodes']}")
        print(f"KG edges: {dataset.kg['num_edges']}")

        # 测试 DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        print("\nTesting dataloader...")
        batch = next(iter(dataloader))

        print(f"Batch trajectories shape: {batch['trajectories'].shape}")  # (4, N, 8, 4)
        print(f"Batch future shape: {batch['future_trajectory'].shape}")  # (4, N, 12, 2)
        print(f"Batch intent labels: {batch['intent_labels'].shape}")
        print(f"Batch vehicle masks: {batch['vehicle_masks'].shape}")
        print(f"KG facility types shape: {batch['kg_data']['facility_types'].shape}")

        print("\n✓ Data loading successful!")
