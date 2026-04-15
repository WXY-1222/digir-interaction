"""
DIGIR Training Script for INTERACTION Dataset
"""
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIGIR_ROOT = os.environ.get("DIGIR_ROOT", os.path.join(PROJECT_ROOT, "digir"))
if not os.path.exists(DEFAULT_DIGIR_ROOT):
    raise FileNotFoundError(
        f"DIGIR root not found: {DEFAULT_DIGIR_ROOT}. "
        "Put digir.py at <interaction>/digir/models/digir.py or set DIGIR_ROOT."
    )
if DEFAULT_DIGIR_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_DIGIR_ROOT)

from models.digir import DIGIR
from interaction_dataset_for_digir import InteractionDatasetForDIGIR, collate_fn


def train_epoch(model, dataloader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

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

        # 数据归一化
        ref_point = trajectories[:, :, -1:, :2].clone()
        trajectories_norm = trajectories.clone()
        trajectories_norm[:, :, :, :2] -= ref_point
        future_traj_norm = future_traj - ref_point

        optimizer.zero_grad()
        outputs = model(
            trajectories_norm,
            kg_data,
            future_traj=future_traj_norm,
            mode='train',
            vehicle_masks=vehicle_masks,
        )

        # 应用车辆掩码计算损失
        losses, loss = model.compute_losses(outputs, future_traj_norm, intent_labels, vehicle_masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    """简化评估"""
    model.eval()
    all_ade = []
    all_fde = []
    all_intent_acc = []

    batch_count = 0
    for batch in tqdm(dataloader, desc="Evaluating", total=min(max_batches, len(dataloader))):
        if batch_count >= max_batches:
            break
        batch_count += 1

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

        # 归一化
        ref_point = trajectories[:, :, -1:, :2].clone()
        trajectories_norm = trajectories.clone()
        trajectories_norm[:, :, :, :2] -= ref_point
        future_traj_norm = future_traj - ref_point

        # 生成预测
        pred = model.generate(trajectories_norm, kg_data, num_points=12, num_samples=1,
                              sampling="ddim", step=10, bestof=False)

        while pred.dim() > 4:
            pred = pred.squeeze(1)

        # 计算误差（只对有效车辆）
        valid_mask = vehicle_masks.bool()
        if valid_mask.any():
            pred_valid = pred[valid_mask]
            future_valid = future_traj_norm[valid_mask]

            ade = torch.mean(torch.norm(pred_valid - future_valid, dim=-1)).item()
            fde = torch.mean(torch.norm(pred_valid[:, -1, :] - future_valid[:, -1, :], dim=-1)).item()
            all_ade.append(ade)
            all_fde.append(fde)

        # 意图准确率
        outputs = model(trajectories_norm, kg_data, mode='eval')
        intent_pred = outputs['intent_logits'].argmax(dim=-1)

        valid_intent = (intent_labels >= 0) & valid_mask
        if valid_intent.any():
            acc = ((intent_pred == intent_labels) & valid_intent).float().sum() / valid_intent.sum()
            all_intent_acc.append(acc.item())

    return {
        'ADE': np.mean(all_ade) if all_ade else 0,
        'FDE': np.mean(all_fde) if all_fde else 0,
        'IntentAcc': np.mean(all_intent_acc) if all_intent_acc else 0,
    }


def main():
    config = {
        'd_model': 64,
        'd_prior': 64,
        'hist_len': 8,
        'prediction_horizon': 12,
        'num_intent_classes': 4,
        'num_facility_types': 10,
        'traj_enc_layers': 2,
        'graph_enc_layers': 2,
        'scene_tf_layers': 2,
        'v2v_layers': 2,
        'diffusion_tf_layers': 2,
        'num_heads': 2,
        'dropout': 0.1,
        'elementwise_gate': True,
        'diffusion_steps': 20,
        'beta_1': 1e-4,
        'beta_T': 5e-2,
        'lambda_fine': 1.0,
        'lambda_coarse': 0.5,
        'lambda_cross': 0.1,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    data_path = './digir_data/interaction_digir.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data not found. Run: python prepare_interaction_for_digir.py")
        return

    train_dataset = InteractionDatasetForDIGIR(data_path, split='train')
    val_dataset = InteractionDatasetForDIGIR(data_path, split='val')

    # 使用子集快速验证
    train_subset = torch.utils.data.Subset(train_dataset, range(min(5000, len(train_dataset))))

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_subset)}, Val: {len(val_dataset)}")

    # 创建模型
    model = DIGIR(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, device, max_batches=10)
        print(f"Val ADE: {metrics['ADE']:.3f}m, FDE: {metrics['FDE']:.3f}m, IA: {metrics['IntentAcc']:.2%}")

    # 保存
    torch.save({'model': model.state_dict(), 'config': config}, './digir_interaction.pt')
    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
