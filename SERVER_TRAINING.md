# Server Multi-GPU Training (DIGIR + INTERACTION)

`train_digir_full.py` now supports:

- Multi-GPU training with `torchrun` (DDP, for example 8x A100).
- Code/data/output path separation via `--digir_root`, `--data_root`, and `--save_root`.

## 1) 8-GPU command (equivalent to your local command)

```bash
cd /path/to/interaction

torchrun --standalone --nproc_per_node=8 train_digir_full.py \
  --digir_root /path/to/DIGIR \
  --data_root /path/to/interaction_data \
  --save_root /path/to/interaction_runs \
  --data interaction_digir_all_12loc_h8_f12.pkl \
  --save gate_force_intent.pt \
  --coord_frame per_agent \
  --batch_by_location \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --train_subset 5000 \
  --eval_batches 0 \
  --k 5 \
  --seed 42 \
  --lambda_rule 1e-3 \
  --map_margin 3.0 \
  --ablate_gate force_intent \
  --log_gate_stats
```

If `--data` and `--save` are relative, they are resolved under `--data_root` and `--save_root`.

## 2) Gate threshold/ratio sweep (`--gate_fixed_ratio`)

`--gate_fixed_ratio` means:

- `0.0`: intent branch only
- `1.0`: interaction branch only
- values in between: linear fusion

```bash
cd /path/to/interaction

for r in 0.0 0.2 0.4 0.6 0.8 1.0; do
  torchrun --standalone --nproc_per_node=8 train_digir_full.py \
    --digir_root /path/to/DIGIR \
    --data_root /path/to/interaction_data \
    --save_root /path/to/interaction_runs \
    --data interaction_digir_all_12loc_h8_f12.pkl \
    --save gate_ratio_${r}.pt \
    --coord_frame per_agent \
    --batch_by_location \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --train_subset 5000 \
    --eval_batches 0 \
    --k 5 \
    --seed 42 \
    --lambda_rule 1e-3 \
    --map_margin 3.0 \
    --gate_fixed_ratio ${r} \
    --log_gate_stats
done
```

You can also run discrete baselines with `--ablate_gate force_intent` and `--ablate_gate force_interaction`.
