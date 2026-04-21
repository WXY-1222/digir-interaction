"""
DIGIR variant: Trajectory-as-Language with VQ-style token decoding.

Key idea:
- Reuse dual-granularity fused condition X^t as prompt.
- Predict future trajectory as a token sequence (autoregressive decoder).
- Decode tokens via a learnable codebook back to continuous points.

This file intentionally does NOT overwrite the original DIGIR model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Trajectory language model built on top of the existing DIGIR backbone.

    Preserved from DIGIR:
    - scene encoding / intent mapping / interaction / gated fusion
    Replaced:
    - diffusion head -> token autoregressive decoder + codebook trajectory decoding
    """

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size = int(config.get("traj_vocab_size", 1024))
        self.chunk_size = int(config.get("traj_chunk_size", 4))
        self.max_token_steps = int(config.get("max_token_steps", 64))
        self.token_layers = int(config.get("token_layers", 4))
        self.token_heads = int(config.get("token_heads", 4))
        self.token_ffn = int(config.get("token_ffn_dim", self.d_model * 4))
        self.token_dropout = float(config.get("token_dropout", config.get("dropout", 0.1)))
        self.token_temperature = float(config.get("token_temperature", 1.0))

        self.lambda_token_ce = float(config.get("lambda_token_ce", 1.0))
        self.lambda_token_recon = float(config.get("lambda_token_recon", 0.5))

        self.code_dim = self.chunk_size * 2
        self.bos_id = self.vocab_size  # dedicated BOS id after vocab ids [0, vocab_size-1]
        self.total_token_emb = self.vocab_size + 1

        # Learnable VQ-style codebook (token -> trajectory chunk prototype).
        self.codebook = nn.Parameter(torch.randn(self.vocab_size, self.code_dim) * 0.02)

        # Prompt + token decoder
        self.prompt_proj = nn.Linear(self.d_model, self.d_model)
        self.token_embed = nn.Embedding(self.total_token_emb, self.d_model)
        self.pos_embed = nn.Embedding(self.max_token_steps + 2, self.d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.token_heads,
            dim_feedforward=self.token_ffn,
            dropout=self.token_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.token_decoder = nn.TransformerEncoder(dec_layer, num_layers=self.token_layers)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        # Disable diffusion usage in this variant.
        self.diffusion = None

    def _required_chunks(self, num_points: int) -> int:
        return int(math.ceil(float(num_points) / float(self.chunk_size)))

    def _pad_future(self, future: torch.Tensor):
        """
        Pad future trajectory to chunk boundary by repeating last point.
        Args:
            future: (B, N, T, 2)
        Returns:
            padded: (B, N, T_pad, 2)
            orig_t: original T
        """
        b, n, t, c = future.shape
        s = self._required_chunks(t)
        t_pad = s * self.chunk_size
        if t_pad == t:
            return future, t
        pad_steps = t_pad - t
        tail = future[:, :, -1:, :].expand(b, n, pad_steps, c)
        return torch.cat([future, tail], dim=2), t

    def _traj_to_token_vectors(self, future: torch.Tensor):
        """
        Convert trajectory to chunk vectors.
        future: (B, N, T, 2)
        return:
            chunk_vec: (B, N, S, chunk_size*2)
            orig_t: original T before internal padding
        """
        future_pad, orig_t = self._pad_future(future)
        b, n, t_pad, _ = future_pad.shape
        s = t_pad // self.chunk_size
        chunks = future_pad.view(b, n, s, self.chunk_size, 2).reshape(b, n, s, self.code_dim)
        return chunks, orig_t

    def _vector_to_tokens(self, chunk_vec: torch.Tensor):
        """
        Nearest-neighbor quantization to token ids.
        chunk_vec: (B, N, S, D)
        return token_ids: (B, N, S)
        """
        b, n, s, d = chunk_vec.shape
        flat = chunk_vec.reshape(-1, d)  # (B*N*S, D)
        # squared L2 distance
        code = self.codebook  # (V, D)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + code.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ code.t()
        )  # (BNS, V)
        ids = dist.argmin(dim=-1).view(b, n, s)
        return ids

    def _decode_tokens_to_traj(self, token_ids: torch.Tensor, num_points: int):
        """
        token_ids: (..., S)
        return traj: (..., T, 2), cropped to num_points
        """
        vec = F.embedding(token_ids, self.codebook)  # (..., S, D)
        shape = list(vec.shape[:-1]) + [self.chunk_size, 2]
        vec = vec.view(*shape)  # (..., S, chunk, 2)
        t = vec.shape[-3] * self.chunk_size
        traj = vec.reshape(*vec.shape[:-3], t, 2)
        return traj[..., :num_points, :]

    def _decode_soft_to_traj(self, logits: torch.Tensor, num_points: int):
        """
        logits: (B, N, S, V)
        return expected trajectory under token probabilities: (B, N, T, 2)
        """
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(
            probs, nan=1.0 / float(self.vocab_size), posinf=1.0 / float(self.vocab_size), neginf=1.0 / float(self.vocab_size)
        )
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # (B, N, S, V) @ (V, D) -> (B, N, S, D)
        exp_vec = torch.matmul(probs, self.codebook)
        b, n, s, _ = exp_vec.shape
        exp_vec = exp_vec.view(b, n, s, self.chunk_size, 2)
        traj = exp_vec.reshape(b, n, s * self.chunk_size, 2)
        return traj[:, :, :num_points, :]

    @staticmethod
    def _causal_mask(length: int, device: torch.device):
        """
        Upper-triangular causal mask for TransformerEncoder.
        """
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _teacher_force_logits(self, prompt: torch.Tensor, target_tokens: torch.Tensor):
        """
        Teacher-forcing decoder logits.
        Args:
            prompt: (BN, d_model)
            target_tokens: (BN, S)
        Returns:
            logits: (BN, S, V)
        """
        bn, s = target_tokens.shape
        device = target_tokens.device

        # decoder inputs: [BOS, t0, ..., t_{S-2}] (length S)
        in_tok = torch.full((bn, 1), self.bos_id, dtype=torch.long, device=device)
        if s > 1:
            in_tok = torch.cat([in_tok, target_tokens[:, :-1]], dim=1)

        tok_emb = self.token_embed(in_tok)  # (BN, S, d)
        prompt_emb = self.prompt_proj(prompt).unsqueeze(1)  # (BN, 1, d)
        seq = torch.cat([prompt_emb, tok_emb], dim=1)  # (BN, S+1, d)

        max_len = seq.shape[1]
        if max_len > self.max_token_steps + 2:
            raise RuntimeError(
                f"Token sequence too long ({max_len}). Increase max_token_steps "
                f"(current={self.max_token_steps})."
            )
        pos = torch.arange(max_len, device=device).unsqueeze(0).expand(bn, -1)
        seq = seq + self.pos_embed(pos)

        mask = self._causal_mask(max_len, device=device)
        hid = self.token_decoder(seq, mask=mask)

        # Predict S targets from the S token positions (exclude prompt position).
        logits = self.lm_head(hid[:, 1:, :])  # (BN, S, V)
        return logits

    def _autoregressive_sample(self, prompt: torch.Tensor, num_steps: int, bestof: bool):
        """
        Sample token sequence autoregressively.
        Args:
            prompt: (BN, d_model)
        Returns:
            tokens: (BN, num_steps)
        """
        bn = prompt.shape[0]
        device = prompt.device

        generated = []
        cur = torch.full((bn, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(num_steps):
            tok_emb = self.token_embed(cur)  # (BN, L, d)
            prompt_emb = self.prompt_proj(prompt).unsqueeze(1)  # (BN,1,d)
            seq = torch.cat([prompt_emb, tok_emb], dim=1)
            max_len = seq.shape[1]
            if max_len > self.max_token_steps + 2:
                raise RuntimeError(
                    f"Token sequence too long ({max_len}) in generate(). "
                    f"Increase max_token_steps (current={self.max_token_steps})."
                )
            pos = torch.arange(max_len, device=device).unsqueeze(0).expand(bn, -1)
            seq = seq + self.pos_embed(pos)

            mask = self._causal_mask(max_len, device=device)
            hid = self.token_decoder(seq, mask=mask)
            logits = self.lm_head(hid[:, -1, :])  # (BN, V)

            if bestof:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                temp = max(self.token_temperature, 1e-4)
                probs = torch.softmax(logits / temp, dim=-1)
                probs = torch.nan_to_num(
                    probs,
                    nan=1.0 / float(self.vocab_size),
                    posinf=1.0 / float(self.vocab_size),
                    neginf=1.0 / float(self.vocab_size),
                )
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                nxt = torch.multinomial(probs, num_samples=1)

            generated.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)

        return torch.cat(generated, dim=1) if generated else torch.empty((bn, 0), dtype=torch.long, device=device)

    def _build_prompt(self, trajectories, kg_data, vehicle_masks=None):
        """
        Produce fused prompt X^t from dual-granularity DIGIR backbone.
        Returns:
            outputs_base: dict with intent/gate fields used by training/eval
            prompt: (B, N, d_model)
        """
        scene_intent, local_contexts, motion_summaries = self.encode_scene(
            trajectories, kg_data, vehicle_masks=vehicle_masks
        )
        intent_priors, intent_logits = self.cross_granularity_mapping(
            scene_intent, local_contexts, vehicle_masks=vehicle_masks
        )
        fused_conditions, interaction_features, gate_weights = self.agent_level_modeling(
            motion_summaries, intent_priors, vehicle_masks=vehicle_masks
        )

        outputs_base = {
            "scene_intent": scene_intent,
            "local_contexts": local_contexts,
            "motion_summaries": motion_summaries,
            "intent_priors": intent_priors,
            "intent_logits": intent_logits,
            "interaction_features": interaction_features,
            "fused_conditions": fused_conditions,
            "gate_weights": gate_weights,
        }
        return outputs_base, fused_conditions

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs["diffusion_loss"] is replaced by token-language training loss
        - outputs["traj_pred_train"] is decoded from predicted tokens
        """
        outputs, prompt = self._build_prompt(trajectories, kg_data, vehicle_masks=vehicle_masks)

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape
            chunks, _ = self._traj_to_token_vectors(future_traj)  # (B,N,S,D)
            token_gt = self._vector_to_tokens(chunks)  # (B,N,S)

            s = token_gt.shape[-1]
            prompt_flat = prompt.reshape(b * n, self.d_model)
            token_flat = token_gt.reshape(b * n, s)

            logits_flat = self._teacher_force_logits(prompt_flat, token_flat)  # (BN,S,V)
            logits = logits_flat.view(b, n, s, self.vocab_size)

            if vehicle_masks is None:
                valid = torch.ones((b, n), dtype=torch.bool, device=future_traj.device)
            else:
                valid = vehicle_masks.bool()

            valid_flat = valid.reshape(-1)
            if valid_flat.any():
                ce = F.cross_entropy(
                    logits_flat[valid_flat].reshape(-1, self.vocab_size),
                    token_flat[valid_flat].reshape(-1),
                    reduction="mean",
                )
            else:
                ce = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)

            traj_soft = self._decode_soft_to_traj(logits, num_points=t)
            if valid.any():
                recon = F.smooth_l1_loss(traj_soft[valid], future_traj[valid], reduction="mean")
            else:
                recon = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)

            token_pred = logits.argmax(dim=-1)  # (B,N,S)
            traj_pred = self._decode_tokens_to_traj(token_pred, num_points=t)  # (B,N,T,2)

            token_loss = self.lambda_token_ce * ce + self.lambda_token_recon * recon

            outputs["token_logits"] = logits
            outputs["token_gt"] = token_gt
            outputs["token_pred"] = token_pred
            outputs["traj_pred_train"] = traj_pred
            outputs["diffusion_loss"] = token_loss
            outputs["loss_token_ce"] = ce
            outputs["loss_token_recon"] = recon

        return outputs

    def generate(
        self,
        trajectories,
        kg_data,
        num_points=12,
        num_samples=20,
        sampling="ddim",
        step=20,
        bestof=True,
        vehicle_masks=None,
    ):
        """
        Interface-compatible generation:
        - num_samples == 1: (B, N, T, 2)
        - num_samples > 1: (S, B, N, T, 2)
        """
        del sampling, step  # not used in token decoder

        _, prompt = self._build_prompt(trajectories, kg_data, vehicle_masks=vehicle_masks)
        b, n, _ = prompt.shape
        s = self._required_chunks(int(num_points))
        prompt_flat = prompt.reshape(b * n, self.d_model)

        def sample_once(use_best):
            tokens_flat = self._autoregressive_sample(prompt_flat, num_steps=s, bestof=use_best)
            tokens = tokens_flat.view(b, n, s)
            traj = self._decode_tokens_to_traj(tokens, num_points=int(num_points))
            return traj

        if int(num_samples) <= 1:
            return sample_once(use_best=bestof)

        preds = []
        # First sample can be best-of for stability if requested.
        preds.append(sample_once(use_best=bestof))
        for _ in range(int(num_samples) - 1):
            preds.append(sample_once(use_best=False))
        return torch.stack(preds, dim=0)

