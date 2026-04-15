"""
DIGIR: Dual-Granularity Intent Rollout
Conditional Diffusion Model - Intent-conditioned trajectory generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VarianceSchedule(nn.Module):
    """Variance schedule for diffusion process"""
    def __init__(self, num_steps=100, beta_1=1e-4, beta_T=5e-2, mode='linear', cosine_s=8e-3):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        return torch.randint(1, self.num_steps + 1, (batch_size,))

    def get_sigmas(self, t, flexibility):
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class ConcatSquashLinear(nn.Module):
    """Linear layer with concatenated context (time + condition)"""
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class ConditionalDenoisingNet(nn.Module):
    """
    Denoising network conditioned on dual-granularity features (Section 4.4.3)
    Predicts noise given current noisy trajectory, timestep, and condition
    """
    def __init__(self, point_dim=2, context_dim=128, tf_layer=4, residual=False):
        super().__init__()
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.residual = residual

        # Positional encoding for trajectory points
        self.pos_emb = nn.Linear(point_dim, 128)

        # Time embedding (sinusoidal)
        self.time_embed_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # Context embedding (dual-granularity condition)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )

        # Combined context dimension: time_emb + context
        combined_ctx_dim = self.time_embed_dim + context_dim

        # Trajectory encoder
        self.traj_encoder = nn.Sequential(
            ConcatSquashLinear(point_dim, 128, combined_ctx_dim),
            nn.ReLU(),
            ConcatSquashLinear(128, 256, combined_ctx_dim),
            nn.ReLU()
        )

        # Transformer for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layer)

        # Output decoder
        self.decoder = nn.Sequential(
            ConcatSquashLinear(256, 128, combined_ctx_dim),
            nn.ReLU(),
            ConcatSquashLinear(128, point_dim, combined_ctx_dim)
        )

    def forward(self, x, beta, context):
        """
        Args:
            x: (batch_size, num_points, point_dim) - noisy trajectory
            beta: (batch_size,) - diffusion timestep
            context: (batch_size, context_dim) - fused condition X^t
        Returns:
            noise_pred: (batch_size, num_points, point_dim) - predicted noise
        """
        batch_size, num_points, _ = x.shape

        # Time embedding
        beta_norm = beta.view(batch_size, 1) / 100.0  # Normalize (B, 1)
        time_emb = self.time_mlp(beta_norm)  # (B, time_embed_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, num_points, -1)  # (B, T, time_embed_dim)

        # Context embedding
        ctx_emb = self.context_mlp(context).unsqueeze(1)  # (B, 1, context_dim)
        ctx_emb = ctx_emb.expand(-1, num_points, -1)  # (B, T, context_dim)

        # Combine context
        combined_ctx = torch.cat([time_emb, ctx_emb], dim=-1)  # (B, T, time_embed_dim + context_dim)

        # Encode trajectory
        h = self.traj_encoder[0](combined_ctx, x)
        h = self.traj_encoder[1](h)
        h = self.traj_encoder[2](combined_ctx, h)
        h = self.traj_encoder[3](h)

        # Apply transformer
        h = self.transformer(h)

        # Decode to noise prediction
        out = self.decoder[0](combined_ctx, h)
        out = self.decoder[1](out)
        out = self.decoder[2](combined_ctx, out)

        if self.residual:
            out = x + out

        return out


class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Probabilistic Model (Section 4.4.3)
    Generates trajectories conditioned on dual-granularity features
    """
    def __init__(self, net, num_steps=100, beta_1=1e-4, beta_T=5e-2):
        super().__init__()
        self.net = net
        self.var_sched = VarianceSchedule(num_steps, beta_1, beta_T)

    def get_loss(self, x_0, context, t=None, mask=None):
        """
        Training loss: MSE between predicted and actual noise
        Args:
            x_0: (batch_size, num_points, point_dim) - ground truth future trajectory
            context: (batch_size, context_dim) - fused condition
            t: optional specific timesteps
            mask: optional (batch_size,) float/bool — 1 = include in mean (e.g. valid agents).
                  Required when padded agents have bogus x_0 (e.g. scene coords with zero-padded future).
        """
        batch_size = x_0.size(0)
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].to(x_0.device)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to(x_0.device)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(x_0.device)

        # Add noise
        eps = torch.randn_like(x_0)
        x_t = c0 * x_0 + c1 * eps

        # Predict noise
        t_batch = t.float().to(x_0.device)
        eps_pred = self.net(x_t, beta=t_batch, context=context)

        per_sample = F.mse_loss(eps_pred, eps, reduction="none").mean(dim=(1, 2))
        if mask is not None:
            m = mask.to(x_0.device).float().clamp(0.0, 1.0)
            denom = m.sum().clamp(min=1.0)
            return (per_sample * m).sum() / denom
        return per_sample.mean()

    def sample(self, num_points, context, num_samples=1, flexibility=0.0,
               sampling="ddpm", step=100, ret_traj=False):
        """
        Generate trajectory samples
        Args:
            num_points: number of trajectory points to generate
            context: (batch_size, context_dim) - condition
            num_samples: number of samples per input
            flexibility: sampling flexibility
            sampling: "ddpm" or "ddim"
            step: stride for sampling (100 for DDPM, less for DDIM)
            ret_traj: return full diffusion trajectory
        """
        traj_list = []
        batch_size = context.size(0)

        for _ in range(num_samples):
            # Start from pure noise
            x_T = torch.randn(batch_size, num_points, 2).to(context.device)
            traj = {self.var_sched.num_steps: x_T}

            stride = step

            # Reverse diffusion
            for t in range(self.var_sched.num_steps, 0, -stride):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[max(t - stride, 0)]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                # Predict noise
                beta_batch = torch.tensor([t] * batch_size, dtype=torch.float32).to(context.device)
                eps_pred = self.net(traj[t], beta=beta_batch, context=context)

                if sampling == "ddpm":
                    x_next = c0 * (traj[t] - c1 * eps_pred) + sigma * z
                elif sampling == "ddim":
                    # DDIM sampling for faster inference
                    x0_pred = (traj[t] - eps_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
                    x_next = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1 - alpha_bar_next) * eps_pred
                else:
                    raise ValueError(f"Unknown sampling method: {sampling}")

                traj[t - stride] = x_next.detach()
                if not ret_traj:
                    traj[t] = traj[t].cpu()

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)


class MultiModalConditionalDiffusion(ConditionalDiffusion):
    """
    Extension: Multi-modal trajectory generation with mode selection
    """
    def __init__(self, net, num_steps=100, beta_1=1e-4, beta_T=5e-2, num_modes=20):
        super().__init__(net, num_steps, beta_1, beta_T)
        self.num_modes = num_modes

    def sample_best_of_k(self, num_points, context, scoring_fn, num_modes=None):
        """
        Sample K modes and select best according to scoring function
        Args:
            scoring_fn: function that takes (trajectory, context) and returns score
        """
        num_modes = num_modes or self.num_modes

        # Generate multiple samples
        samples = self.sample(
            num_points, context,
            num_samples=num_modes,
            sampling="ddim",
            step=20  # Faster sampling
        )  # (K, B, T, 2)

        # Score each sample
        scores = []
        for k in range(num_modes):
            score = scoring_fn(samples[k], context)
            scores.append(score)

        scores = torch.stack(scores)  # (K, B)
        best_idx = scores.argmax(dim=0)  # (B,)

        # Select best samples
        batch_size = context.size(0)
        best_samples = samples[best_idx, torch.arange(batch_size)]

        return best_samples, scores
