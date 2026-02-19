"""
Diffusion Policy for manipulation.

Based on Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
https://github.com/real-stanford/diffusion_policy

Architecture (matching the paper):
- Visual encoder: ResNet18 (ImageNet pretrained) -> hidden_dim features per image
- Observation encoder: MLP that fuses visual + proprio across obs_horizon frames
- Noise prediction: ConditionalUnet1D — 1D conv UNet with FiLM conditioning
  from timestep + observation embedding. Skip connections across temporal scales.
- Diffusion: DDPM with cosine beta schedule

Key dimensions:
- Observation: 11D (pose 7D + joints 4D)
- Action: 11D (pose 7D + joints 4D)
- Action horizon: 16 timesteps predicted per chunk
- Obs horizon: 2 past frames conditioned on
"""

import math
from typing import Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDPMScheduler

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


@dataclass
class PolicyConfig:
    """Configuration for diffusion policy."""

    # Observation
    obs_dim: int = 11  # 7 (pose) + 4 (joints)
    image_size: int = 224

    # Action
    action_dim: int = 11  # 7 (pose) + 4 (joints)
    action_horizon: int = 16  # Predict this many future steps
    obs_horizon: int = 2  # Condition on this many past obs

    # Diffusion
    num_diffusion_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 4  # Kept for config compat (UNet uses fixed 2 blocks/level)

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    freeze_backbone: bool = True


# ============================================================
# ConditionalUnet1D building blocks
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device).float() * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """1D conv residual block with FiLM conditioning.

    Two conv blocks with a FiLM modulation (scale + shift) injected after
    the first block. Skip connection with optional channel projection.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), cond: (B, cond_dim)."""
        h = self.blocks[0](x)
        # FiLM: scale and shift from conditioning
        scale, bias = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + bias
        h = self.blocks[1](h)
        return h + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D UNet with FiLM conditioning for denoising action sequences.

    Follows the architecture from Chi et al. (Diffusion Policy):
    - Encoder: conv residual blocks with strided downsampling
    - Mid: residual blocks at lowest resolution
    - Decoder: residual blocks with transposed conv upsampling + skip connections
    - Conditioning: timestep + observation embedding via FiLM at every block

    Designed for short sequences (action_horizon=16).

    Architecture with down_dims=(256, 512, 1024), action_horizon=16:
        Input:  (B, 11, 16)  -- action_dim channels, 16 timesteps
        Proj:   (B, 256, 16)
        Enc L0: (B, 256, 16), skip saved, downsample -> (B, 512, 8)
        Enc L1: (B, 512, 8),  skip saved, downsample -> (B, 1024, 4)
        Enc L2: (B, 1024, 4), skip saved
        Mid:    (B, 1024, 4)
        Dec L2: cat(mid, skip) -> (B, 2048, 4) -> (B, 1024, 4), upsample -> (B, 512, 8)
        Dec L1: cat(up, skip)  -> (B, 1024, 8) -> (B, 512, 8),  upsample -> (B, 256, 16)
        Dec L0: cat(up, skip)  -> (B, 512, 16) -> (B, 256, 16)
        Output: (B, 11, 16)
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        n_levels = len(down_dims)

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Input projection: action_dim -> down_dims[0]
        self.input_conv = nn.Conv1d(input_dim, down_dims[0], 1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_levels):
            self.encoder_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    down_dims[i], down_dims[i], cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(
                    down_dims[i], down_dims[i], cond_dim, kernel_size, n_groups),
            ]))
            if i < n_levels - 1:
                self.downsamples.append(
                    nn.Conv1d(down_dims[i], down_dims[i + 1], 3, stride=2, padding=1)
                )

        # Mid block
        self.mid_blocks = nn.ModuleList([
            ConditionalResidualBlock1D(
                down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(
                down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups),
        ])

        # Decoder (mirror of encoder)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            # First block takes 2x channels (concat with skip connection)
            self.decoder_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    down_dims[i] * 2, down_dims[i], cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(
                    down_dims[i], down_dims[i], cond_dim, kernel_size, n_groups),
            ]))
            if i > 0:
                self.upsamples.append(
                    nn.ConvTranspose1d(
                        down_dims[i], down_dims[i - 1], 4, stride=2, padding=1)
                )

        # Output projection: down_dims[0] -> input_dim
        self.output_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, T, input_dim) noisy actions
            timestep: (B,) diffusion timesteps
            global_cond: (B, global_cond_dim) observation conditioning

        Returns:
            (B, T, input_dim) predicted noise
        """
        x = self.input_conv(sample.transpose(1, 2))  # (B, ch[0], T)

        # Conditioning: timestep + observation
        t_emb = self.timestep_encoder(timestep)
        cond = torch.cat([t_emb, global_cond], dim=-1)

        # Encoder
        skips = []
        ds_idx = 0
        for blocks in self.encoder_blocks:
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            if ds_idx < len(self.downsamples):
                x = self.downsamples[ds_idx](x)
                ds_idx += 1

        # Mid
        for block in self.mid_blocks:
            x = block(x, cond)

        # Decoder
        us_idx = 0
        for blocks in self.decoder_blocks:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond)
            if us_idx < len(self.upsamples):
                x = self.upsamples[us_idx](x)
                us_idx += 1

        return self.output_conv(x).transpose(1, 2)  # (B, T, input_dim)


# ============================================================
# DiffusionPolicy
# ============================================================

class DiffusionPolicy(nn.Module):
    """
    Diffusion-based visuomotor policy.

    Predicts action_horizon future actions conditioned on obs_horizon past observations.

    Components:
    - visual_encoder: ResNet18 (pretrained) or simple CNN fallback
    - obs_encoder: MLP fusing visual features + proprio across obs_horizon
    - noise_pred_net: ConditionalUnet1D for iterative denoising
    - noise_scheduler: DDPM with cosine schedule
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_steps,
            beta_schedule=config.beta_schedule,
            prediction_type="epsilon",
        )

        # Visual encoder
        if TORCHVISION_AVAILABLE:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except (AttributeError, TypeError):
                resnet = models.resnet18(pretrained=True)
            resnet.fc = nn.Linear(512, config.hidden_dim)
            if config.freeze_backbone:
                for name, param in resnet.named_parameters():
                    if not name.startswith("fc."):
                        param.requires_grad_(False)
            self.visual_encoder = resnet
        else:
            # Fallback: simple CNN (no pretrained features)
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, config.hidden_dim),
            )

        # Observation encoder: fuses visual + proprio across obs_horizon
        obs_input_dim = config.hidden_dim + config.obs_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim * config.obs_horizon, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Noise prediction: ConditionalUnet1D
        down_dims = (config.hidden_dim, config.hidden_dim * 2, config.hidden_dim * 4)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.hidden_dim,
            diffusion_step_embed_dim=config.hidden_dim,
            down_dims=down_dims,
        )

    def encode_obs(
        self,
        images: torch.Tensor,  # (B, T, C, H, W)
        proprio: torch.Tensor,  # (B, T, obs_dim)
    ) -> torch.Tensor:
        """Encode observation history into conditioning vector (B, hidden_dim)."""
        B, T = images.shape[:2]

        # Encode each image through ResNet/CNN
        img_flat = images.view(B * T, *images.shape[2:])
        img_features = self.visual_encoder(img_flat)  # (B*T, hidden_dim)
        img_features = img_features.view(B, T, -1)

        # Concatenate visual + proprio, flatten across time
        obs = torch.cat([img_features, proprio], dim=-1)
        obs_flat = obs.view(B, -1)

        return self.obs_encoder(obs_flat)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            images: (B, obs_horizon, 3, H, W) observation images
            proprio: (B, obs_horizon, 11) proprioception
            actions: (B, action_horizon, 11) ground truth action chunk

        Returns:
            {"loss": scalar} MSE loss on noise prediction
        """
        B = images.shape[0]
        device = images.device

        # Encode observations
        obs_cond = self.encode_obs(images, proprio)

        # Sample noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device, dtype=torch.long
        )

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # Predict noise with UNet
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, obs_cond)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        return {"loss": loss}

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict actions via iterative denoising (DDPM sampling).

        Args:
            images: (B, obs_horizon, 3, H, W)
            proprio: (B, obs_horizon, 11)

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        B = images.shape[0]
        device = images.device

        # Encode observations
        obs_cond = self.encode_obs(images, proprio)

        # Start from pure noise
        actions = torch.randn(
            B, self.config.action_horizon, self.config.action_dim,
            device=device
        )

        # Iterative denoising
        self.noise_scheduler.set_timesteps(self.config.num_diffusion_steps)

        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(actions, timesteps, obs_cond)
            actions = self.noise_scheduler.step(
                noise_pred, t, actions
            ).prev_sample

        return actions
