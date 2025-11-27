# -*- coding:utf-8 -*-
"""
End-to-End Semantic Communication Training Script with Flow Matching Modulation.
Includes support for Straight-Through Estimator (STE) quantization and Decision Annealing.
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.channels import Channels
# ==============================================================================
# 1. Global Configuration & Constants
# ==============================================================================

try:
    ROOT = Path(__file__).resolve().parents[2]
    # Assuming config.json exists in the current directory
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except (FileNotFoundError, IndexError):
    print("Warning: Config file or path setup failed. Using default placeholders.")
    CONFIG = {
        "repo_dir": "./",
        "backbone_name": "dinov3_vitl14",
        "weight_path": {"backbone": {}, "head": {"image_classification": ""}},
        "data": {
            "dataset_dir": "/path/to/imagenet",
            "image_size": 224,
            "batch_size": 256,
            "num_worker": 8
        },
        "DEVICE": "cuda:0"
    }

DEVICE = torch.device(CONFIG.get("DEVICE", "cuda:0"))
VIZ_DIR = Path("./viz_IQ")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Paths for weights (Placeholders based on original code)
REPO_DIR = CONFIG["repo_dir"]
BACKBONE_NAME = CONFIG["backbone_name"]
# Handle dictionary lookups safely
BACKBONE_DIR = CONFIG["weight_path"].get("backbone", {}).get(BACKBONE_NAME, "")
HEAD_DIR = CONFIG["weight_path"].get("head", {}).get("image_classification", "")


# ==============================================================================
# 2. Math & Signal Processing Utilities
# ==============================================================================

class Channels:
    """Simulates communication channels (e.g., AWGN)."""
    def __init__(self, device: torch.device):
        self.device = device

    def AWGN(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Adds Additive White Gaussian Noise to the signal."""
        signal_power = (signal ** 2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal, device=self.device) * noise_std
        return signal + noise


def make_square_qam(M: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Generates standard Square QAM constellation points."""
    K = int(math.isqrt(M))
    if K * K != M:
        raise ValueError("M must be a perfect square (4, 16, 64, ...)")
    levels = torch.arange(-(K - 1), K, 2, dtype=dtype, device=device)
    I, Q = torch.meshgrid(levels, levels, indexing='xy')
    return torch.stack([I.flatten(), Q.flatten()], dim=1)


def normalize_constellation(points: torch.Tensor, mode="unit_avg_energy", eps=1e-12) -> torch.Tensor:
    """Normalizes constellation points to have unit average energy."""
    if points.dim() != 2:
        raise ValueError("Points shape must be [M, 2]")
    if mode == "unit_avg_energy":
        avg_E = points.pow(2).sum(dim=1).mean()
        scale = 1.0 / torch.sqrt(avg_E + eps)
        return points * scale
    raise ValueError(f"Unknown normalization mode: {mode}")


def sample_qam_targets_1d(M: int, B: int, d_i: int, d_q: int,
                          device: torch.device, sigma_scale: float = 1/6):
    """Samples target QAM points with Gaussian jitter for training targets."""
    const2d = normalize_constellation(make_square_qam(M, device=device))
    xs = torch.unique(const2d[:, 0], sorted=True)
    ys = torch.unique(const2d[:, 1], sorted=True)

    gap_x = torch.min(xs[1:] - xs[:-1]) if xs.numel() > 1 else torch.tensor(1.0, device=device)
    gap_y = torch.min(ys[1:] - ys[:-1]) if ys.numel() > 1 else torch.tensor(1.0, device=device)
    
    idx_i = torch.randint(0, xs.numel(), (B, d_i), device=device)
    idx_q = torch.randint(0, ys.numel(), (B, d_q), device=device)

    x1_i = xs[idx_i] + (sigma_scale * gap_x) * torch.randn(B, d_i, device=device)
    x1_q = ys[idx_q] + (sigma_scale * gap_y) * torch.randn(B, d_q, device=device)
    return x1_i, x1_q, idx_i, idx_q


def add_quant_noise(x: torch.Tensor, delta: torch.Tensor, strength: float = 1.0, mode: str = "uniform") -> torch.Tensor:
    """Adds quantization noise to simulate discrete levels during training."""
    if mode == "uniform":
        # U(-a, a) where a = 0.5 * strength * delta
        a = 0.5 * strength * float(delta)
        noise = (torch.rand_like(x) - 0.5) * (2.0 * a)
    elif mode == "gaussian":
        # Match variance of uniform distribution
        std = (strength * float(delta)) / math.sqrt(12.0)
        noise = torch.randn_like(x) * std
    else:
        raise ValueError("mode must be 'uniform' or 'gaussian'")
    return x + noise


# ==============================================================================
# 3. Data Loading & Wrappers
# ==============================================================================

class Classifier(nn.Module):
    """Wrapper to combine a backbone and a linear classification head."""
    def __init__(self, backbone: nn.Module, linear_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        cls_token = feats["x_norm_clstoken"]
        patch_tokens = feats["x_norm_patchtokens"].mean(dim=1)
        linear_input = torch.cat([cls_token, patch_tokens], dim=1)
        return self.linear_head(linear_input)
        
def compute_sigma_scale(global_step: Optional[int],
                        anneal_steps: int,
                        k: float) -> float:
    """
    根据 3σ 规则和退火进度计算当前的 sigma_scale。
    - 初始: 3σ = Δ/2 => σ = Δ/6 => sigma_scale_start = 1/6
    - 末期: σ = Δ/k   => sigma_scale_end   = 1/k
    """
    sigma_scale_start = 1.0 / 6.0
    sigma_scale_end = 1.0 / k

    if global_step is None or anneal_steps <= 0:
        return sigma_scale_start

    progress = min(float(global_step) / float(anneal_steps), 1.0)
    sigma_scale = sigma_scale_start + (sigma_scale_end - sigma_scale_start) * progress
    return float(sigma_scale)

def build_imagenet_loaders(root: str, img_size: int = 224,
                           batch_size: int = 256, num_workers: int = 8
                           ) -> Tuple[DataLoader, DataLoader]:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ==============================================================================
# 4. Model Components: Flow Matching Modulator
# ==============================================================================

class SymbolFlowHead(nn.Module):
    """Conditioned projection head for the flow vector field."""
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowMatchingModulator(nn.Module):
    """
    Main Modulation Module using Conditional Flow Matching (CFM).
    Maps continuous features (t=0) to discrete QAM constellations (t=1).
    """
    def __init__(self, d: int,
                 device: str = "cuda:0",
                 uniform_range: float = 1.0,
                 emb_dim: int = 8,
                 hidden_dim: int = 64,
                 mod_orders: List[int] = (4, 16, 64, 256),
                 order_emb_dim: int = 8
                 ):
        super().__init__()
        self.d_total = d
        self.d_i = d // 2
        self.d_q = d - self.d_i
        self.uniform_range = float(uniform_range)

        self.mod_orders = list(mod_orders)
        self.num_orders = len(self.mod_orders)
        self.order_embed = nn.Embedding(self.num_orders, order_emb_dim)

        # Statistics for whitening
        self.register_buffer("mu", torch.zeros(self.d_total))
        self.register_buffer("std", torch.ones(self.d_total))
        self.momentum = 0.01

        # Constellation reference
        self.default_M = self.mod_orders[1]
        const2d = make_square_qam(self.default_M, device=device)
        self.register_buffer("const2d", const2d)

        # Embeddings for symbol conditioning
        num_levels = int(math.isqrt(max(mod_orders)))
        self.i_symbol_embed = nn.Embedding(num_embeddings=num_levels, embedding_dim=emb_dim)
        self.q_symbol_embed = nn.Embedding(num_embeddings=num_levels, embedding_dim=emb_dim)

        # Vector field heads
        in_dim = 1 + 1 + emb_dim + order_emb_dim # [x_t, t, embedding]
        self.i_head = SymbolFlowHead(in_dim, hidden_dim)
        self.q_head = SymbolFlowHead(in_dim, hidden_dim)
        
        self.use_tanh_head = True
        self.i_out_scale = nn.Parameter(torch.tensor(2.5))
        self.q_out_scale = nn.Parameter(torch.tensor(2.5))

    @torch.no_grad()
    def update_running_stats(self, z: torch.Tensor):
        batch_mu = z.mean(dim=0)
        batch_std = z.std(dim=0).clamp_min(1e-6)
        self.mu = (1 - self.momentum) * self.mu + self.momentum * batch_mu
        self.std = (1 - self.momentum) * self.std + self.momentum * batch_std

    def whiten(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu) / self.std

    def dewhiten(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mu

    def forward_u(self,
                  x_t_i: torch.Tensor,
                  x_t_q: torch.Tensor,
                  t: torch.Tensor,
                  idx_i: torch.Tensor,
                  idx_q: torch.Tensor,
                  order_id: torch.Tensor
                  ):
        """Calculates the velocity field u_t(x|symbol)."""
        B, d_i = x_t_i.shape
        _, d_q = x_t_q.shape

        if t.dim() == 1: t = t.view(-1, 1)

        t_i_full = t.repeat(1, d_i)
        t_q_full = t.repeat(1, d_q)

        emb_i_raw = self.i_symbol_embed(idx_i) # (B,d_i,E_sym)
        emb_q_raw = self.q_symbol_embed(idx_q)

        order_e = self.order_embed(order_id)          # (B, Eo)
        order_e_i = order_e.view(B, 1, -1).expand(B, d_i, -1)
        order_e_q = order_e.view(B, 1, -1).expand(B, d_q, -1)

        in_i_cat = torch.cat(
            [x_t_i.unsqueeze(-1), t_i_full.unsqueeze(-1), emb_i_raw, order_e_i],
            dim=-1
        ).reshape(B * d_i, -1)
        in_q_cat = torch.cat(
            [x_t_q.unsqueeze(-1), t_q_full.unsqueeze(-1), emb_q_raw, order_e_q],
            dim=-1
        ).reshape(B * d_q, -1)

        v_i_hat = self.i_head(in_i_cat).view(B, d_i)
        v_q_hat = self.q_head(in_q_cat).view(B, d_q)

        if self.use_tanh_head:
            v_i_hat = torch.tanh(v_i_hat) * self.i_out_scale
            v_q_hat = torch.tanh(v_q_hat) * self.q_out_scale

        return v_i_hat, v_q_hat

    @torch.no_grad()
    def heun_step(self, x_i, x_q, t0, t1, idx_i, idx_q, order_id):
        """Single step of Heun's ODE solver method."""
        dt = (t1 - t0).view(-1, 1)
        v0_i, v0_q = self.forward_u(x_i, x_q, t0, idx_i, idx_q, order_id)
        x_star_i = x_i + dt * v0_i
        x_star_q = x_q + dt * v0_q
        v1_i, v1_q = self.forward_u(x_star_i, x_star_q, t1, idx_i, idx_q, order_id)
        x_new_i = x_i + 0.5 * dt * (v0_i + v1_i)
        x_new_q = x_q + 0.5 * dt * (v0_q + v1_q)
        return x_new_i, x_new_q

    def flow_to_t1(self, x0_i, x0_q, idx_i, idx_q, steps, order_id):
        """Integrate from t=0 (features) to t=1 (constellation)."""
        B = x0_i.size(0)
        t = torch.zeros(B, device=x0_i.device)
        x_i, x_q = x0_i, x0_q
        for s in range(steps):
            t0 = t.clone()
            t1 = torch.full_like(t0, (s + 1) / steps)
            x_i, x_q = self.heun_step(x_i, x_q, t0, t1, idx_i, idx_q, order_id)
            t = t1
        return x_i, x_q
    
    def flow_from_t1_to_t0(self, x1_i, x1_q, idx_i, idx_q, steps, order_id):
        """Integrate backwards from t=1 to t=0 (Demodulation)."""
        B = x1_i.size(0)
        x_i, x_q = x1_i, x1_q
        time_steps = torch.linspace(1, 0, steps + 1, device=x1_i.device)
        for i in range(steps):
            t0 = time_steps[i]
            t1 = time_steps[i + 1]
            # dt is negative automatically
            x_i, x_q = self.heun_step(x_i, x_q, t0.expand(B), t1.expand(B), idx_i, idx_q, order_id)
        return x_i, x_q

    # --- Straight-Through Estimator (STE) for Hard Quantization ---
    class _STEQuantizeConstellation(torch.autograd.Function):
        """Custom autograd function for hard quantization with gradient pass-through."""
        @staticmethod
        def forward(ctx, x_i_in, x_q_in, constellation_points):
            B, d = x_i_in.shape
            # Combine to 2D for distance calculation
            x_2d = torch.stack([x_i_in.reshape(-1), x_q_in.reshape(-1)], dim=1)
            dists = torch.cdist(x_2d, constellation_points, p=2.0)
            nearest_indices = torch.argmin(dists, dim=1)
            quantized_points_2d = constellation_points[nearest_indices]
            
            quantized_i = quantized_points_2d[:, 0].reshape(B, d)
            quantized_q = quantized_points_2d[:, 1].reshape(B, d)
            return quantized_i, quantized_q

        @staticmethod
        def backward(ctx, grad_output_i, grad_output_q):
            # Pass gradients directly to input (STE)
            return grad_output_i, grad_output_q, None

    def ste_quantize_to_constellation(self, x_i_in: torch.Tensor, x_q_in: torch.Tensor, M: int):
        const_norm = normalize_constellation(make_square_qam(M, device=x_i_in.device))
        return self._STEQuantizeConstellation.apply(x_i_in, x_q_in, const_norm)

    @torch.no_grad()
    def infer_and_demodulate(self, x1_i: torch.Tensor, x1_q: torch.Tensor, steps: int = 20, hard_decision: bool = True, order_id: Optional[torch.Tensor] = 16):
        """
        Full inference pipeline: 
        1. (Optional) Hard decision on received symbols.
        2. Reverse flow to reconstruct features.
        """
        device = x1_i.device
        B = x1_i.size(0)

        if M is None:
            M = self.default_M
        if order_id is None:
            order_id = torch.zeros(B, dtype=torch.long, device=device)

        norm_const = normalize_constellation(make_square_qam(M, device=device))
        levels_i = torch.unique(norm_const[:, 0], sorted=True)
        levels_q = torch.unique(norm_const[:, 1], sorted=True)

        idx_i = torch.argmin(torch.abs(x1_i.unsqueeze(2) - levels_i), dim=2)
        idx_q = torch.argmin(torch.abs(x1_q.unsqueeze(2) - levels_q), dim=2)

        demod_input_i, demod_input_q = x1_i, x1_q
        if hard_decision:
            demod_input_i = levels_i[idx_i]
            demod_input_q = levels_q[idx_q]

        x0_i, x0_q = self.flow_from_t1_to_t0(
            demod_input_i, demod_input_q,
            idx_i, idx_q,
            steps=steps,
            order_id=order_id
        )
        z_hat = torch.cat([x0_i, x0_q], dim=1)
        return z_hat, idx_i, idx_q


# ==============================================================================
# 5. Training Engine
# ==============================================================================

def get_levels_from_M(M: int, device):
    const = normalize_constellation(make_square_qam(M, device=device))
    levels_i = torch.unique(const[:, 0], sorted=True)
    levels_q = torch.unique(const[:, 1], sorted=True)
    return levels_i, levels_q

def get_levels_from_mod(modulator: FlowMatchingModulator, device):
    const = normalize_constellation(make_square_qam(modulator.M, device=device))
    levels_i = torch.unique(const[:, 0], sorted=True)
    levels_q = torch.unique(const[:, 1], sorted=True)
    return levels_i, levels_q


def plot_flowed_constellation(x1_i: torch.Tensor,
                              x1_q: torch.Tensor,
                              levels_i: torch.Tensor,
                              levels_q: torch.Tensor,
                              M: int,
                              vis_dir: Path,
                              global_step: int):
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        xi = x1_i.detach().flatten().cpu().numpy()
        xq = x1_q.detach().flatten().cpu().numpy()
        li = levels_i.detach().cpu().numpy()
        lq = levels_q.detach().cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.scatter(xi, xq, s=1, alpha=0.25, label="Flowed symbols")

    for a in li:
        for b in lq:
            plt.scatter(a, b, marker='x')

    plt.axis('equal')
    plt.title(f"Flowed Symbols | M={M} | step={global_step}")
    plt.tight_layout()
    out_path = vis_dir / f"flow_M{M}_step{global_step}.png"
    plt.savefig(out_path)
    plt.close()

def e2e_train_step_quant_noise(
    modulator: FlowMatchingModulator,
    linear_head: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    mod_optimizer: torch.optim.Optimizer,
    cls_optimizer: torch.optim.Optimizer,
    channels: Channels,
    *,
    snr: float = 20.0,
    flow_steps: int = 20,
    cfm_loss_weight: float = 0.5,
    anneal_steps: int = 100_000,   # 退火总步数（可按你总 iter 粗略设）
    sigma_k: float = 12.0,         # 上面说的 k
    p_dd: float = 0.0,
    lambda_sym: float = 0.1,
    quant_noise_strength: float = 1.0,
    quant_noise_mode: str = "uniform",
    symbol_source: str = "random",
    log_vis: bool = False,
    vis_dir: Optional[Path] = None,
    global_step: Optional[int] = None,
) -> Tuple[float, float, float, float, int, float, float]:
    """
    Executes a single E2E training step with quantization noise and decision annealing.
    
    Args:
        symbol_source: "random" (for diversity) or "feature_based" (like VQ-VAE).
        p_dd: Probability of using Decision Annealing (using RX decision for reverse flow).
    """
    modulator.train()
    linear_head.train()

    device = z.device
    B, _ = z.shape
    d_i, d_q = modulator.d_i, modulator.d_q

    order_idx = np.random.randint(len(modulator.mod_orders))
    M_cur = modulator.mod_orders[order_idx]
    order_id = torch.full((B,), order_idx, device=device, dtype=torch.long)

    # 1. Prepare x0 (Whitening)
    x0_white = modulator.whiten(z)
    x0_i, x0_q = x0_white[:, :d_i].contiguous(), x0_white[:, d_i:].contiguous()

    sigma_scale = compute_sigma_scale(global_step, anneal_steps, sigma_k)
    # 2. CFM Loss (Standard Flow Matching Objective - keeps vector field robust)
    with torch.no_grad():
        cfm_x1_i, cfm_x1_q, cfm_idx_i, cfm_idx_q = sample_qam_targets_1d(
            M_cur, B, d_i, d_q, device, sigma_scale
        )
    t = torch.rand(B, device=device).view(-1, 1)
    xt_i = (1 - t) * x0_i + t * cfm_x1_i
    xt_q = (1 - t) * x0_q + t * cfm_x1_q
    
    v_star_i, v_star_q = cfm_x1_i - x0_i, cfm_x1_q - x0_q
    v_hat_i, v_hat_q = modulator.forward_u(xt_i, xt_q, t.squeeze(1), cfm_idx_i, cfm_idx_q, order_id)
    
    cfm_loss = F.mse_loss(v_hat_i, v_star_i) + F.mse_loss(v_hat_q, v_star_q)

    # 3. E2E Forward Pass
    levels_i, levels_q = get_levels_from_M(M_cur, device)
    
    # Select Target Symbols
    with torch.no_grad():
        if symbol_source == "random":
            e2e_idx_i = torch.randint(0, levels_i.numel(), (B, d_i), device=device)
            e2e_idx_q = torch.randint(0, levels_q.numel(), (B, d_q), device=device)
        elif symbol_source == "feature_based":
            # "Attach" to nearest constellation point in x0 space
            dist_i = torch.abs(x0_i.unsqueeze(-1) - levels_i.view(1, 1, -1))
            e2e_idx_i = torch.argmin(dist_i, dim=-1)
            dist_q = torch.abs(x0_q.unsqueeze(-1) - levels_q.view(1, 1, -1))
            e2e_idx_q = torch.argmin(dist_q, dim=-1)
        else:
            raise ValueError(f"Unknown symbol_source: {symbol_source}")

    # Forward Flow to t=1
    x1_i, x1_q = modulator.flow_to_t1(x0_i, x0_q, e2e_idx_i, e2e_idx_q, steps=flow_steps, order_id=order_id)

    with torch.no_grad():
        used_i = torch.unique(e2e_idx_i)
        used_q = torch.unique(e2e_idx_q)
        cov_i = used_i.numel() / levels_i.numel()
        cov_q = used_q.numel() / levels_q.numel()

    if log_vis and vis_dir is not None and global_step is not None:
        plot_flowed_constellation(x1_i, x1_q, levels_i, levels_q, M_cur, vis_dir, global_step)

    # Add Quantization Noise
    delta_i = torch.min(levels_i[1:] - levels_i[:-1]) if levels_i.numel() > 1 else torch.tensor(1.0, device=device)
    delta_q = torch.min(levels_q[1:] - levels_q[:-1]) if levels_q.numel() > 1 else torch.tensor(1.0, device=device)

    x1_noisy_i = add_quant_noise(x1_i, delta_i, strength=quant_noise_strength, mode=quant_noise_mode)
    x1_noisy_q = add_quant_noise(x1_q, delta_q, strength=quant_noise_strength, mode=quant_noise_mode)

    # Channel Simulation
    y1_i = channels.AWGN(x1_noisy_i, snr)
    y1_q = channels.AWGN(x1_noisy_q, snr)

    # Decision Annealing (Determine reverse flow condition)
    with torch.no_grad():
        dec_idx_i = torch.argmin(torch.abs(y1_i.unsqueeze(2) - levels_i), dim=2)
        dec_idx_q = torch.argmin(torch.abs(y1_q.unsqueeze(2) - levels_q), dim=2)

        if p_dd <= 0.0:
            idx_inv_i, idx_inv_q = e2e_idx_i, e2e_idx_q  # Use TX intent
        elif p_dd >= 1.0:
            idx_inv_i, idx_inv_q = dec_idx_i, dec_idx_q  # Use RX blind decision
        else:
            mask_i = (torch.rand_like(e2e_idx_i.float()) < p_dd).bool()
            mask_q = (torch.rand_like(e2e_idx_q.float()) < p_dd).bool()
            idx_inv_i = torch.where(mask_i, dec_idx_i, e2e_idx_i)
            idx_inv_q = torch.where(mask_q, dec_idx_q, e2e_idx_q)

    # Reverse Flow (Demodulation)
    x0_hat_i, x0_hat_q = modulator.flow_from_t1_to_t0(y1_i, y1_q, idx_inv_i, idx_inv_q, flow_steps, order_id)
    z_hat = torch.cat([x0_hat_i, x0_hat_q], dim=1)

    # 4. Losses
    logits = linear_head(z_hat)
    task_loss = F.cross_entropy(logits, labels)

    # Symbol Consistency Loss (Regularization to pull x1 towards target levels)
    target_val_i = levels_i[e2e_idx_i]
    target_val_q = levels_q[e2e_idx_q]
    sym_mse = ((x1_i - target_val_i) ** 2).mean() + ((x1_q - target_val_q) ** 2).mean()

    total_loss = task_loss + cfm_loss_weight * cfm_loss + lambda_sym * sym_mse

    # 5. Optimization
    mod_optimizer.zero_grad()
    cls_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(modulator.parameters(), max_norm=5.0)
    cls_optimizer.step()
    mod_optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()

    return total_loss.item(), cfm_loss.item(), task_loss.item(), acc, M_cur, float(cov_i), float(cov_q)


# ==============================================================================
# 6. Evaluation Logic
# ==============================================================================

def extract_z0(feature_extractor, images, device):
    """Extracts features from backbone depending on the object type."""
    if hasattr(feature_extractor, 'forward_features'):
        feats = feature_extractor.forward_features(images.to(device, non_blocking=True))
    elif hasattr(feature_extractor, 'backbone'):
        feats = feature_extractor.backbone.forward_features(images.to(device, non_blocking=True))
    else:
        # Assuming DINOv3 output structure
        feats = feature_extractor(images.to(device, non_blocking=True))
        
    if isinstance(feats, dict):
        return torch.cat([feats["x_norm_clstoken"], feats["x_norm_patchtokens"].mean(dim=1)], dim=1)
    return feats # Fallback if already tensor

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def evaluate_multi_order_multi_snr(
    feature_extractor,          # Generally the backbone
    modulator: FlowMatchingModulator,
    linear_head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    channels: Channels,
    order_list: List[int],      # e.g., [4, 16, 64, 256]
    snr_list: List[float],      # e.g., [-10, -5, 0, 5, 10, 15, 20, 25]
    flow_steps: int = 50,
    hard_decision: bool = True
) -> Dict[Tuple[int, float], Dict[str, float]]:
    """
    Joint evaluation for multiple modulation orders (M) and SNRs.
    SER is calculated based on 'complex symbols': a symbol is considered correct 
    only if both I and Q components are demodulated correctly.

    Returns:
        results[(M, snr)] = {
            'top1': top1_acc,     # Percentage
            'top5': top5_acc,     # Percentage
            'ser':  ser_sym       # 0~1
        }
    """
    feature_extractor.eval()
    modulator.eval()
    linear_head.eval()

    results: Dict[Tuple[int, float], Dict[str, float]] = {}

    print("\n" + "=" * 80)
    print("Multi-Order & Multi-SNR Evaluation (SER = complex symbol error rate)")
    print("=" * 80)
    print(f"{'M':>6} {'SNR(dB)':>8} {'Top-1(%)':>10} {'Top-5(%)':>10} {'SER_sym':>10}")
    print("-" * 80)

    # Loop over Modulation Orders
    for M_eval in order_list:
        if M_eval not in modulator.mod_orders:
            print(f"[Warning] M={M_eval} not in modulator.mod_orders, skipping.")
            continue
        order_idx = modulator.mod_orders.index(M_eval)

        # Pre-compute constellation levels for current M
        levels_i, levels_q = get_levels_from_M(M_eval, device)

        # Loop over SNRs
        for snr in snr_list:
            top1_sum, top5_sum, total_samples = 0.0, 0.0, 0
            err_sym, total_sym = 0, 0   

            for images, target in val_loader:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                B = images.size(0)

                # 1) Extract features z0
                z0_batch = extract_z0(feature_extractor, images, device)

                # 2) Whiten and split into I/Q
                x0_white = modulator.whiten(z0_batch)
                d_i, d_q = modulator.d_i, modulator.d_q
                x0_i, x0_q = x0_white[:, :d_i], x0_white[:, d_i:]

                # 3) Feature-based symbol assignment (nearest neighbor)
                #    Map features to the closest constellation level indices
                dist_i = torch.abs(x0_i.unsqueeze(-1) - levels_i.view(1, 1, -1))
                idx_i  = torch.argmin(dist_i, dim=-1)

                dist_q = torch.abs(x0_q.unsqueeze(-1) - levels_q.view(1, 1, -1))
                idx_q  = torch.argmin(dist_q, dim=-1)

                # 4) Prepare modulation order condition
                order_id = torch.full((B,), order_idx, device=device, dtype=torch.long)

                # 5) Forward Flow (Tx): t=0 -> t=1
                x1_i, x1_q = modulator.flow_to_t1(x0_i, x0_q, idx_i, idx_q,
                                                  steps=flow_steps, order_id=order_id)

                # 6) Channel simulation (AWGN)
                y1_i = channels.AWGN(x1_i, snr_db=snr)
                y1_q = channels.AWGN(x1_q, snr_db=snr)

                # 7) Blind detection at Rx (Symbol Error Rate calculation)
                rx_idx_i = torch.argmin(
                    torch.abs(y1_i.unsqueeze(2) - levels_i.view(1, 1, -1)), dim=2
                )
                rx_idx_q = torch.argmin(
                    torch.abs(y1_q.unsqueeze(2) - levels_q.view(1, 1, -1)), dim=2
                )

                # Complex symbol error: if either I or Q is wrong, the symbol is wrong
                sym_err_mask = (rx_idx_i != idx_i) | (rx_idx_q != idx_q)
                err_sym  += sym_err_mask.sum().item()
                total_sym += sym_err_mask.numel()

                # 8) Reverse Flow (Demodulation): t=1 -> t=0
                if hard_decision:
                    # Snap to nearest grid point before reversing
                    demod_i = levels_i[rx_idx_i]
                    demod_q = levels_q[rx_idx_q]
                else:
                    # Use raw noisy values
                    demod_i, demod_q = y1_i, y1_q

                x0_hat_i, x0_hat_q = modulator.flow_from_t1_to_t0(
                    demod_i, demod_q,
                    rx_idx_i, rx_idx_q,
                    steps=flow_steps,
                    order_id=order_id
                )
                z_hat = torch.cat([x0_hat_i, x0_hat_q], dim=1)

                # 9) Classification
                logits = linear_head(z_hat)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

                total_samples += B
                top1_sum += acc1.item() * B
                top5_sum += acc5.item() * B

            # --- Aggregate results for current (M, SNR) ---
            top1_final = top1_sum / max(1, total_samples)
            top5_final = top5_sum / max(1, total_samples)
            ser_sym    = err_sym / max(1, total_sym)

            results[(M_eval, snr)] = {
                "top1": top1_final,
                "top5": top5_final,
                "ser":  ser_sym,
            }

            print(f"{M_eval:6d} {snr:8.1f} {top1_final:10.2f} {top5_final:10.2f} {ser_sym:10.4f}")

    print("=" * 80)
    print("Evaluation finished.\n")
    return results


# ==============================================================================
# 7. Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    # --- A. Setup ---
    print(f"Loading backbone {BACKBONE_NAME}...")
    # Loading DINOv3 or similar via Hub
    backbone = torch.hub.load(REPO_DIR, BACKBONE_NAME, source='local', weights=BACKBONE_DIR)
    d_feature = backbone.embed_dim * 2

    print(f"Loading linear head from {PRETRAINED_HEAD_PATH}...")
    linear_head = nn.Linear(d_feature, 1000).to(DEVICE)
    # linear_head.load_state_dict(torch.load(PRETRAINED_HEAD_PATH, map_location=DEVICE))

    # Classifier wrapper handles feature extraction logic
    classifier = Classifier(backbone, linear_head).to(DEVICE).eval()

    order_list = [4, 16, 64, 256]
    snr_list   = [-10, -5, 0, 5, 10, 15, 20, 25]
    
    channels_layer = Channels(DEVICE)
    modulator = FlowMatchingModulator(
        d=d_feature,
        device=DEVICE,
        mod_orders=order_list
    ).to(DEVICE)
    
    # Save directory
    save_root = VIZ_DIR / "gmm" / "e2e"
    save_root.mkdir(parents=True, exist_ok=True)
    flow_vis_dir = save_root / "flow_vis"
    flow_vis_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_imagenet_loaders(
        root=CONFIG["data"]["dataset_dir"],
        img_size=CONFIG["data"]["image_size"],
        batch_size=CONFIG["data"]["batch_size"],
        num_workers=CONFIG["data"]["num_worker"],
    )

    # --- B. Load Pretrained Modulator ---
    # try:
    #     modulator.load_state_dict(torch.load(PRETRAINED_MOD_PATH, map_location=DEVICE))
    #     print(f"Loaded Pretrained Modulator from {PRETRAINED_MOD_PATH}")
    # except FileNotFoundError:
    #     print(f"Error: Pretrained modulator not found at {PRETRAINED_MOD_PATH}")
    #     sys.exit(1)

    # --- C. End-to-End Fine-tuning Loop ---
    
    E2E_NUM_EPOCHS = 1
    
    training_configs = [
        # Strategy: Based on feature attraction + Hard Quantization (Fast convergence)
        {
            'name': 'FeatBased_HardQuant', 
            'symbol_plan': 'uniform', 
            'hard_quant_train': True,
            'symbol_source_train': 'feature_based' 
        },
    ]

    for config in training_configs:
        config_name = config['name']
        print(f"\n=== Starting Training Config: {config_name} ===")

        # Re-init optimizers
        e2e_mod_optimizer = torch.optim.AdamW(modulator.parameters(), lr=5e-4, weight_decay=1e-5)
        e2e_cls_optimizer = torch.optim.AdamW(linear_head.parameters(), lr=5e-4, weight_decay=1e-8)

        for epoch in range(1, E2E_NUM_EPOCHS + 1):
            
            # --- Dynamic Hyperparameters ---
            # Decision Annealing Schedule (Not active in this config snippet, but available)
            p_dd = 0.0 # Keep fixed for now, or ramp up: min(0.8, 0.8 * (epoch/E2E_NUM_EPOCHS))
            
            # Noise Schedule: Ramp up quantization noise strength
            quant_noise_strength = 0.3 + 0.7 * (epoch / E2E_NUM_EPOCHS)
            
            total_loss_accum, total_acc_accum = 0, 0
            
            # --- Training Steps ---
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Ep {epoch}")):
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                # Extract features
                with torch.no_grad():
                    z = extract_z0(classifier.backbone, images, DEVICE)

                global_step = (epoch - 1) * len(train_loader) + i
                log_flag = (i % 100 == 0)

                loss, cfm_l, task_l, acc, M_cur, cov_i, cov_q = e2e_train_step_quant_noise(
                    modulator, linear_head, z, labels,
                    e2e_mod_optimizer, e2e_cls_optimizer, channels_layer,
                    snr=20, flow_steps=20, cfm_loss_weight=0.5,
                    anneal_steps=E2E_NUM_EPOCHS * len(train_loader),
                    sigma_k=8.0,
                    p_dd=p_dd, lambda_sym=0.1,
                    quant_noise_strength=quant_noise_strength,
                    symbol_source=config['symbol_source_train'],
                    log_vis=log_flag,
                    vis_dir=flow_vis_dir,
                    global_step=global_step
                )

                total_loss_accum += loss
                total_acc_accum += acc
                
                if log_flag:
                    tqdm.write(
                        f"[Ep {epoch} | Step {i}] "
                        f"M={M_cur} | CovI={cov_i:.2f}, CovQ={cov_q:.2f} | "
                        f"Loss: {loss:.4f} (CFM: {cfm_l:.3f}, Task: {task_l:.3f}) | Acc: {acc:.3f}"
                    )

            # --- Checkpointing ---
            model_save_path = save_root / f"modulator_{config_name}_epoch_{epoch}_final_step2.pth"
            head_save_path = save_root / f"linear_{config_name}_epoch_{epoch}_final_step2.pth"
            torch.save(modulator.state_dict(), model_save_path)
            torch.save(linear_head.state_dict(), head_save_path)

            # --- Validation ---
            results = evaluate_multi_order_multi_snr(
                feature_extractor=backbone,        # 或者 classifier.backbone 也行
                modulator=modulator,
                linear_head=linear_head,
                val_loader=val_loader,
                device=DEVICE,
                channels=channels_layer,
                order_list=[16, 256],
                snr_list=[-5, 20],
                flow_steps=50,
                hard_decision=True
            )

        print("\nTraining Complete. Strating the final test...\n")
        results = evaluate_multi_order_multi_snr(
            feature_extractor=backbone,        # 或者 classifier.backbone 也行
            modulator=modulator,
            linear_head=linear_head,
            val_loader=val_loader,
            device=DEVICE,
            channels=channels_layer,
            order_list=order_list,
            snr_list=snr_list,
            flow_steps=50,
            hard_decision=True
        )
