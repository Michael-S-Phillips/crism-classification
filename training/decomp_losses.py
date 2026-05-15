"""
Composite loss for the signal/noise decomposition encoder.

  L_total = L_cls
          + λ_recon  · L_recon
          + λ_eps    · L_eps_reg
          + λ_T      · L_T_reg
          + λ_b      · L_b_reg
          + λ_smooth · L_smooth

L_cls is the existing AsymmetricLoss on (logits, labels, sample_weights,
optional class_weights). The other terms enforce the physical decomposition
structure documented in
docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.losses import AsymmetricLoss


class DecompositionLoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_eps: float = 0.1,
        lambda_T: float = 0.01,
        lambda_b: float = 0.01,
        lambda_smooth: float = 0.001,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_eps = lambda_eps
        self.lambda_T = lambda_T
        self.lambda_b = lambda_b
        self.lambda_smooth = lambda_smooth
        self.cls_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, n_tokens, n_bands) or (B, P, P, n_bands)
        logits: torch.Tensor,       # (B, n_classes)
        labels: torch.Tensor,       # (B, n_classes)
        weights: torch.Tensor,      # (B,)
        s_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        T_hat: torch.Tensor,        # (B, n_bands)
        b_hat: torch.Tensor,        # (B, n_bands)
        eps_hat: torch.Tensor,      # (B, n_tokens, n_bands)
        x_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # If x came in as a 4D patch, flatten to (B, n_tokens, n_bands) for recon.
        if x.dim() == 4:
            B, P, P2, n_bands = x.shape
            assert P == P2, "patches must be square"
            x_flat = x.reshape(B, P * P2, n_bands)
        else:
            x_flat = x

        # 1. Classification loss
        cls = self.cls_loss(
            logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 2. Reconstruction loss (MSE on valid pixels).
        # NODATA pixels were zeroed upstream (see data/extract_pixels.py).
        # Pixels with |x|>1 are pathological; mask them out so they don't dominate.
        valid_mask = (x_flat.abs() < 1.0).float()
        sq_err = (x_hat - x_flat) ** 2 * valid_mask
        recon = sq_err.sum() / (valid_mask.sum() + 1e-8)

        # 3. Residual magnitude regularizer
        eps_reg = (eps_hat ** 2).mean()

        # 4. Atmospheric priors
        T_reg = ((T_hat - 1.0) ** 2).mean()
        b_reg = (b_hat ** 2).mean()

        # 5. Spatial smoothness on signal — TV on the (B, P, P, n_bands) layout.
        # Reshape s_hat to (B, P, P, n_bands), compute horizontal + vertical
        # first-differences, average over everything.
        B, N, nb = s_hat.shape
        P = int(N ** 0.5)
        s_spatial = s_hat.view(B, P, P, nb)
        dv = (s_spatial[:, 1:, :, :] - s_spatial[:, :-1, :, :]).abs()
        dh = (s_spatial[:, :, 1:, :] - s_spatial[:, :, :-1, :]).abs()
        smooth = (dv.mean() + dh.mean()) * 0.5

        total = (
            cls
            + self.lambda_recon * recon
            + self.lambda_eps * eps_reg
            + self.lambda_T * T_reg
            + self.lambda_b * b_reg
            + self.lambda_smooth * smooth
        )
        components = {
            'cls': cls, 'recon': recon,
            'eps_reg': eps_reg, 'T_reg': T_reg, 'b_reg': b_reg,
            'smooth': smooth,
        }
        return total, components
