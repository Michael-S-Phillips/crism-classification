"""
Composite loss for the adversarial signal/noise decomposition encoder (v2).

  L_total = L_cls
          + λ_recon  · L_recon
          + L_adv               (gradient-reversed for encoder side via GRL in the model)
          + λ_smooth · L_smooth

L_cls is ASL on the classifier logits. L_recon is MSE of (s_hat + n_hat)
against the input. L_adv is ASL on the discriminator logits — the GRL
inside the model handles the encoder-side sign flip, so the loss itself
just treats L_adv as a standard classification term. L_smooth is TV on
s_hat.

Spec: docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.losses import AsymmetricLoss


class AdversarialDecompositionLoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 10.0,
        lambda_smooth: float = 0.001,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.cls_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )
        self.adv_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, n_tokens, n_bands) or (B, P, P, n_bands)
        logits: torch.Tensor,       # (B, n_classes)
        labels: torch.Tensor,       # (B, n_classes)
        weights: torch.Tensor,      # (B,)
        s_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        n_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        x_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        disc_logits: torch.Tensor,  # (B, n_classes)
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() == 4:
            B, P, P2, n_bands = x.shape
            assert P == P2
            x_flat = x.reshape(B, P * P2, n_bands)
        else:
            x_flat = x

        # 1. Classification
        cls = self.cls_loss(
            logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 2. Reconstruction MSE on valid pixels
        valid_mask = (x_flat.abs() < 1.0).float()
        sq_err = (x_hat - x_flat) ** 2 * valid_mask
        recon = sq_err.sum() / (valid_mask.sum() + 1e-8)

        # 3. Adversarial loss — discriminator predicts class from (GRL'd) n_emb.
        # GRL inside the model flipped the encoder-side gradient; here we just
        # compute the standard classification loss.
        adv = self.adv_loss(
            disc_logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 4. Spatial smoothness on signal (TV penalty over the 7×7 layout)
        B, N, nb = s_hat.shape
        P = int(N ** 0.5)
        s_spatial = s_hat.view(B, P, P, nb)
        dv = (s_spatial[:, 1:, :, :] - s_spatial[:, :-1, :, :]).abs()
        dh = (s_spatial[:, :, 1:, :] - s_spatial[:, :, :-1, :]).abs()
        smooth = (dv.mean() + dh.mean()) * 0.5

        total = (
            cls
            + self.lambda_recon * recon
            + adv
            + self.lambda_smooth * smooth
        )
        components = {
            'cls': cls, 'recon': recon, 'adv': adv, 'smooth': smooth,
        }
        return total, components
