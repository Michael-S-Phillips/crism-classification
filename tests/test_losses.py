import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.losses import WeightedBCEWithLogitsLoss


def test_loss_accepts_pos_weight():
    """Loss should accept optional pos_weight tensor."""
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(8, 6)
    targets = torch.randint(0, 2, (8, 6)).float()
    weights = torch.ones(8)
    pos_weight = torch.tensor([1.0, 1.0, 1.0, 4.0, 6.0, 2.0])
    loss_val = loss_fn(logits, targets, weights, pos_weight=pos_weight)
    assert loss_val.item() > 0


def test_pos_weight_increases_loss_for_rare_class():
    """pos_weight should increase loss when rare class positive is missed."""
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.full((4, 6), -3.0)   # predicts all negative
    targets = torch.zeros(4, 6)
    targets[:, 4] = 1.0                  # class 4 is positive (rare)
    weights = torch.ones(4)
    loss_no_pw = loss_fn(logits, targets, weights)
    pos_weight = torch.ones(6)
    pos_weight[4] = 10.0
    loss_with_pw = loss_fn(logits, targets, weights, pos_weight=pos_weight)
    assert loss_with_pw > loss_no_pw


def test_focal_loss_down_weights_easy_examples():
    from training.losses import FocalBCEWithLogitsLoss, WeightedBCEWithLogitsLoss
    # Very confident correct prediction — focal should give much lower loss than BCE
    logits = torch.tensor([[5.0, -5.0, 5.0, -5.0, 5.0, -5.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    weights = torch.ones(1)
    bce_loss = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    focal_loss = FocalBCEWithLogitsLoss(gamma=2.0)(logits, targets, weights)
    assert focal_loss < bce_loss, "Focal loss should be lower for easy (confident correct) examples"


def test_focal_loss_same_as_bce_when_gamma_zero():
    from training.losses import FocalBCEWithLogitsLoss, WeightedBCEWithLogitsLoss
    torch.manual_seed(0)
    logits = torch.randn(8, 6)
    targets = (torch.randn(8, 6) > 0).float()
    weights = torch.ones(8)
    bce = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    focal0 = FocalBCEWithLogitsLoss(gamma=0.0)(logits, targets, weights)
    assert abs(bce.item() - focal0.item()) < 1e-5


def test_asl_output_is_scalar():
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()
    weights = torch.ones(8)
    val = loss_fn(logits, targets, weights)
    assert val.shape == (), f"Expected scalar, got {val.shape}"
    assert val.item() > 0


def test_asl_accepts_pos_weight_kwarg():
    """ASL must accept pos_weight= kwarg for API compatibility with training loop."""
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss()
    logits = torch.randn(4, 5)
    targets = torch.zeros(4, 5)
    weights = torch.ones(4)
    val = loss_fn(logits, targets, weights, pos_weight=torch.ones(5))
    assert val.item() >= 0


def test_asl_clip_zeroes_easy_negatives():
    """With clip=0.5, confident negatives (p << 0.5) should incur near-zero loss."""
    from training.losses import AsymmetricLoss
    loss_fn = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.5)
    logits = torch.full((4, 5), -10.0)
    targets = torch.zeros(4, 5)
    weights = torch.ones(4)
    val = loss_fn(logits, targets, weights)
    assert val.item() < 0.01, f"Expected near-zero loss for confident negatives, got {val.item()}"


def test_asl_higher_gamma_neg_suppresses_negatives():
    """Higher gamma_neg should reduce loss on easy negatives more than lower gamma_neg."""
    from training.losses import AsymmetricLoss
    torch.manual_seed(0)
    logits = torch.full((8, 5), -2.0)
    targets = torch.zeros(8, 5)
    weights = torch.ones(8)
    loss_low  = AsymmetricLoss(gamma_neg=1.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    loss_high = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    assert loss_high < loss_low, "Higher gamma_neg should down-weight easy negatives more"


def test_asl_gamma_zero_clip_zero_equals_bce():
    """ASL with gamma_neg=0, gamma_pos=0, clip=0 should equal WeightedBCE."""
    from training.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
    torch.manual_seed(42)
    logits = torch.randn(8, 5)
    targets = (torch.randn(8, 5) > 0).float()
    weights = torch.ones(8)
    bce = WeightedBCEWithLogitsLoss()(logits, targets, weights)
    asl = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.0)(logits, targets, weights)
    assert abs(bce.item() - asl.item()) < 1e-4, f"Expected BCE≈ASL(γ=0,clip=0): {bce.item()} vs {asl.item()}"


def test_build_class_balanced_weights():
    import numpy as np
    import pandas as pd
    from training.train_torch import build_class_balanced_weights
    n = 1000
    labels = np.zeros((n, 6))
    labels[:10, 4] = 1.0   # 10 plagioclase positives (very rare)
    labels[:500, 0] = 1.0  # 500 olivine_t1 positives (common)
    df = pd.DataFrame(labels, columns=['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other'])
    df['confidence_weight'] = 1.0
    weights = build_class_balanced_weights(df)
    plag_idx = np.where(labels[:, 4] > 0.4)[0]
    non_plag_idx = np.where(labels[:, 4] <= 0.4)[0]
    assert weights[plag_idx].mean() > weights[non_plag_idx].mean() * 5
