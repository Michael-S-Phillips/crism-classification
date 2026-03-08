import torch
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
