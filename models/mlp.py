import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron for per-pixel multi-label classification.
    Returns logits (no sigmoid). Use BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        n_features: int = 60,
        n_classes: int = 6,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = [nn.BatchNorm1d(n_features)]
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
