import torch
import torch.nn as nn


def mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )


class Classifier(nn.Module):
    def __init__(self, latent_dim: int, classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(32, latent_dim // 2)  # a bit bigger than latent//4 helps sometimes

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)