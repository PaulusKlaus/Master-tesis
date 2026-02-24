import torch
import torch.nn as nn


def mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )


class Classifier(nn.Module):
    def __init__(self, latent_dim: int = 16, classes: int = 32):
        super().__init__()
        # ensure integer hidden size
        hidden = int(latent_dim // 4)


        self.layer_1 = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True)
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(hidden, out_features=classes )
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.layer_1(x)
        z = self.layer_2(x)
        
        return z