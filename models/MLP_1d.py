import torch
import torch.nn as nn


def mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )

class MLP(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):

        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.expand = mlp_block(512, 1024)

        self.trunk = nn.Sequential(
            mlp_block(1024, 1024),
            mlp_block(1024, 512),
            mlp_block(512, 256),
            mlp_block(256, 128),
            mlp_block(128, 64),
        )

        self.flatten = nn.Flatten()
        self.to_out = nn.Sequential(
            nn.Linear(64, out_channel))

    def forward(self, x):

        # Support [B, D] or [B, 1, D] (or any shape that flattens to D)
        x = self.flatten(x)
        d = x.size(1)

        if d == 512: # x.shape[2] is the feature size (Batch ,Channel, Length)
            x = self.expand(x) # -> [B, 1024]

        x = self.trunk(x)
        out = self.to_out(x)

        return out

