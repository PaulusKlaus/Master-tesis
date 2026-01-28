import torch
import torch.nn as nn


def mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )



class Encoder(nn.Module):
    def __init__(self, in_channel: int = 1, latent_dim: int = 16):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = latent_dim

        # If unput is 512, first expand to 1024, or skip it
        self.expand = mlp_block(512, 1024)

        self.trunk = nn.Sequential(
            mlp_block(1024, 1024),
            mlp_block(1024, 512),
            mlp_block(512, 256),
            mlp_block(256, 128),
            mlp_block(128, 64),
        )
        self.to_latent = nn.Linear(64, latent_dim)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support [B, D] or [B, 1, D] (or any shape that flattens to D)
        x = self.flatten(x)
        d = x.size(1)


        if d == 512: # x.shape[2] is the feature size (Batch ,Channel, Length)
            x = self.expand(x) # -> [B, 1024]

        x = self.trunk(x)
        z = self.to_latent(x)

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, out_channel=1024):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_channel= out_channel

        self.net = nn.Sequential(
            mlp_block(latent_dim, 64),
            mlp_block(64, 128),
            mlp_block(128, 256),
            mlp_block(256, 512),
            mlp_block(512, 1024),
        )

        self.to_out = nn.Linear(1024, out_channel)


    def forward(self, z:torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        x = self.to_out(x)
        return x


class Classifie(nn.Module):
    def __init__(self, latent_dim: int = 16, num_classes: int = 10):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        label = self.fc(z)
        return label
