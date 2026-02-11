
import torch.nn as nn

from .ResNet_1d import resnet18

class SimSiamResNet18_1d(nn.Module):
    def __init__(self, in_channel=1, out_channel=16):
        super().__init__()

        # backbone: resnet18 but we will use forward_features()
        self.backbone = resnet18(in_channel=in_channel, out_channel=out_channel)  # out_channel here is unused if we bypass fc

        feat_dim = 512  # for resnet18 basic block

        # projector (like SimSiam)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channel),
            nn.BatchNorm1d(out_channel),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(out_channel, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, out_channel),
        )

    def forward(self, x1, x2):
        f = self.backbone.forward_features

        h = self.projector
        p = self.predictor

        z1 = h(f(x1))
        z2 = h(f(x2))

        p1 = p(z1)
        p2 = p(z2)

        return z1, z2, p1, p2
