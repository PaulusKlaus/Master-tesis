
import torch.nn as nn

class SimSiam(nn.Module):
    """
    Generic 1D Simplified SimSiam with increasing chanel size 
    """
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 16,
        conv_channels=None,  
        r: str = "val"
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [16, 32, 64, 128, 256, 512]

        self.in_channel = in_channel
        self.output_channel = out_channel
        self.retur = r

        layers = []

        # ---- Backbone ----
        prev_c = in_channel
        for c in conv_channels:
            layers.append(nn.Conv1d(prev_c, c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c

        # length-agnostic feature
        layers.append(nn.AdaptiveAvgPool1d(1))  # -> (N, C, 1)
        layers.append(nn.Flatten())             # -> (N, C)


        # ---- Projector (MLP) ----
        hidden_projector  = prev_c // 2
        layers.append(nn.Linear(prev_c, hidden_projector))
        layers.append(nn.BatchNorm1d(hidden_projector))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_projector, out_channel))
        layers.append(nn.BatchNorm1d(out_channel))

        self.encoder = nn.Sequential(*layers)

        hidden_predictor = int(out_channel * 2)

        # ---- Predictor (MLP) ----
        self.predictor = nn.Sequential(
            nn.Linear(out_channel, hidden_predictor),
            nn.BatchNorm1d(hidden_predictor),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_predictor, out_channel),
        )

    def forward(self, x_1, x_2):
        f, h = self.encoder, self.predictor

        # Encoder step -> Features to be used for AD and classification
        z_1, z_2 = f(x_1), f(x_2)

        # Predictor step -> Needed for similarity and loss computation
        p_1, p_2 = h(z_1), h(z_2)
        
        return z_1, z_2, p_1, p_2
