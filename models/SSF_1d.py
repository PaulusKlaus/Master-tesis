import torch.nn as nn



class ConvBlock1D(nn.Module):
    """
    One encoder block (CB): Conv1d -> BatchNorm1d -> ReLU -> Pool1d
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 256,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: str = "max",
        pool_kernel: int = 2,
        pool_stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm1d(out_ch, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        if pool == "max":
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)
        elif pool == "avg":
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        elif pool == None:
            self.pool = nn.Identity()
        else:
            raise ValueError("pool must be 'max', 'avg' or None")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SSF(nn.Module):
    """
    Self-Supervised Simple Siamese (SSF) network for 1D timeseries signals.

    Defaults:
      - 10 Conv blocks
      - Each block keeps 256 channels
      - Conv1d(kernel=3, stride=1, padding=1)
      - Pool1d(kernel=2, stride=2)
      - Final AdaptiveAvgPool1d -> Flatten -> 256-D latent code
    """
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 16,
        num_blocks: int = 10,
        hidden_channels: int = 256,
    ):
        super().__init__()

        blocks = []

        # First block changes channel count from in_channels -> hidden_channels
        blocks.append(
            ConvBlock1D(
                in_ch=in_channel,
                out_ch=hidden_channels
            )
        )

        # Remaining blocks keep hidden_channels
        for _ in range(num_blocks - 2):
            blocks.append(
                ConvBlock1D(
                    in_ch=hidden_channels,
                    out_ch=hidden_channels
                )
            )
        blocks.append(
            ConvBlock1D(
                in_ch=hidden_channels,
                out_ch=out_channel,
                pool= "avg"
            )
        )

        self.encoder = nn.Sequential(*blocks,
                                    nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten()  )
        
        self.predictor = nn.Sequential(
            nn.Linear(out_channel,8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8,out_channel))

    def forward( self, x_1, x_2): 

        f, h = self.encoder, self.predictor

        # Encoder step
        z_1, z_2 = f(x_1), f(x_2)

        # Predictor step 
        p_1, p_2 = h(z_1), h(z_2)

        return z_1, z_2, p_1, p_2



