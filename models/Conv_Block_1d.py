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