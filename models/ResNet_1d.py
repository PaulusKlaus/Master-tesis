# This code is based on the implementation of the ResNet (Residual Network) from Pytorch: 
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
# designed my Microsoft: 
# https://arxiv.org/abs/1512.03385
# adopted for the 1D timeseries data input, like vibration signal

import torch
import torch.nn as nn


class ResidualBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # skip connection
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes* self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=16, zero_init_residual=False):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatt = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        _make_layer makes up one Residual Block 
        
        :param block: Can be either 
            - ResidualBlock: inplanes-> planes -> planes * expantion (1)
            - Bottleneck(with expantion): inplanes-> planes -> planes * expantion (4)
        :param blocks: Number of residual blocks
        """
        downsample = None
        # Making sure that the size of the input is the same as the output of each residual block  
        # by downsampling 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # First conv layer with maxpooling 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks with skip connection and downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Avarage pooling and Flattening 
        x = self.avgpool(x)
        x = self.flatt(x)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(ResidualBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(ResidualBasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

# make model (adjust in_channel/out_channel to your use case)
model = resnet18(in_channel=1, out_channel=10)
model.eval()

# Dummy input: (batch, channels, length)
# length must be >= 32 as your comment says; use 256 to be safe
x = torch.randn(4, 1, 256)

with torch.no_grad():
    y = model(x)

print("input:", x.shape)
print("output:", y.shape)

assert y.shape == (4, 10)
print("✅ Forward pass OK")