import torch 
from torchinfo import summary

from CNN_1d import CNN_1d
from SimSiam_1d import SimSiam
from SSF_1d import SSF
from ResNet_1d import resnet18

model_CNN = CNN_1d(in_channel=1, out_channel=10)

x_1 = torch.randn(8,1,1024)  # (Batch, channel, signal_length)
x_2 = torch.randn(8, 1, 1024)


# ---- CNN: forward + summary ----
y = model_CNN(x_1)               # <- fixed: no weird (.venv) suffix
print("Output shape of CNN: ", y.shape)

print("\nCNN summary:")
summary(model_CNN, input_size=(8, 1, 1024))


# ---- SimSiam  ----
model_SimSiam = SimSiam(in_channel =1, out_channel = 16)
z1, z2, p1, p2 = model_SimSiam(x_1, x_2)

print("z1:", z1.shape)  # expected (B, out_channel)
print("z2:", z2.shape)
print("p1:", p1.shape)  # expected (B, out_channel)
print("p2:", p2.shape)
summary(model_SimSiam, input_size=((8, 1, 1024),(8, 1, 1024)))

# ---- SSF (Simplified SimSiam) ----
model_SSF = SSF(in_channel =1, out_channel = 16)
z1, z2, p1, p2 = model_SSF(x_1, x_2)

print("z1:", z1.shape)  # expected (B, out_channel)
print("z2:", z2.shape)
print("p1:", p1.shape)  # expected (B, out_channel)
print("p2:", p2.shape)
summary(model_SSF, input_size=((8, 1, 1024),(8, 1, 1024)))


# ---- ResNet18 ----

model_resnet18 = resnet18(in_channel = 1, out_channel = 16)
y = model_resnet18(x_1)

print("ResNet18 output shape: ", y.shape)
summary(model_resnet18, input_size=(8,1,1024))