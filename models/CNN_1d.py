import torch.nn as nn 


class CNN_1d(nn.Module):
    """
    Gereric 1D CNN with maxpooling after each layer


    - conv_channel can be used to change the depth of the model
    """

    def __init__(
            self,
            in_channel: int = 1,
            out_channel: int = 16, 
            conv_channels = None 
    ):
        super.__init__()

        self.in_channel = in_channel
        self.output_channel = out_channel

        layers = []

        if 

        prev_in = in_channel 
        for c in conv_channels:
            layers.append(nn.Conv1d(prev_in,c,kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_in = c