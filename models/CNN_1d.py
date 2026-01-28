import torch.nn as nn 


class CNN_1d(nn.Module):
    """
    Gereric 1D CNN 


    - conv_channel can be used to change the depth of the model
    """

    def __init__(
            self,
            in_channel: int = 1,
            out_channel: int = 16, 
            conv_channels = None 
    ):
        super().__init__()

        self.in_channel = in_channel
        self.output_channel = out_channel
    
        # ----   The Encoder of the CNN   -----
        if conv_channels == None:
            # Default convolutional structure
            conv_channels = [16, 32, 64, 128]

        layers = []
        prev_in = in_channel 

        for i, c in enumerate(conv_channels):
            layers.append(nn.Conv1d(prev_in,c,kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))

            if i != len(conv_channels)-1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                print("Last layer in the convolutional block without maxpooling")

            prev_in = c

        layers.append(nn.AdaptiveAvgPool1d(4))   # -> (N, C, 4)
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)



        # ----     Predictor of the CNN (MLP)   ----
        self.predictor = nn.Sequential(
            nn.Linear(prev_in * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(64, out_channel)

        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.predictor(x)
        x = self.fc(x)

        return x 
