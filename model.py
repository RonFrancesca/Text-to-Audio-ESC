from torch import nn
import torch.nn.functional as F


class CNNNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2400, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        

    def forward(self, x):

         # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(4,2), stride=(4,2))
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(4,2), stride=(4,2))
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # # global average pooling 2D
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(-1, 48)
        x = self.flatten(x)

        # dense layer-1
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
    
        # dense output layer
        x = self.fc2(x)
        logits = F.dropout(x, p=0.5)
        
        return logits
