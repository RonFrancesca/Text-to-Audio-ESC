from torch import nn
import torch.nn.functional as F


class CNNNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        
        self.flatten = nn.Flatten()

        # to be removed afterwards
        if config["feats"]["n_mels"] == 128:
            in_features_layer_1 = 1680
        elif config["feats"]["n_mels"] == 64:
            in_features_layer_1 = 336
            
        self.fc1 = nn.Linear(in_features=in_features_layer_1, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        

    def forward(self, x):

         # cnn layer-1 - with 64 mel band (ks = 3) ! with 64 mel band (ks = 3)
        x = self.conv1(x) #input size [2, 1, 64, 45] | [2, 1, 128, 45]
        x = F.max_pool2d(x, kernel_size=(4,2), stride=(4,2)) #input size [2, 24, 62, 43] | [2, 1, 126, 45]
        x = F.relu(x) #input size [2, 24, 15, 21] | [2, 24, 31, 21]

        # cnn layer-2
        x = self.conv2(x) #input size [2, 24, 15, 20] | [2, 24, 31, 21]
        x = F.max_pool2d(x, kernel_size=(4,2), stride=(4,2)) #input size [2, 48, 13, 19] | [2, 48, 29, 19]
        x = F.relu(x)  # input size [2, 48, 3, 9] | [2, 48, 7, 9]

        # cnn layer-3
        x = self.conv3(x) #input size [2, 48, 3, 9] | [2, 48, 7, 9]
        x = F.relu(x) # input size [2, 48, 1, 7] | [2, 48, 5, 7]

        # # global average pooling 2D
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(-1, 48)
        x = self.flatten(x)  # input size [2, 48, 1, 7] | [2, 48, 5, 7]

        # dense layer-1
        x = self.fc1(x) # [2, 336] | [2, 1680]
        x = F.dropout(x, p=0.5) # [2, 64] | [2, 64]
        x = F.relu(x) # [2, 64] | [2, 64]
    
        # dense output layer
        x = self.fc2(x) # [2, 64] | [2, 64]
        logits = F.dropout(x, p=0.5) # [2, 10] | [2, 10]
        
        return logits
