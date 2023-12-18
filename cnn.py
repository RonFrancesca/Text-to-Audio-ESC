from torch import nn
from torchsummary import summary

# you should send the value as params to your network so you can create your net diamically.

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        #4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16, #filters in the conv layers
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)          
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32, #filters in the conv layers
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)          
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64, #filters in the conv layers
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # ---> what is this doing?    
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128, #filters in the conv layers
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)          
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)    # 128 * 5 * 4: should be the same of the flatten, 10 is the number of classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        
if __name__ == "__main__":

   cnn = CNNNetwork()
   summary(cnn, (1, 64, 44))  #if you have cuda, you will need to do cnn.cuda()