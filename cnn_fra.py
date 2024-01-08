from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        #4 conv blocks / flatten / linear / softmax
        self.cnn = nn.Sequential()
        
        for i in range(len(config["net"]["out_channel"])):
            
            self.cnn.add_module(f"conv_{i}", 
                nn.Conv2d(
                    in_channels = config["net"]["in_channel"][i], 
                    out_channels = config["net"]["out_channel"][i],
                    kernel_size=config["net"]["kernel_size"],
                    stride=config["net"]["stride"],
                    padding=config["net"]["padding"],
                )
            )
            self.cnn.add_module(f"relu_{i}", nn.ReLU())
            self.cnn.add_module(f"max_pool_{i}", nn.MaxPool2d(config["net"]["max_pooling_ks"]))
        

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, config["net"]["nclass"])    #TODO: need to checck this from the flatten layer output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        
        x = self.cnn(input_data)

        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        
if __name__ == "__main__":

   cnn = CNNNetwork()
   summary(cnn, (1, 64, 44))  #if you have cuda, you will need to do cnn.cuda()