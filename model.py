from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        # three convolutional layers interleaved with two pooling operations
        self.cnn = nn.Sequential()
        
        for i in range(len(config["net"]["out_channel"])):
            # add convolutional layer 
            self.cnn.add_module(
                f"conv_{i}", 
                nn.Conv2d(
                    in_channels = config["net"]["in_channel"][i], 
                    out_channels = config["net"]["out_channel"][i],
                    kernel_size=config["net"]["kernel_size"],
                )
            )
            
            # add max pooling layer, skipping the last layer
            if i < (len(config["net"]["out_channel"]) - 1):
                self.cnn.add_module(
                    f"max_pool_{i}", 
                    nn.MaxPool2d(kernel_size = tuple(config["net"]["maxp_ks"]), stride=tuple(config["net"]["maxp_stride"])),
                )
            
            # add ReLu
            self.cnn.add_module(f"relu_{i}", nn.ReLU())
        
        # flatten layer
        self.flatten = nn.Flatten()
        
        # followed by two fully connected (dense) layers.
        self.dense_layers = nn.Sequential()
        
        for i in range(len(config["net"]["dense_in"])):
            self.dense_layers.add_module(
                    f"dense_{i}", 
                    nn.Dropout(p = config["net"]["dense_drop"][i]),
                )
            
            self.dense_layers.add_module(
                    f"linear_{i}", 
                    nn.Linear(config["net"]["dense_in"][i], config["net"]["dense_out"][i]),
                )
            
            if i == 0:
                self.dense_layers.add_module(
                    f"relu_{i}", 
                    nn.ReLU()
                )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        x = self.cnn(input_data)
        x = self.flatten(x)
        logits = self.dense_layers(x)
        #predictions = self.softmax(logits)
        return logits
        
if __name__ == "__main__":

   cnn = CNNNetwork()
   summary(cnn, (1, 128, 172))  #if you have cuda, you will need to do cnn.cuda()