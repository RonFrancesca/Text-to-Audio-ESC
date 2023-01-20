import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 


# 1. download dataset
# 2. create data __loader__
# 3. build model 
# 4. train 
# 5. save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

class FeedForwardNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()    # of the base class
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),      # 28*28 are the images size from MNIST
            nn.ReLU(), 
            nn.Linear(256, 10)          # 10 classes as output 
        )
        # sort of normalizatio, it takes all the input values and transform them such that the sum will be 1
        self.softmax = nn.Softmax(dim=1)   

    def forward(self, input_data):

        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        
        return predictions

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):

    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model,data_loader, loss_fn, optimiser, device)
        print(f"----------------")
    print("Training is done ")


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data", 
        download=True,
        train=True, 
        transform=ToTensor()     #every value is normalized between 0 and 1
    )

    validation_data = datasets.MNIST(
        root="data", 
        download=True,
        train=False, 
        transform=ToTensor()     #every value is normalized between 0 and 1
    )

    return train_data, validation_data

if __name__== "__main__":
    # download MNIST object
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the dataset 
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model 
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Use device {device}")
    feed_forward_net = FeedForwardNet().to(device)

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), 
                                lr=LEARNING_RATE
                                )
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feed_forward_net.pth")
    print("Model trained and stored at feed_forward_net.pth")




