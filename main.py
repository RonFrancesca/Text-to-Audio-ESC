import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
from torchsummary import summary
import os
import argparse
import yaml

import torchaudio
from model import CNNNetwork

from custom_dataset import UrbanSoundDataset



def init(argv=None):
    
    parser = argparse.ArgumentParser("Training a Audio Event Classification (AEC) syststem")
    parser.add_argument(
        "--conf_file",
        default="./config/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    
    parser.add_argument(
        "--log_dir",
        default="./exp/",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)
        
    return configs


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
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print(f"----------------")
    print("Training is done ")


if __name__== "__main__":
    
    config = init()
    
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    
    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]
    os.environ['CUDA_ALLOW_GROWTH'] = config["allow_growth"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # instantiating dataset
    metadata_file = config["data"]["metadata_file"]
    audio_dir = config["data"]["audio_dir"]
    sample_rate = config["feats"]["sample_rate"]
    audio_max_len = config["data"]["audio_max_len"]
    num_samples = sample_rate * audio_max_len
    
    # dataset
    usd = UrbanSoundDataset(config, num_samples, device)
    
    print(torch.cuda.device_count())
    
    # create a data loader for the dataset 
    train_data_loader = DataLoader(usd, 
                        batch_size=config["training"]["batch_size"], 
                        num_workers=torch.cuda.device_count() * 8,
                        pin_memory=True
                        )
    
    model = CNNNetwork(config)
    model = model.to(device)
    
    # example of the input
    input_example = (1, config["feats"]["n_mels"], int(num_samples/config["feats"]["n_window"]))
    summary(model, input_example)  

    # train model
    loss_fn = nn.CrossEntropyLoss()
    
    
    # Specify different weight decay values for different layers
    # For example, you may want to apply a higher weight decay to the weights of the fully connected layers
    params = [
        {'params': model.cnn.parameters(), 'weight_decay': 0},
        {'params': model.flatten.parameters(), 'weight_decay': 0},
        {'params': model.dense_layers.parameters(), 'weight_decay': 0.001},
    ]   
    
    # Add parameters of each linear layer in dense_layers with different weight decay - # If I want to apply different weigts to them
    # for i, layer in enumerate(model.dense_layers):
    #     if isinstance(layer, nn.Linear):
    #         params.append({'params': layer.parameters(), 'weight_decay': 0.001})  # Adjust weight_decay as needed
    
    optimiser = torch.optim.Adam(params, 
                                lr=config["opt"]["lr"]
                                )

    n_epochs = 2 if config["fast_run"] else config["training"]["n_epochs"]
    
    train(model, train_data_loader, loss_fn, optimiser, device, n_epochs)
    
    torch.save(model.state_dict(), os.path.join(config["data"]["checkpoint_folder"], "urban-sound-cnn.pth"))
    print("Model trained and stored at urban-sound-cnn.pth")




