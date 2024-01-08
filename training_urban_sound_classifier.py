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
from cnn_fra import CNNNetwork

from custom_dataset import UrbanSoundDataset


def prepare_run(argv=None):
    
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
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
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
    
    config = prepare_run()
    
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    # Imports to select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # instantiating dataset
    metadata_file = config["data"]["metadata_file"]
    audio_dir = config["data"]["audio_dir"]
    sample_rate = config["feats"]["sample_rate"]
    audio_lenght = 4
    #num_samples = sample_rate * audio_lenght
    num_samples = 22050
    
    # dataset
    usd = UrbanSoundDataset(config, num_samples, device)
    

    # create a data loader for the dataset 
    train_data_loader = DataLoader(usd, batch_size=config["training"]["batch_size"])
    
    cnn = CNNNetwork(config).to(device)
    
    input_example = (1, 64, 44)
    summary(cnn, input_example)  #if you have cuda, you will need to do cnn.cuda()

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), 
                                lr=config["opt"]["lr"]
                                )
    
    train(cnn, train_data_loader, loss_fn, optimiser, device, config["training"]["n_epochs"])

    checkpoint_folder = "./checkpoint"
    
    torch.save(cnn.state_dict(), os.path.join(checkpoint_folder, "urban-sound-cnn.pth"))
    print("Model trained and stored at urban-sound-cnn.pth")




