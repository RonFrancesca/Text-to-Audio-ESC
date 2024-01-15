import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchsummary import summary
import os
import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torchaudio
from model import CNNNetwork

from custom_dataset import UrbanSoundDataset
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram

class_mapping = [
   "0",  
   "1", 
   "2", 
   "3", 
   "4",
   "5",
   "6", 
   "7", 
   "8", 
   "9"
]


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


def train_one_epoch(model, data_loader, transformation, loss_fn, optimizer, device):
    
    num_batches = len(data_loader.dataset) / 256
    #print(f"Number of batches: {num_batches}")
    transformation.to(device)
    running_loss = 0.
    model.train()


    for _, (inputs, targets) in enumerate(tqdm(data_loader)):
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs = transformation(inputs)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # make prediction for this batch
        predictions = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(predictions, targets)
        loss.backward()

        # adjust learning weights
        optimiser.step()

        running_loss += loss.detach().item()
    
    return running_loss / num_batches

def val_one_epoch(model, data_loader, transformation, loss_fn, optimizer, device):
    
    num_batches = len(data_loader.dataset) / 128
    #print(f"Number of batches: {num_batches}")
    transformation.to(device)
    running_loss = 0.
    model.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(data_loader)):
                
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = transformation(inputs)

            # make prediction for this batch
            predictions = model(inputs)

            # Compute the loss 
            loss = loss_fn(predictions, targets)
            running_loss += loss.detach().item()
    
    return running_loss / num_batches

def train(model, 
          train_loader, 
          val_loader, 
          transformation, 
          loss_fn, 
          optimiser,
          device, 
          epochs, 
          checkpoint_folder, 
          early_stop_patience=20
    
    ):
    
    best_epoch = 0
    
    for i in tqdm(range(epochs)):
        
        print(f"Epoch: {i+1}")
        # training epoch
        train_loss = train_one_epoch(model, train_loader, transformation, loss_fn, optimiser, device)
        print(f"Train_loss: {train_loss}")
        
        val_loss = val_one_epoch(model, val_loader, transformation, loss_fn, optimiser, device)
        print(f"Val_loss: {val_loss}")
        
        # Handle saving best model + early stopping
        if i == 0:
            val_loss_best = val_loss
            early_stop_counter = 0
            saved_model_path = os.path.join(checkpoint_folder, "urban-sound-cnn.pth")
            torch.save(model.state_dict(), saved_model_path)
        
        if i > 0 and val_loss < val_loss_best:
            saved_model_path = saved_model_path
            torch.save(model.state_dict(), saved_model_path)
            val_loss_best = val_loss
            early_stop_counter = 0
            best_epoch = i
        
        else:
            early_stop_counter += 1
            print('Patience status: ' + str(early_stop_counter) + '/' + str(early_stop_patience))

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch ' + str(i))
            break
    
    print(f"Model saved at epoch: {best_epoch}")
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
    
    # dataset temporary
    
    annotations = pd.read_csv(config["data"]["metadata_file"])
    
    #paths_list = annotations.apply(lambda row: os.path.join(audio_dir, f"fold{row[5]}", row[0]), axis=1)
    
    
    # split train and test dataset
    train_dataset, test_dataset = train_test_split(annotations, shuffle=False, test_size=0.2, random_state=42)
    
    # reset index for the testing set
    test_dataset = test_dataset.reset_index(drop=True)
    
    # split the test set into validation and test set
    val_dataset, test_dataset = train_test_split(annotations, shuffle=False, test_size=0.1, random_state=42)
    
    # reset the index for the test dataset
    test_dataset = test_dataset.reset_index(drop=True)
    
    
    # dataset
    usd_train = UrbanSoundDataset(config, train_dataset, num_samples)
    usd_val = UrbanSoundDataset(config, val_dataset, num_samples)
    usd_test = UrbanSoundDataset(config, test_dataset, num_samples)
    
    train_data_loader = DataLoader(usd_train, 
                        shuffle=True,
                        batch_size=config["training"]["batch_size"],
                        num_workers=torch.cuda.device_count() * 8,
                        prefetch_factor=4,
                        pin_memory=True
                        )
    
    val_data_loader = DataLoader(usd_val, 
                        shuffle=True,
                        batch_size=config["training"]["batch_size_val"],
                        num_workers=torch.cuda.device_count() * 8,
                        prefetch_factor=4,
                        pin_memory=True
                        )
    
    test_data_loader = DataLoader(usd_test, 
                        batch_size=config["testing"]["batch_size"],
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
    
    transformation = MelSpectrogram(
        sample_rate=sample_rate, 
        n_fft=config["feats"]["n_window"],
        hop_length=config["feats"]["hop_length"],
        n_mels=config["feats"]["n_mels"]
    )
    
    checkpoint_folder = config["data"]["checkpoint_folder"]
    
    train(model, train_data_loader, val_data_loader, transformation, loss_fn, optimiser, device, n_epochs, checkpoint_folder)
    #torch.save(model.state_dict(), os.path.join(config["data"]["checkpoint_folder"], "urban-sound-cnn.pth"))
    print("Model trained and stored at urban-sound-cnn.pth")
    
    ###############
    ## inference ##
    ############### 
    
    # load the model
    #model = CNNNetwork(config)
    state_dict = torch.load(os.path.join(checkpoint_folder, "urban-sound-cnn.pth")) #train model that would need to be created
    model.load_state_dict(state_dict)
    #model = model.to(device)
    
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    
    model.eval()
    # get a sample from urban sound set for inference
    with torch.no_grad():
        
        for inputs, labels in test_data_loader:  # Use the test_loader for testing
            inputs, labels = inputs.to(device), labels.to(device)
            transformation = transformation.to(device)
            inputs = transformation(inputs)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.detach()

            # Get predicted labels
            #_, predicted = torch.max(outputs, 1)
            
            predicted_index = outputs[0].argmax(0)
            predicted = int(class_mapping[predicted_index])
            excepted = int(class_mapping[labels])
            
            # Append true and predicted labels to lists
            true_labels.append(excepted)
            predicted_labels.append(predicted)

        # Calculate accuracy using scikit-learn's accuracy_score
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    




