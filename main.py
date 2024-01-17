import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
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
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import ipdb
import numpy as np

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

def log_mels(mels):
    """ Apply the log transformation to mel spectrograms.
    Args:
        mels: torch.Tensor, mel spectrograms for which to apply log.

    Returns:
        Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
    """

    amp_to_db = AmplitudeToDB(stype="amplitude").to(device)
    amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
    #return amp_to_db(mels).clamp(min=-50, max=80)  # clamp taken fromn desed task
    return amp_to_db(mels)

def take_patch_frame(patch_lenght):
    
    frames_1s = int(sample_rate / 1024)
    start_frame = torch.randint(0, frames_1s, (1,))
    #print("Random integer between 0 and 50 with a step of 1:", start_frame.item())
    end_frame = start_frame + patch_lenght
    return start_frame, end_frame

def train_one_epoch(model, 
        data_loader, 
        transformation,
        loss_fn, 
        optimizer, 
        device, 
        patch_lenght):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    #print(f"Number of batches: {num_batches}")
    transformation.to(device)
    running_loss = 0.
    model.train()


    for _, (inputs, targets) in enumerate(tqdm(data_loader)):
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # log-mel spectrogram
        inputs = transformation(inputs)
        inputs = log_mels(inputs)
        
        # TAKE ONLY THREE SECONDS PATCH
        start_frame, end_frame = take_patch_frame(patch_lenght)
        inputs = inputs[:, :, :, start_frame:end_frame]

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

def val_one_epoch(model, data_loader, transformation, loss_fn, device, patch_lenght):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    #print(f"Number of batches: {num_batches}")
    transformation.to(device)
    running_loss = 0.
    model.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(data_loader)):
                
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = transformation(inputs)
            inputs = log_mels(inputs)
            
            start_frame, end_frame = take_patch_frame(patch_lenght)
            inputs = inputs[:, :, :, start_frame:end_frame]

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
          patch_lenght,
          early_stop_patience=100
    ):
    
    best_epoch = 0
    
    for i in tqdm(range(epochs)):
        
        print(f"Epoch: {i+1}")
        # training epoch
        train_loss = train_one_epoch(model, train_loader, transformation, loss_fn, optimiser, device, patch_lenght)
        print(f"Train_loss: {train_loss:.2f}")
        
        val_loss = val_one_epoch(model, val_loader, transformation, loss_fn, device, patch_lenght)
        print(f"Val_loss: {val_loss:.2f}")
        
        # Handle saving best model + early stopping
        if i == 0:
            val_loss_best = val_loss
            early_stop_counter = 0
            saved_model_path = os.path.join(checkpoint_folder, "urban-sound-cnn_1.pth")
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
    return train_loss, val_loss_best
    

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
    
    num_samples_3s = sample_rate * config["data"]["patch_lenght_s"]
    patch_lenght = int((num_samples_3s / config["feats"]["n_window"]) - 1)
    
    
    # dataset temporary
    annotations = pd.read_csv(config["data"]["metadata_file"])
    
    accuracy_history = []
    loss_train_history = []
    loss_val_history = []
    
    
    for n_fold in range(1, 11):
        print(f"Testing folder: {n_fold}")
        val_fold = (n_fold + 1) % 11
        if val_fold == 0:
            val_fold = 1
        print(f"Validation folder: {val_fold}")
        # take the files of all the rows a part from the one in the folder which are testing 
        
        train_data = annotations[~annotations['fold'].isin([n_fold, val_fold])]
        train_data.reset_index(drop=True, inplace=True)
        
        val_data = annotations[annotations['fold'] == val_fold]
        val_data.reset_index(drop=True, inplace=True)
        
        test_data = annotations[annotations['fold'] == n_fold]
        test_data.reset_index(drop=True, inplace=True)
        
    
        # dataset
        usd_train = UrbanSoundDataset(config, train_data, num_samples)
        usd_val = UrbanSoundDataset(config, val_data, num_samples)
        usd_test = UrbanSoundDataset(config, test_data, num_samples)
        
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
        
        
        n_frames_network = int(num_samples_3s/config["feats"]["n_window"])
        # example of the input
        if n_fold == 1:
            input_example = (1, config["feats"]["n_mels"], n_frames_network)
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

        n_epochs = 1 if config["fast_run"] else config["training"]["n_epochs"]
        
        # TOD: plot the logaritm
        transformation = MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=config["feats"]["n_window"],
            win_length=config["feats"]["n_window"],
            hop_length=config["feats"]["hop_length"],
            f_min=config["feats"]["f_min"],
            f_max=config["feats"]["f_max"],
            n_mels=config["feats"]["n_mels"],
            # window_fn=torch.hamming_window,
            # wkwargs={"periodic": False},
            # power=1
        )
        
        checkpoint_folder = config["data"]["checkpoint_folder"]
        
        
        loss_train, loss_val = train(model, train_data_loader, val_data_loader, transformation, loss_fn, optimiser, device, n_epochs, checkpoint_folder, patch_lenght)
        #torch.save(model.state_dict(), os.path.join(config["data"]["checkpoint_folder"], "urban-sound-cnn.pth"))
        print("Model trained and stored at urban-sound-cnn.pth")
        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)
        
        
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
                
                # log-mel spectogram
                inputs = transformation(inputs)
                inputs = log_mels(inputs)
                
                start_frame, end_frame = take_patch_frame(patch_lenght)
                inputs = inputs[:, :, :, start_frame:end_frame]
                
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
        accuracy_history.append(accuracy_score(true_labels, predicted_labels))
    
    #ipdb.set_trace() 
    print(f"Loss_train_final: {np.mean(loss_train_history):.2f}%")
    print(f"Loss_validation_final: {np.mean(loss_val_history):.2f}%")
    print(f"Accuracy: {np.mean(accuracy_history) * 100:.2f}%")
    
    




