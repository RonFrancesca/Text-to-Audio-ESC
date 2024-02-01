import os
import argparse
import yaml

import numpy as np
import pandas as pd

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import accuracy_score

import log
from model_notebook import Net
from model import CNNNetwork
from urban_sound_dataset import UrbanSoundDataset, UrbanSoundDatasetValTest, UrbanSoundDataset_generated
from training import train
from inference import inference
from data_preprocess import extract_stat_data
from utils import save_confusion_matrix, collect_generated_metadata

#import ipdb

def init(argv=None):
    
    parser = argparse.ArgumentParser("Training a Audio Event Classification (AEC) syststem")
    parser.add_argument(
        "--conf_file",
        default="./config/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)
        
    return configs
    
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

    # get audio and metadata path
    metadata_file = config["data"]["metadata_file"]
    audio_dir = config["data"]["audio_dir"]
    checkpoint_folder = os.path.join(config["data"]["checkpoint_folder"], config["session_id"])
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    # features for audio
    sample_rate = config["feats"]["sample_rate"]
    audio_max_len = config["data"]["audio_max_len"]
    num_samples = sample_rate * audio_max_len
    num_samples_3s = sample_rate * config["data"]["patch_lenght_s"]
    patch_lenght_samples = int((num_samples_3s / config["feats"]["n_window"]) - 1)
    
    # number of epochs
    n_epochs = 1 if config["fast_run"] else config["training"]["n_epochs"]
    
    # get writer for tensorboard log
    log_fold = os.path.join(config["log_dir"], config["session_id"])
    #writer = log.get_writer(config["log_dir"])
    
    # get dataset annotations
    annotations = pd.read_csv(config["data"]["metadata_file"])
    
    # folder for cross validation
    if config['fast_run']:
        max_fold = annotations["fold"].min() + 1
        
    else:
        max_fold = annotations["fold"].max() + 1
    
    # initiate the list for the metrics
    # accuracy
    accuracy_history = []
    
    # training loss
    loss_train_history = []
    
    # validation loss
    loss_val_history = []
    
    for n_fold in range(1, max_fold):
        
        print(f"Testing folder: {n_fold}")
        checkpoint_folder_fold = os.path.join(checkpoint_folder, f"fold_{n_fold}")
        os.makedirs(checkpoint_folder_fold, exist_ok=True)
        
        writer = log.get_writer(os.path.join(log_fold, f"fold_{n_fold}"))
        
        # validation folder
        val_fold = (n_fold + 1) % max_fold
        if val_fold == 0:
            val_fold = 1
            
        if config["fast_run"]:
            val_fold = 2
        
        print(f"Validation folder: {val_fold}")

        # training data: all the files from any folder but the testing and validation folder
        if config["audio_used"] == 'original':
            train_data = annotations[~annotations['fold'].isin([n_fold, val_fold])]
            if config["fast_run"] == 1:
                train_data = train_data[:100]
            train_data.reset_index(drop=True, inplace=True)
            train_data_path = train_data.apply(lambda row: os.path.join(config["data"]["audio_dir"], f"fold{row[5]}", row[0]), axis=1)
        elif config["audio_used"] == 'gen':
            # get all the data from the folders different than the testing and validation
            train_data = collect_generated_metadata(config["data"]["metadata_file_generated"], n_fold, val_fold)
            if config["fast_run"] == 1:
                train_data = train_data[:100]
        else:
            print("Dataset not suppported")
        
        
        val_data = annotations[annotations['fold'] == val_fold]
        val_data.reset_index(drop=True, inplace=True)
        
        # dataset normalization 
        if config["data"]["normalization"] == 'dataset':
            file_path_mean = f"./tf_{n_fold}_vf_{val_fold}_mean.npy"
            file_path_std = f"./tf_{n_fold}_vf_{val_fold}_std.npy"
            if os.path.exists(file_path_mean) and os.path.exists(file_path_std):
                mean_train = np.load(file_path_mean)
                std_train = np.load(file_path_std)
                print(f"Mean: {mean_train}, std: {std_train}")
            else:
                mean_train, std_train = extract_stat_data(train_data_path, config, sample_rate, num_samples)
                np.save(file_path_mean, mean_train)
                np.save(file_path_std, std_train)
        elif config["data"]["normalization"] == 'spec':
            print("Normalizing spectogram by spectogram")
            # need to be changed
            mean_train = None
            std_train = None
        else:
            print(f"Normalization not defined")
        
        # collecting dataset
        if config["audio_used"] == "original":
            # original dataset used
            usd_train = UrbanSoundDataset(config, train_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        elif config["audio_used"] == 'gen':
            # generated dataset used
            usd_train = UrbanSoundDataset_generated(config, train_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        else:
            print("Dataset not supported")
        
        testing_mode = config["testing"]["mode"]
        if testing_mode == 'f':
            usd_val = UrbanSoundDatasetValTest(config, val_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        elif testing_mode == 'a':
            usd_val = UrbanSoundDataset(config, val_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        else:
            # throw error
            print(f"Option not available")
        
        # dataloader for dataset
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
        
        
        ###########
        ## train ##
        ###########
        
        if config["model"] == 'Cnn':
            model = CNNNetwork(config)
        else:
            model = Net()
        model = model.to(device)
        
        # print the summary of the folder, only for the first iteartion in the loop
        if n_fold == 1:
            input_example = (1, config["feats"]["n_mels"], patch_lenght_samples)
            summary(model, input_example)  

        # loss function of the model 
        loss_fn = nn.CrossEntropyLoss()
        
        # Specify different weight decay values for different layers
        # L2 regularization
        if config["model"] == 'Cnn':
            # params = [
            #     {'params': model.cnn.parameters(), 'weight_decay': 0},
            #     {'params': model.flatten.parameters(), 'weight_decay': 0},
            #     {'params': model.dense_layers.parameters(), 'weight_decay': 0.001},
            # ]   
            
            #optimizer = torch.optim.Adam(params, lr=config["opt"]["lr"])
            optimizer =torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)
        else:
            optimizer =torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)
        
        loss_train, loss_val, best_epoch = train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, checkpoint_folder_fold, writer, testing_mode)
        
        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)
        
        ###############
        ## inference ##
        ############### 
        
        test_data = annotations[annotations['fold'] == n_fold]
        test_data.reset_index(drop=True, inplace=True)
        
        if testing_mode == 'f':
            usd_test = UrbanSoundDatasetValTest(config, test_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        elif testing_mode == 'a':
            usd_test = UrbanSoundDataset(config, test_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        
        test_data_loader = DataLoader(usd_test, 
                            batch_size=config["testing"]["batch_size"],
                            num_workers=torch.cuda.device_count() * 8,
                            pin_memory=True
                            )
        
        # load the model
        if config['model'] == 'Cnn':
            inference_model = CNNNetwork(config)
        else:
            inference_model = Net()
        
        state_dict = torch.load(os.path.join(checkpoint_folder_fold, "urban-sound-cnn.pth")) 
        inference_model.load_state_dict(state_dict)
        inference_model = inference_model.to(device)        
        inference_model.eval()
        
        # accuracy score for the testing folder
        target_labels, predicted_labels = inference(inference_model, 
            test_data_loader, 
            device, 
            testing_mode
            )
        
         # calculate accuracy
        accuracy = accuracy_score(target_labels, predicted_labels)
        
        classes = ["air_conditioner", 
                   "car_horn", 
                   "children_playing",
                   "dog_bark", 
                   "drilling", 
                   "engine_idling", 
                   "gun_shot", 
                   "jackhammer", 
                   "siren", 
                   "street_music"
                   ]
        
        confusion_matrix_filename = os.path.join(log_fold, f"fold_{n_fold}_cmx.png")
        save_confusion_matrix(target_labels, predicted_labels, classes, confusion_matrix_filename)
        print(f"Accuracy score for folder: {n_fold}: {accuracy * 100:.2f}%")
        
        # sentence to print on the accuracy file
        sentence = f"Accuracy score for testing folder: {n_fold}: {accuracy * 100:.2f}%.\n"
        sentence = sentence + f"Model selected at epoch: {best_epoch}"

        # Specify the file path
        session_id = config["session_id"]
        file_path = f"accuracy_{session_id}.txt"

        # Check if the file exists
        if os.path.exists(file_path):
            # If the file exists, open it in append mode ("a")
            with open(file_path, "a") as file:
                # Append the sentence to the file
                file.write(sentence + "\n")
        else:
            # If the file does not exist, open it in write mode ("w")
            with open(file_path, "w") as file:
                # Write the sentence to the file
                file.write(sentence + "\n")
        
        accuracy_history.append(accuracy)
    
    #ipdb.set_trace() 
    print(f"Loss_train_final: {np.mean(loss_train_history):.2f}")
    print(f"Loss_validation_final: {np.mean(loss_val_history):.2f}")
    print(f"Accuracy: {np.mean(accuracy_history) * 100:.2f}%")