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
from utils import save_confusion_matrix, collect_generated_metadata, collect_val_generated_metadata, get_classes

import ipdb

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

    # metadata file for real and fake dataset
    metadata_file_real = config["data"]["metadata_file_real"]
    metadata_file_fake = config["data"]["metadata_file_fake"]
    
    # audio path for real and fake dataset
    audio_dir_real = config["data"]["audio_dir_real"]
    audio_dir_fake = config["data"]["audio_dir_fake"]
    
    # make all the necessary folder
    checkpoint_folder = os.path.join(config["data"]["checkpoint_folder"], config["session_id"])
    os.makedirs(checkpoint_folder, exist_ok=True)
    accuracy_folder = os.path.join(config["base_dir"], 'accuracy')
    os.makedirs(accuracy_folder, exist_ok=True)
    
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
    
    # get dataset annotations
    annotations_real = pd.read_csv(config["data"]["metadata_file_real"])
    
    # folder for cross validation
    if config['fast_run']:
        max_fold = annotations_real["fold"].min() + 1
    else:
        max_fold = annotations_real["fold"].max() + 1
    
    # initiate the list for the metricc: accuracy, loss train and loss validation
    accuracy_history = []
    loss_train_history = []
    loss_val_history = []
    
    # definition of classes
    classes = get_classes()
    
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
        if config["concat_data"] == 0:
            # training from urban sound dataset - only original dataset
            train_data_real = annotations_real[~annotations_real['fold'].isin([n_fold, val_fold])]
            if config["fast_run"] == 1:
                train_data_real = train_data_real[:100]
            train_data_real.reset_index(drop=True, inplace=True)
            
            # it should not be important until we don't normalize the entire dataset
            #train_data_path = train_data.apply(lambda row: os.path.join(config["data"]["audio_dir_real"], f"fold{row[5]}", row[0]), axis=1)
            
            # validation from urban sound
            val_data_real = annotations_real[annotations_real['fold'] == val_fold]
            val_data_real.reset_index(drop=True, inplace=True)
        
        elif config["concat_data"]:
            # get all the data from the folders different than the testing and validation folder, from the generated dataset
            
            # training fake
            train_data_fake = collect_generated_metadata(config["data"]["metadata_file_fake"], n_fold, val_fold)
            if config["fast_run"] == 1:
                train_data_fake = train_data_fake[:100]
            
            # training real
            train_data_real = annotations_real[~annotations_real['fold'].isin([n_fold, val_fold])]
            if config["fast_run"] == 1:
                train_data_real = train_data_real[:100]
            train_data_real.reset_index(drop=True, inplace=True)
           
            # validation fake
            val_data_fake = collect_val_generated_metadata(config["data"]["metadata_file_fake"], val_fold)
                        
            # validation real
            val_data_real = annotations_real[annotations_real['fold'] == val_fold]
            val_data_real.reset_index(drop=True, inplace=True)
        
        else:
            print("Dataset not suppported yet")
        
        # dataset normalization 
        if config["data"]["normalization"] == 'dataset':
            # make a directory where to save mena and std
            # mean_dir_path = os.path.join(config['base_dir'], "mean_std")
            # os.makedirs(mean_dir_path, exist_ok=True)
            
            # file_path_mean = os.path.join(mean_dir_path, f"./tf_{n_fold}_vf_{val_fold}_mean.npy")
            # file_path_std = os.path.join(mean_dir_path, f"./tf_{n_fold}_vf_{val_fold}_std.npy")
            
            # if os.path.exists(file_path_mean) and os.path.exists(file_path_std):
            #     # the files already exist in the directory and have been calculated already
            #     mean_train = np.load(file_path_mean)
            #     std_train = np.load(file_path_std)
            #     print(f"Mean: {mean_train}, std: {std_train}")
            # else:
            #     # the files need to be calculated and saved in the mean and std folder
            #     mean_train, std_train = extract_stat_data(train_data_path, config, sample_rate, num_samples)
            #     np.save(file_path_mean, mean_train)
            #     np.save(file_path_std, std_train)
            pass
       
        elif config["data"]["normalization"] == 'spec':
            print("Normalizing spectogram by spectogram")
            mean_train = None
            std_train = None
        else:
            print(f"Normalization not defined")
        
        # collecting dataset
        if config['concat_data']:
            # I use both original and fake data to train my model
            usd_train_real =  UrbanSoundDataset(config, train_data_real, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='real')
            usd_train_fake =  UrbanSoundDataset(config, train_data_fake, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='fake')
            usd_train = torch.utils.data.ConcatDataset([usd_train_real, usd_train_fake])
        else:
            # I only use original data to train my model
            usd_train =  UrbanSoundDataset(config, train_data_real, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='real')
        
        
        testing_mode = config["testing"]["mode"]
        if testing_mode == 'f':
            # checking frame by frame. not used for now
            usd_val = UrbanSoundDatasetValTest(config, val_data, num_samples, mean_train, std_train, patch_lenght_samples, device)
        elif testing_mode == 'a':
            # testing considering the whole clip. 
            if config['concat_data']:
                # I use both original and fake data to validate my model
                usd_val_real =  UrbanSoundDataset(config, val_data_real, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='real')
                usd_val_fake =  UrbanSoundDataset(config, val_data_fake, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='fake')
                usd_val = torch.utils.data.ConcatDataset([usd_val_real, usd_val_fake])
            else:
                # I only use original date to validate my model
                usd_val =  UrbanSoundDataset(config, val_data_real, num_samples, mean_train, std_train, patch_lenght_samples, device, origin='real')
        else:
            # need to add to throw the error
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
        
        test_data = annotations_real[annotations_real['fold'] == n_fold]
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
        confusion_matrix_filename = os.path.join(log_fold, f"fold_{n_fold}_cmx.png")
        save_confusion_matrix(target_labels, predicted_labels, classes, confusion_matrix_filename)
        print(f"Accuracy score for folder: {n_fold}: {accuracy * 100:.2f}%")
        
        # sentence to print on the accuracy file
        sentence = f"Accuracy score for testing folder: {n_fold}: {accuracy * 100:.2f}%.\n"
        sentence = sentence + f"Model selected at epoch: {best_epoch}"

        # Specify the file path
        session_id = config["session_id"]
        accuracy_file_path = os.path.join(accuracy_folder, f"accuracy_{session_id}.txt")

        # Check if the file exists
        if os.path.exists(accuracy_file_path):
            # If the file exists, open it in append mode ("a")
            with open(accuracy_file_path, "a") as file:
                # Append the sentence to the file
                file.write(sentence + "\n")
        else:
            # If the file does not exist, open it in write mode ("w")
            with open(accuracy_file_path, "w") as file:
                # Write the sentence to the file
                file.write(sentence + "\n")
        
        accuracy_history.append(accuracy)

    print(f"Loss_train_final: {np.mean(loss_train_history):.2f}")
    print(f"Loss_validation_final: {np.mean(loss_val_history):.2f}")
    print(f"Accuracy: {np.mean(accuracy_history) * 100:.2f}%")