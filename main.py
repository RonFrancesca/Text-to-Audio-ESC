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
from model import CNNNetwork
from CRNN_baseline import CRNNBaseline
from urban_sound_dataset import (
    UrbanSoundDataset,
    UrbanSoundDatasetValTest,
    UrbanSoundDataset_generated,
)
from training import train
from inference import inference
from data_preprocess import extract_stat_data
from utils import (
    save_confusion_matrix,
    collect_generated_metadata,
    collect_val_generated_metadata,
    get_classes,
    make_folder,
)

from training_data_processing import (
    Dataset_Settings, 
    Features
)

from cross_validation_process import (
    get_val_folder
)

import ipdb
import datetime


def init(argv=None):

    parser = argparse.ArgumentParser(
        "Training a Audio Event Classification (AEC) system with generative AI data"
    )
    parser.add_argument(
        "--conf_file",
        default="./config/default.yaml",
        help="The configuration file with all the experiment parameters needed for the experiment.",
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    return configs


if __name__ == "__main__":

    config = init()

    # cuda related code
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    os.environ["CUDA_ALLOW_GROWTH"] = config["allow_growth"]

    # select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # folders to generate for the current run
    base_dir = config["base_dir"]
    runs_folders = os.path.join(base_dir, "runs")
    current_run = os.path.join(runs_folders, config["session_id"])
    checkpoint_folder = os.path.join(current_run, "checkpoints")
    accuracy_folder = os.path.join(current_run, "accuracy")
    img_folder = os.path.join(current_run, "images")
    log_fold = os.path.join(current_run, "log")
    
    for folder in [checkpoint_folder, accuracy_folder, img_folder]:
        make_folder(folder)
    
    # path to metadata and data folders
    metadata_real = config["metadata_real"]
    metadata_gen = config["metadata_gen"]
    audio_dir_real = config["audio_dir_real"]
    audio_dir_fake = metadata_gen
    
    # annotations
    annotations_real = pd.read_csv(metadata_real)

    # features for audios - when do I need them? 
    
    features = Features(config)

    # folder for cross validation
    if features.fast_run:
        max_fold = annotations_real["fold"].min() + 3
    else:
        max_fold = annotations_real["fold"].max() + 1

    # initiate the list for the metricc: accuracy, loss train and loss validation
    accuracy_history = []
    loss_train_history = []
    loss_val_history = []
    target_labels_all = []
    predicted_labels_all = []


    for n_fold in range(1, max_fold):

        print(f"Testing folder: {n_fold}")
        checkpoint_folder_path = os.path.join(checkpoint_folder, f"fold_{n_fold}")
        make_folder(checkpoint_folder_path)
        
        writer = log.get_writer(
            os.path.join(
                log_fold,
                f"fold_{n_fold}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        )

        # validation folder
        val_fold = get_val_folder(n_fold, max_fold, features.fast_run)
        
        dataset_settings = Dataset_Settings(
            val_fold, 
            n_fold, 
            annotations_real,
            metadata_real, 
            metadata_gen,
            features.fast_run, 
            n_rep=1, 
        )
        
        # take only the original and base dataset
        if features.data_type == 'original':
            
            # real dataset
            train_data_real, val_data_real = dataset_settings.get_original_data()
            usd_train = UrbanSoundDataset(
                config,
                train_data_real,
                features,
                device,
                origin="real",
            )
            
            usd_val = UrbanSoundDataset(
                config,
                val_data_real,
                features,
                device,
                origin="real",
            )
            
            # real dataset with augmentation applied
            if features.data_aug is not None:
                train_data_aug, val_data_aug = dataset_settings.get_augmented_dataset(features)
                usd_train_aug = UrbanSoundDataset(
                    config,
                    train_data_aug,
                    features,
                    device,
                    origin="aug",
                )
                
                usd_val_aug = UrbanSoundDataset(
                    config,
                    val_data_aug,
                    features,
                    device,
                    origin="aug",
                )
                
                usd_train = torch.utils.data.ConcatDataset([usd_train, usd_train_aug])
                usd_val = torch.utils.data.ConcatDataset([usd_val, usd_val_aug])
            
        elif features.data_type == 'generated':
            
            train_data_gen, val_data_gen = dataset_settings.get_generated_data()
            
            usd_train = UrbanSoundDataset(
                config,
                train_data_gen,
                features,
                device,
                origin="fake",
            )
            
            usd_val = UrbanSoundDataset(
                config,
                val_data_gen,
                features,
                device,
                origin="fake",
            )
        
        elif features.data_type == 'both':
            
            train_data_real, val_data_real = dataset_settings.get_original_data()
            train_data_gen, val_data_gen = dataset_settings.get_generated_data()
            
            # training dataset 
            usd_train_real = UrbanSoundDataset(
                config,
                train_data_real,
                features,
                device,
                origin="real",
            )
            
            usd_train_gen = UrbanSoundDataset(
                config,
                train_data_gen,
                features,
                device,
                origin="fake",
            )
            
            usd_train = torch.utils.data.ConcatDataset([usd_train_real, usd_train_gen])
            
            # validation dataset
            usd_val_real = UrbanSoundDataset(
                    config,
                    val_data_real,
                    features,
                    device,
                    origin="real",
                )
            
            usd_val_gen = UrbanSoundDataset(
                    config,
                    val_data_gen,
                    features,
                    device,
                    origin="fake",
                )
            
            usd_val = torch.utils.data.ConcatDataset([usd_val_real, usd_val_gen])
        
        else:
            raise Exception("Sorry, the value you inserted for the concatentaion mode is not valid!")

        # dataloader for dataset
        train_data_loader = DataLoader(
            usd_train,
            shuffle=True,
            batch_size=config["batch_size"],
            num_workers=torch.cuda.device_count() * 4,
            prefetch_factor=4,
            pin_memory=True,
        )

        val_data_loader = DataLoader(
            usd_val,
            shuffle=True,
            batch_size=config["batch_size_val"],
            num_workers=torch.cuda.device_count() * 4,
            prefetch_factor=4,
            pin_memory=True,
        )

        ###########
        ## train ##
        ###########
        run = config["run"]
        if config["model"] == "CNN":
            model = CNNNetwork(config)
            checkpoint_file_name = f"urban-sound-cnn_{run}.pth"
        elif config["model"] == "CRNN":
            model = CRNNBaseline(config)
            checkpoint_file_name = f"urban-sound-crnn_{run}.pth"
        else:
            print("None model selected")
        model = model.to(device)

        # print the summary of the folder, only for the first iteraction in the loop
        if n_fold == 1:
            input_example = (1, config["n_mels"], features.patch_samples)
            summary(model, input_example)

        # loss function of the model
        loss_fn = nn.CrossEntropyLoss()

        # Specify different weight decay values for different layers
        # L2 regularization
        if config["model"] == "Cnn":
            # params = [
            #     {'params': model.cnn.parameters(), 'weight_decay': 0},
            #     {'params': model.flatten.parameters(), 'weight_decay': 0},
            #     {'params': model.dense_layers.parameters(), 'weight_decay': 0.001},
            # ]

            # optimizer = torch.optim.Adam(params, lr=config["opt"]["lr"])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3
            )

        loss_train, loss_val, best_epoch = train(
            model,
            config,
            train_data_loader,
            val_data_loader,
            loss_fn,
            optimizer,
            features.n_epochs,
            device,
            checkpoint_folder_path,
            writer,
            img_folder,
            features.patch_samples,
            features.sr,
            features.n_window,
            # testing_mode,
            checkpoint_filename=checkpoint_file_name,
        )

        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)

        ###############
        ## inference ##
        ###############

        test_data = annotations_real[annotations_real["fold"] == n_fold]
        test_data.reset_index(drop=True, inplace=True)

        # if testing_mode == "f":
        #     usd_test = UrbanSoundDatasetValTest(
        #         config,
        #         test_data,
        #         num_samples,
        #         mean_train,
        #         std_train,
        #         patch_samples,
        #         device,
        #     )
        # elif testing_mode == "a":
        
        usd_test = UrbanSoundDataset(
            config,
            test_data,
            features,
            device,
        )

        test_data_loader = DataLoader(
            usd_test,
            batch_size=config["batch_size_test"],
            num_workers=torch.cuda.device_count() * 4,
            prefetch_factor=4,
            pin_memory=True,
        )

        # load the model
        if config["model"] == "CNN":
            inference_model = CNNNetwork(config)
            checkpoint_file_name = f"urban-sound-cnn_{run}.pth"
        elif config["model"] == "CRNN":
            inference_model = CRNNBaseline(config)
            checkpoint_file_name = f"urban-sound-crnn_{run}.pth"
        else:
            print("None model selected")

        state_dict = torch.load(
            os.path.join(checkpoint_folder_path, checkpoint_file_name)
        )
        inference_model.load_state_dict(state_dict)
        inference_model = inference_model.to(device)
        inference_model.eval()

        # accuracy score for the testing folder
        target_labels, predicted_labels = inference(
            inference_model,
            config,
            img_folder,
            test_data_loader,
            device,
            features.patch_samples,
            features.sr,
            features.n_window,
            mode='a',
        )

        # calculate accuracy
        accuracy = accuracy_score(target_labels, predicted_labels)

        # save confusion matrix per file
        confusion_matrix_filename = os.path.join(
            log_fold, f"fold_{n_fold}_{run}_cmx.png"
        )
        save_confusion_matrix(
            target_labels, predicted_labels, get_classes(), confusion_matrix_filename
        )
        print(f"Accuracy score for folder: {n_fold}: {accuracy * 100:.2f}%")

        # sentence to print on the accuracy file
        sentence = (
            f"Accuracy score for testing folder: {n_fold}: {accuracy * 100:.2f}%.\n"
        )
        sentence = sentence + f"Model selected at epoch: {best_epoch}"

        # Specify the file path
        session_id = config["session_id"]
        accuracy_file_path = os.path.join(accuracy_folder, f"{session_id}.txt")

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
        target_labels_all.extend(target_labels)
        predicted_labels_all.extend(predicted_labels)

    # save total and final confusion matrix

    confusion_matrix_filename_final = os.path.join(
        log_fold, f"confusion_matrix_final_{run}.png"
    )
    save_confusion_matrix(
        target_labels_all,
        predicted_labels_all,
        get_classes(),
        confusion_matrix_filename_final,
    )

    print(f"Loss_train_final: {np.mean(loss_train_history):.2f}")
    print(f"Loss_validation_final: {np.mean(loss_val_history):.2f}")
    print(f"Accuracy: {np.mean(accuracy_history) * 100:.2f}%")
