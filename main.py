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

from TrainClass import TrainClass
from NetworkData import NetworkData

from training import train
from inference import inference
from data_preprocess import extract_stat_data
from utils import (
    save_confusion_matrix,
    collect_generated_metadata,
    collect_val_generated_metadata,
    get_classes,
    make_folder,
    save_accuracy_to_csv,
)

from training_data_processing import Dataset_Settings, Features

from cross_validation_process import get_val_folder

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
    dataset_gen = config["metadata_gen"].split('/')[-1]
    session_id = config["session_id"] + '_' + config["training"]["model"] + '_' + dataset_gen
    current_run = os.path.join(runs_folders, session_id)
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

    # features for audios
    features = Features(config["feats"])
    training_data = TrainClass(config["training"])
    network_data = NetworkData(config["net"])

    # folder for cross validation
    fast_run = config["fast_run"]
    if fast_run:
        max_fold = annotations_real["fold"].min() + 3
        training_data.n_epochs = 1
    else:
        max_fold = annotations_real["fold"].max() + 1

    # loss function of the model
    loss_fn = nn.CrossEntropyLoss()
    
    classes_to_remove = ['street_music']

    for run in range(training_data.runs):

        print(f"\n****\nStarting run: {run}\n****\n")

        checkpoint_folder_run = os.path.join(checkpoint_folder, f"run_{run}")
        make_folder(checkpoint_folder_run)

        # dictionary for metrics
        metrics_dic = {
            "accuracy": [],
            "loss_train": [],
            "loss_val": [],
            "target_labels_all": [],
            "predicted_labels_all": [],
        }

        for n_fold in range(1, max_fold):

            print(f"Testing folder: {n_fold}")
            checkpoint_folder_path = os.path.join(
                checkpoint_folder_run, f"fold_{n_fold}"
            )
            make_folder(checkpoint_folder_path)

            writer = log.get_writer(
                os.path.join(
                    log_fold,
                    f"fold_{n_fold}_"
                    + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
            )

            # validation folder
            val_fold = get_val_folder(n_fold, max_fold, fast_run)

            dataset_settings = Dataset_Settings(
                val_fold,
                n_fold,
                annotations_real,
                metadata_real,
                metadata_gen,
                fast_run,
                n_rep=training_data.n_rep,
            )

            # take only the original and base dataset
            if training_data.data_type == "original":

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
                if training_data.data_aug is not None:
                    train_data_aug, val_data_aug = (
                        dataset_settings.get_augmented_dataset(features)
                    )
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

                    usd_train = torch.utils.data.ConcatDataset(
                        [usd_train, usd_train_aug]
                    )
                    usd_val = torch.utils.data.ConcatDataset([usd_val, usd_val_aug])

            elif training_data.data_type == "generated":

                train_data_gen, val_data_gen = dataset_settings.get_generated_data()
                
                # TODO: attention, this lines need to be removed for proper experiments
                train_data_gen = train_data_gen[~train_data_gen['class'].isin(classes_to_remove)]
                val_data_gen = val_data_gen[~val_data_gen['class'].isin(classes_to_remove)]

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

            elif training_data.data_type == "both":

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

                usd_train = torch.utils.data.ConcatDataset(
                    [usd_train_real, usd_train_gen]
                )

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
                
            elif training_data.data_type == "mixed":
                # replace replace_n_folder folders with generated AI data
                dataset_settings.set_folders(training_data.replace_n_folder)
                train_data_real, val_data_real = dataset_settings.get_original_data()
                train_data_gen, _ = dataset_settings.get_generated_data()

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

                usd_train = torch.utils.data.ConcatDataset(
                    [usd_train_real, usd_train_gen]
                )

                # validation dataset
                usd_val = UrbanSoundDataset(
                    config,
                    val_data_real,
                    features,
                    device,
                    origin="real",
                )

            else:
                raise Exception(
                    "Sorry, the value you inserted for the concatentaion mode is not valid!"
                )
                
            # dataloader for dataset
            train_data_loader = DataLoader(
                usd_train,
                shuffle=True,
                batch_size=training_data.batch_size,
                num_workers=torch.cuda.device_count() * 4,
                prefetch_factor=4,
                pin_memory=True,
            )

            val_data_loader = DataLoader(
                usd_val,
                shuffle=True,
                batch_size=training_data.batch_size_val,
                num_workers=torch.cuda.device_count() * 4,
                prefetch_factor=4,
                pin_memory=True,
            )

            ###########
            ## train ##
            ###########

            if training_data.model == "CNN":
                model = CNNNetwork(features.mel_bands, network_data)
                checkpoint_file_name = f"urban-sound-cnn_{run}.pth"
            elif training_data.model == "CRNN":
                model = CRNNBaseline(features.mel_bands)
                checkpoint_file_name = f"urban-sound-crnn_{run}.pth"
            else:
                print("Model selected is not implemented ")

            model = model.to(device)

            # optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=network_data.lr, eps=1e-07, weight_decay=1e-3
            )

            # print the summary of the folder, only for the first iteraction in the loop
            if n_fold == 1 and run == 0:
                input_example = (1, features.mel_bands, features.patch_samples)
                summary(model, input_example)

            loss_train, loss_val, best_epoch = train(
                model,
                config,
                train_data_loader,
                val_data_loader,
                loss_fn,
                optimizer,
                training_data.n_epochs,
                device,
                checkpoint_folder_path,
                writer,
                img_folder,
                features,
                checkpoint_filename=checkpoint_file_name,
            )
            print(f"Training folder {n_fold} done! :)")

            metrics_dic["loss_train"].append(loss_train)
            metrics_dic["loss_val"].append(loss_val)

            ###############
            ## inference ##
            ###############

            test_data = annotations_real[annotations_real["fold"] == n_fold]
            
            test_data.reset_index(drop=True, inplace=True)

            usd_test = UrbanSoundDataset(
                config,
                test_data,
                features,
                device,
            )

            test_data_loader = DataLoader(
                usd_test,
                batch_size=training_data.batch_size_test,
                num_workers=torch.cuda.device_count() * 4,
                prefetch_factor=4,
                pin_memory=True,
            )

            # load the model
            if training_data.model == "CNN":
                inference_model = CNNNetwork(features.mel_bands, network_data)
                checkpoint_file_name = f"urban-sound-cnn_{run}.pth"
            elif training_data.model == "CRNN":
                inference_model = CRNNBaseline(features.mel_bands)
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
                features,
                mode="a",
            )

            # calculate accuracy for current run and save confusion matrix
            accuracy = accuracy_score(target_labels, predicted_labels)
            save_confusion_matrix(
                target_labels,
                predicted_labels,
                get_classes(),
                os.path.join(log_fold, f"cfmx_fold_{n_fold}_run_{run}.png"),
            )

            # save metrics for the current run
            metrics_dic["accuracy"].append(accuracy)
            metrics_dic["target_labels_all"].extend(target_labels)
            metrics_dic["predicted_labels_all"].extend(predicted_labels)

        save_confusion_matrix(
            metrics_dic["target_labels_all"],
            metrics_dic["predicted_labels_all"],
            get_classes(),
            os.path.join(log_fold, f"cfmx_total_{run}.png"),
        )

        # save final results as csv file
        accuracy_filename = os.path.join(
            accuracy_folder, f"{config['session_id']}.csv "
        )
        save_accuracy_to_csv(metrics_dic["accuracy"], accuracy_filename)

        print(f"Accuracy: {np.mean(metrics_dic['accuracy']) * 100:.2f}%")
