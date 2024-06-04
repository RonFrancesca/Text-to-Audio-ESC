import pandas as pd

from utils import (
    collect_generated_metadata,
    collect_val_generated_metadata,
    data_augmentation_list,
)

import random


class Features:

    def __init__(self, config):

        self.sr = config["sr"]
        self.n_window = config["n_window"]
        self.hop_length = config["hop_length"]
        self.f_min = config["f_min"]
        self.f_max = config["f_max"]
        self.audio_s = config["audio_s"]
        self.n_filters = config["n_filters"]
        self.num_samples = self.sr * self.audio_s
        self.patch_samples = int(
            ((self.sr * config["patch_s"]) / config["n_window"]) - 1
        )
        self.mel_bands = config["n_mels"]
        self.mean = None
        self.std = None


class Dataset_Settings:

    def __init__(
        self,
        val_fold,
        n_fold,
        annotations_real,
        metadata_real,
        annotations_gen,
        fast_run,
        annotations_aug=None,
        n_rep=1,
        dataset="UrbanSound8K",
        total_folders = 10
    ):

        self.val_fold = val_fold
        self.n_fold = n_fold
        self.annotations_real = annotations_real
        self.metadata_real = metadata_real
        self.annotations_gen = annotations_gen
        self.annotations_aug = annotations_aug
        self.n_rep = n_rep
        self.fast_run = fast_run
        self.dataset = dataset
        self.original_folders = None
        self.fakes_folders = None
        self.total_folder = total_folders

    def _get_dataset(self):
        
        if self.fakes_folders:
            folder_not_condidered = [self.n_fold, self.val_fold] + self.fakes_folders
        else:
            folder_not_condidered = [self.n_fold, self.val_fold] 

        train_data = self.annotations_real[
            ~self.annotations_real["fold"].isin(folder_not_condidered)
        ]
        val_data = self.annotations_real[self.annotations_real["fold"] == self.val_fold]

        if self.fast_run:
            train_data = train_data[:100]

        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        return train_data, val_data
    

    def get_original_data(self):
        return self._get_dataset()
        
    
    def apply_augmentation(self):
        return self._get_dataset(self.annotations_aug)

    def get_generated_data(self):
        
        if self.original_folders:
            folders_not_considered = [self.n_fold, self.val_fold] + self.original_folders
        else:
            folders_not_considered = [self.n_fold, self.val_fold]
        
        train_data_gen = collect_generated_metadata(
            self.annotations_gen, self.n_rep, folders_not_considered
        )

        val_data_gen = collect_val_generated_metadata(
            self.annotations_gen, self.n_rep, self.val_fold
        )

        if self.fast_run:
            train_data_gen = train_data_gen[:100]

        return train_data_gen, val_data_gen

    def get_augmented_dataset(self, features):


        # only one augmentation
        if len(features.data_aug.split("_")) == 1:

            annotation_aug = pd.read_csv(
                self.metadata_real.replace(
                    f"{self.dataset}.csv",
                    f"{self.dataset}_{features.data_aug}_{features.mel_bands}.csv",
                )
            )
            return self._get_dataset(annotation_aug)

        # multiple augmentation
        else:

            train_data_aug = []
            val_data_aug = []

            data_aug_values = data_augmentation_list(features.data_aug)
            for value in data_aug_values:
                annotations_aug = pd.read_csv(
                    self.metadata_real.replace(
                        f"{self.dataset}.csv",
                        f"{self.dataset}_{features.data_aug[:2]}_{features.mel_bands}_{value}.csv",
                    )
                )
                train_data_tmp, val_data_tmp = self._get_dataset(annotations_aug)
                if len(train_data_aug) == 0 and len(val_data_aug) == 0:
                    train_data_aug = train_data_tmp
                    val_data_aug = val_data_tmp
                else:
                    train_data_aug = pd.concat(
                        [train_data_aug, train_data_tmp], ignore_index=True
                    )
                    val_data_aug = pd.concat(
                        [val_data_aug, val_data_tmp], ignore_index=True
                    )

        return train_data_aug, val_data_aug

    def set_folders(self, replace_n_fold):
        
        numbers = [num for num in range(1, self.total_folder + 1) if num not in [self.val_fold, self.n_fold]]
        random.shuffle(numbers)
                
        # get the first replace_n_fold folds of the list
        self.fakes_folders = numbers[:replace_n_fold]
        self.original_folders = numbers[replace_n_fold:]
        
        return