import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torchaudio.transforms import MelSpectrogram
import ipdb
import matplotlib.pyplot as plt
from utils import plot_audio, get_transformations, log_mels, take_patch_frames


def pad_to(signal, num_samples):

    length_signal = signal.shape[1]

    # cut if necessary
    if length_signal > num_samples:
        signal = signal[:, :num_samples]

    # pad if necessary
    if signal.shape[1] < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)

        # Pad signal by replicating it (LUCA)
        N_replicas = int(num_missing_samples / length_signal) + 1
        signal_padded = signal.repeat(1, N_replicas + 1)
        signal_padded = signal_padded[:, :num_samples]
        signal = signal_padded

        # plt.plot(signal[0].detach().numpy())
        # plt.savefig(os.path.join("img", f"signal_padded_{index}.png"))
        # plt.show()

    return signal


def process_audio(audio_sample_path, target_sample_rate, num_samples):

    signal, sr = torchaudio.load(audio_sample_path)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)

    # make the signal mono if it is not
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # pad the signal in necessary
    signal = pad_to(signal, num_samples)

    return signal


class UrbanSoundDataset(Dataset):

    def __init__(
        self,
        config,
        annotations,
        features,
        device,
        origin="real",
        processing="CPU",
    ):

        self.annotations = annotations
        self.origin = origin
        self.processing = config["processing"]

        if self.origin == "real":
            # original dataset
            self.audio_dir = config["audio_dir_real"]
            self.paths_list = self.annotations.apply(
                lambda row: os.path.join(self.audio_dir, f"fold{row[5]}", row[0]),
                axis=1,
            )
        elif self.origin == "fake":
            # generated dataset
            self.audio_dir = config["metadata_gen"]
            self.paths_list = list(annotations["slice_file_name"])
        elif self.origin == "aug":
            self.audio_dir = config["audio_dir_real"]
            self.paths_list = list(annotations["slice_file_name"])

        self.features = features
        self.transformation = get_transformations(self.features)
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        # audio

        if self.processing == "GPU":
            signal = process_audio(
                self.paths_list[index], self.features.sr, self.features.num_samples
            )
        else:

            if self.origin == "real" or self.origin == "fake":
                if self.features.mel_bands == 64:
                    spec_file_path = self.paths_list[index].replace(".wav", ".npy")
                else:
                    spec_file_path = self.paths_list[index].replace(".wav", "_128.npy")
            else:
                spec_file_path = self.paths_list[index]

            if not os.path.exists(spec_file_path):
                signal = process_audio(
                    self.paths_list[index], self.features.sr, self.features.num_samples
                )

                # log-mel spectogram
                signal = self.transformation(signal)
                signal = log_mels(signal, self.device)

                # normalization spectogram by spectogram
                if self.features.mean == None and self.features.std == None:
                    signal = (signal - torch.mean(signal)) / torch.var(signal)
                else:
                    signal = (signal - self.features.mean) / self.features.std

                np.save(spec_file_path, signal.numpy())
            else:
                signal = np.load(spec_file_path)
                # adding this because I have forgot to pad the data that I needed to pad before
                # was throwing an errore for the T1_all augmentation technique
                nsample = int((self.features.sr * 4) / self.features.n_window) + 1
                if signal.shape[2] > nsample:
                    signal = signal[:, :, :nsample]

                signal = torch.from_numpy(signal)

        # label
        if self.origin == "real" or self.origin == "aug":
            label = self.annotations.iloc[index, 6]
        else:
            label = self.annotations.iloc[index, 2]

        return signal, label


class UrbanSoundDataset_generated(Dataset):

    def __init__(
        self, config, annotations, num_samples, mean, std, patch_lenght_samples, device
    ):

        self.annotations = annotations
        self.audio_dir = config["data"]["audio_dir"]
        self.paths_list = list(annotations["slice_file_name"])
        self.target_sample_rate = config["feats"]["sample_rate"]
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.transformation = get_transformations(config)
        self.device = device
        self.patch_lenght_samples = patch_lenght_samples
        self.target_sample_rate = config["feats"]["sample_rate"]
        self.window_size = config["feats"]["n_window"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        signal = process_audio(
            self.paths_list[index], self.target_sample_rate, self.num_samples
        )

        # log-mel spectogram
        signal = self.transformation(signal)
        signal = log_mels(signal, self.device)
        if self.mean == None and self.std == None:
            signal = (signal - torch.mean(signal)) / torch.var(signal)
        else:
            signal = (signal - self.mean) / self.std

        start_frame, end_frame = take_patch_frames(
            self.patch_lenght_samples, self.target_sample_rate, self.window_size
        )
        signal = signal[:, :, start_frame:end_frame]

        return signal, self.annotations.iloc[index, 2]


# TODO: could be done to take from the UrbanSoundDataset directy
class UrbanSoundDatasetValTest(Dataset):

    def __init__(
        self, config, annotations, num_samples, mean, std, patch_lenght_samples, device
    ):

        self.annotations = annotations
        self.audio_dir = config["data"]["audio_dir"]
        self.paths_list = self.annotations.apply(
            lambda row: os.path.join(self.audio_dir, f"fold{row[5]}", row[0]), axis=1
        )
        self.target_sample_rate = config["feats"]["sample_rate"]
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.transformation = get_transformations(config)
        self.device = device
        self.patch_lenght_samples = patch_lenght_samples
        self.target_sample_rate = config["feats"]["sample_rate"]
        self.window_size = config["feats"]["n_window"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        signal = process_audio(
            self.paths_list[index], self.target_sample_rate, self.num_samples
        )

        # log-mel spectogram
        signal = self.transformation(signal)

        if self.mean == None and self.std == None:
            signal = (signal - torch.mean(signal)) / torch.var(signal)
        else:
            signal = (signal - self.mean) / self.std

        # start_frame, end_frame = take_patch_frames(self.patch_lenght_samples, self.target_sample_rate, self.window_size)
        # signal = signal[:, :, start_frame:end_frame]

        # consider frame by frame instead of three seconds TF-patch
        N_frames = signal.shape[-1]
        N_slices = N_frames - self.patch_lenght_samples

        # generate a tensor of 0, one for each slices in the signal
        signal_slices_stacked = torch.zeros(
            N_slices, self.patch_lenght_samples, self.patch_lenght_samples
        )

        # for each slice
        for slice in range(N_slices):
            signal_slices_stacked[slice] = signal[
                :, :, slice : slice + self.patch_lenght_samples
            ]

        classes_signal = torch.ones(N_slices) * self.annotations.iloc[index, 6]

        return signal_slices_stacked.unsqueeze(1), classes_signal
