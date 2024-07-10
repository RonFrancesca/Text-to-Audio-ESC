import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torchaudio.transforms import (
    MelSpectrogram,
    TimeStretch,
    PitchShift,
    Spectrogram,
    InverseSpectrogram,
)
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from utils import log_mels

def get_spectrogram(waveform, n_fft=400, win_len=None, hop_len=None, power=2.0):

    spectrogram = Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )

    return spectrogram(waveform)


def plot_figure(data, filename=None):
    # Plot Mel Spectrogram

    plt.figure(figsize=(8, 4))
    # take the first audio of each frame
    plt.imshow(data, cmap="viridis", aspect="auto", origin="lower"), plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.savefig(filename)
    plt.close()


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


if __name__ == "__main__":

    metadata_file_real = "/nas/home/fronchini/EUSIPCO/urban-sound-class/UrbanSound8K/metadata/UrbanSound8K.csv"
    # metadata_file_fake= "/nas/home/fronchini/EUSIPCO/urban-sound-class/audio_generation/AUDIOGEN_gpt"
    audio_dir_real = "/nas/home/fronchini/EUSIPCO/urban-sound-class/UrbanSound8K/audio"
    # audio_dir_fake= "/nas/home/fronchini/EUSIPCO/urban-sound-class/audio_generation/AUDIOGEN_gpt"

    # parameters
    mel_bands = 128
    target_sample_rate = 16000
    num_samples = target_sample_rate * 4  # sample rate * audio_max_lenght
    n_window = 1204
    n_filters = 2048
    hop_length = 1024
    n_window = 1024
    f_min = 0
    f_max = 8000

    annotations_real = pd.read_csv(metadata_file_real)
    annotations_augmented = annotations_real.copy()

    paths_list = annotations_real.apply(
        lambda row: os.path.join(audio_dir_real, f"fold{row[5]}", row[0]), axis=1
    )
    
    pitch_shift_values = [-2, -1, 1, 2]

    mel_spectogram = MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_window,
        win_length=n_window,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=mel_bands,
    )

    for index in tqdm(range(len(paths_list))):

        # read audio
        signal = process_audio(paths_list[index], target_sample_rate, num_samples)

        ##
        # Pitch Shift is applied directly to the audio waveform
        ##

        # selecting a random values for the picth shift
        for pitch_shift_selected in pitch_shift_values[:1]:

            file_audio_PS_path = paths_list[index].replace(
                ".wav", f"_PS_{pitch_shift_selected}_{mel_bands}.npy"
            )

            if not os.path.exists(file_audio_PS_path):

                # https://pytorch.org/audio/stable/generated/torchaudio.transforms.PitchShift.html#torchaudio.transforms.PitchShift
                # using the same values of the mel spectogram for the pitch shift
                pitch_shift = PitchShift(
                    sample_rate=target_sample_rate, n_steps=pitch_shift_selected
                )
                signal_pitch_shifthed = pitch_shift(signal)

                # log-mel spectogram applied to the picth shifted signal
                signal_pitch_shifthed = mel_spectogram(signal_pitch_shifthed)
                signal_pitch_shifthed = log_mels(signal_pitch_shifthed, "GPU")

                # the file will need to be saved as npy file
                np.save(file_audio_PS_path, signal_pitch_shifthed.detach().numpy())

            index_file = paths_list[index].split("/")[-1]
            annotations_augmented.loc[
                annotations_augmented["slice_file_name"] == index_file,
                "slice_file_name",
            ] = file_audio_PS_path

    # save the metadata
    augmented_PS_annotations = f"/nas/home/fronchini/EUSIPCO/urban-sound-class/UrbanSound8K/metadata/UrbanSound8K_PS_{mel_bands}_{ps}.csv"
    annotations_augmented.to_csv(augmented_PS_annotations, index=False)
