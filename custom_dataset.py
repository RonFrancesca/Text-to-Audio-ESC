import os 

from torch.utils.data import Dataset
import torchaudio
import torch

import pandas as pd

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_files, audio_dir, transformation, target_sr):
        self.annotations = pd.read_csv(annotations_files)
        self.audio_dir = audio_dir
        self.transformation = transformation #callable object 
        self.target_sr = target_sr

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sr:
            resample = torchaudio.transforms.Resample(sr, self.target_sr) #callable object
            signal = resample(signal)
        return signal 

    def _mix_down_if_necessary(self, signal):
        """mix down a signal with multiple channels into a single channel. 
        """
        # signal -> (number_channels, samples) -> (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":

    ANNOTATIONS_FILES = "/Users/francescaronchini/Desktop/Corsi/thesoundofai/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/francescaronchini/Desktop/Corsi/thesoundofai/data/UrbanSound8K/audio"
    SAMPLE_RATE = 16000


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    usd = UrbanSoundDataset(ANNOTATIONS_FILES, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[0]

    print(signal, label)
    

    



    