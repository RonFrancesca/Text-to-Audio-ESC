import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torchaudio.transforms import MelSpectrogram


def pad_to(signal, num_samples):
    
    length_signal = signal.shape[1]
    
    # cut if necessary
    if length_signal > num_samples:
        signal = signal[:, :num_samples]
        
    # pad if necessary
    if signal.shape[1] < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
        
    return signal
    
    
def process_audio(audio_sample_path,annotation_file, target_sample_rate, num_samples, index, device):
    
    signal, sr = torchaudio.load(audio_sample_path)
    
    # resample if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    
    # make the signal mono if it is not
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
            
    # pad the signal in necessary
    signal = pad_to(signal, num_samples)
    
    return signal, annotation_file.iloc[index, 6]

class UrbanSoundDataset(Dataset):

    def __init__(self,
                 config,
                 num_samples,
                 device):
        
        if config["fast_run"]:
            self.annotations = pd.read_csv(config["data"]["metadata_file"])[:200]
        else:
            self.annotations = pd.read_csv(config["data"]["metadata_file"])
        
        self.audio_dir = config["data"]["audio_dir"]
        self.device = device
        
        sample_rate = config["feats"]["sample_rate"]
        
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        fold = f"fold{self.annotations.iloc[index, 5]}"
        audio_sample_path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        
        signal, label = process_audio(audio_sample_path, self.annotations, self.target_sample_rate, self.num_samples, index, self.device)
        
        #signal = self.transformation(signal)
        
        return signal, label

    

    