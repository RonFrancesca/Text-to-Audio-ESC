import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torchaudio.transforms import MelSpectrogram
import ipdb

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

    def __init__(self,
                 config,
                 annotations, 
                 num_samples,
                 ):
        
        
        self.annotations = annotations
        
        self.audio_dir = config["data"]["audio_dir"]
        
        self.paths_list = self.annotations.apply(lambda row: os.path.join(self.audio_dir, f"fold{row[5]}", row[0]), axis=1)
        
     
        self.target_sample_rate = config["feats"]["sample_rate"]
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        signal = process_audio(self.paths_list[index], self.target_sample_rate, self.num_samples)
        
        return signal, self.annotations.iloc[index, 6]
        
    

    