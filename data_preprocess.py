import copy
import torch
import torchaudio
from urban_sound_dataset import process_audio
from utils import get_transformations, log_mels
from tqdm import tqdm
# m amount of samples (or pixels) over all previous badges
# n amount of samples in new incoming batch
# mu1 previous mean
# mu2 mean of current batch
# v1 previous variance
# v2 variance of current batch
#https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
def combine_means(mu1, mu2, m, n):
    
    # Updates old mean mu1 from m samples with mean mu2 of n samples.
    # Returns the mean of the m+n samples.
    
    return (m / (m+n)) * mu1 + (n/(m+n))*mu2
def combine_vars(v1, v2, mu1, mu2, m, n):
    
    # Updates old variance v1 from m samples with variance v2 of n samples.
    # Returns the variance of the m+n samples.
    
    return (m/(m+n)) *v1 + n/(m+n) *v2 + m*n/(m+n)**2 * (mu1 - mu2)**2

def extract_stat_data(files_list, config, sample_rate, num_samples):
    
    # Computes running mean and std dev of log-mel spec data
    # Args
    #     file_list: list of audiofiles (global paths)
    # Returns
    #     mean: Float, mean of data
    #     var: Float, standard deviation of data
    
    # Cycle through audio files (GLOBAL PATHS)
    for i in tqdm(range(len(files_list))):
        
        # Load one audio waveform
        waveform = process_audio(files_list[i], sample_rate, num_samples)
        transformations = get_transformations(config)
        
        # Compute (desired) log-mel spectrogram (ADAPT TO DESIRED AUDIO PIPELINE)
        mel_spec = transformations(waveform)
        mel_spec = log_mels(mel_spec, 'cpu')
        
        # Handle first iteration (i.e. compute mean and var of first vile)
        if i == 0:
            mean_curr = torch.mean(mel_spec) # Mean
            var_curr = torch.var(mel_spec) # Variance
            N_curr = len(mel_spec.ravel()) # Number of samples (i.e. N_mels X N_frames)
        else:
            # Compute mean and variance online following https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
            N_new = len(mel_spec.ravel()) # Number of samples of new spectrogram
            mean_new = combine_means(mean_curr,torch.mean(mel_spec),N_curr,N_new) # Mean of previous spectrograms + new
            var_new = combine_vars(var_curr,torch.var(mel_spec),mean_curr,torch.mean(mel_spec),N_curr,N_new) # Var of previous spectrograms + new
            # Update running mean and variance
            mean_curr, var_curr = copy.deepcopy(mean_new), copy.deepcopy(var_new)
            # Update total number of samples
            N_curr += N_new
    print(f"Mean: {mean_curr}\nStandar deviation: {var_curr}")
    return mean_curr, var_curr