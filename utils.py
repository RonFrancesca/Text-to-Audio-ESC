import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob 

def get_classes():
    classes_list = ["air_conditioner", 
                   "car_horn", 
                   "children_playing",
                   "dog_bark", 
                   "drilling", 
                   "engine_idling", 
                   "gun_shot", 
                   "jackhammer", 
                   "siren", 
                   "street_music"]
    
    return classes_list

def plot_figure(data, filename, label):
    # Plot Mel Spectrogram

    plt.figure(figsize=(12, 8))  
    # take the first audio of each frame
    plt.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Mel Spectrogram - label')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.savefig(filename)
    plt.close()
    
def save_confusion_matrix(y_true, y_pred, classes, filename):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

    

def plot_audio(data, filename):
    # Plot Mel Spectrogram

    #plt.figure(figsize=(12, 8))  
    # take the first audio of each frame
    #plt.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    #plt.colorbar(format='%+2.0f dB')
    plt.plot(data)
    plt.title('Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(f'./img/{filename}'))
    plt.close()
    
def get_class_mapping():
    class_mapping = ["0",  "1", "2", "3", "4","5","6", "7", "8", "9"]
    return class_mapping

# utils relatives to the audio
def normalize_batch(input_tensor):
	"""
        Performs 0-1 normalization
    """
	min_tensor = input_tensor.min()
	max_tensor = input_tensor.max()
	norm_tensor = (input_tensor - min_tensor)/(max_tensor - min_tensor)
	return norm_tensor

def log_mels(mels, device):
    """ Apply the log transformation to mel spectrograms.
    Args:
        mels: torch.Tensor, mel spectrograms for which to apply log.

    Returns:
        Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
    """

    #amp_to_db = AmplitudeToDB(stype="amplitude").to(device)
    #amp_to_db.amin = 1e-5  
    
    # log mel spectogram
    #plot_figure(mels[0].cpu().numpy().squeeze(), f'normalized-melspectogram_{i}')
    
    #log_output = amp_to_db(mels)
    computation_method = 'luca' #luca
    if computation_method=='luca':
        log_offset=0.001 # Avoids NaNs
        log_output = torch.log(mels + log_offset) 
    elif computation_method == 'fra':
        amp_to_db = AmplitudeToDB(stype="amplitude").to(device)
        amp_to_db.amin = 1e-5 
        log_output = amp_to_db(mels)
        #log_output = log_output.clamp(min=-50, max=80) 
        log_output = (log_output - torch.mean(log_output))/torch.var(log_output)
    
    #log_output = log_output.clamp(min=-50, max=80)  
    #plot_figure(log_output[0].cpu().numpy().squeeze(), f'log_spectogram_{i}.png')
    return log_output

def take_patch_frames(patch_lenght, sample_rate, window_size):
    
    frames_1s = int(sample_rate / window_size)
    start_frame = torch.randint(0, frames_1s, (1,))
    end_frame = start_frame + patch_lenght
    
    return start_frame, end_frame

def get_transformations(config):
    
    transformation = MelSpectrogram(
            sample_rate=config["feats"]["sample_rate"], 
            n_fft=config["feats"]["n_window"],
            win_length=config["feats"]["n_window"],
            hop_length=config["feats"]["hop_length"],
            f_min=config["feats"]["f_min"],
            f_max=config["feats"]["f_max"],
            n_mels=config["feats"]["n_mels"],
            # window_fn=torch.hamming_window,
            # wkwargs={"periodic": False},
            # power=1
    )
    
    return transformation

def collect_generated_metadata(
    metadata_fold, 
    test_fold, 
    val_fold):
    
    
    audio_gen_df = pd.DataFrame(columns=['slice_file_name', 'class', 'classid'])
    
    fold_folders = [folder for folder in os.listdir(metadata_fold) if folder.startswith('fold_') and int(folder.split('_')[1]) not in [test_fold, val_fold]]

    # Iterate through each fold folder
    for fold_folder in fold_folders:
        # Get the path to the CSV file in the current fold folder
        csv_file_path = glob.glob(os.path.join(metadata_fold, fold_folder, '*.csv'))
        
        # Check if a CSV file exists in the folder
        if csv_file_path:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path[0])  # Assuming there's only one CSV file per folder
            
            # Append the DataFrame to the list of DataFrames
            audio_gen_df = pd.concat([audio_gen_df, df], ignore_index=True)
    
    return audio_gen_df

def collect_val_generated_metadata(metadata_fold, val_fold):
    
    audio_gen_df = pd.DataFrame(columns=['slice_file_name', 'class', 'classid'])
    
    fold_folders = [folder for folder in os.listdir(metadata_fold) if folder.startswith('fold_') and int(folder.split('_')[1]) in [val_fold]]

    # should be only one folder
    csv_file_path = glob.glob(os.path.join(metadata_fold, fold_folders[0], '*.csv'))
        
    # Check if a CSV file exists in the folder
    if csv_file_path:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path[0])  # Assuming there's only one CSV file per folder
            
        # Append the DataFrame to the list of DataFrames
        audio_gen_df = pd.concat([audio_gen_df, df], ignore_index=True)
    
    return audio_gen_df