import torch
import os
from custom_dataset import UrbanSoundDataset
from model import CNNNetwork
import argparse
import yaml
from torchaudio.transforms import MelSpectrogram

# TODO: should be classes as urbansound

# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music


class_mapping = [
   "0",  
   "1", 
   "2", 
   "3", 
   "4",
   "5",
   "6", 
   "7", 
   "8", 
   "9"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1 (input), 10 (n classes try to predict))
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        excepted = class_mapping[target]

    return predicted, excepted

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a Audio Event Classification (AEC) syststem")
    parser.add_argument(
        "--conf_file",
        default="./config/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    
    parser.add_argument(
        "--log_dir",
        default="./exp/",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )

    args = parser.parse_args()

    
    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)
        
        
    metadata_file = config["data"]["metadata_file"]
    audio_dir = config["data"]["audio_dir"]
    sample_rate = config["feats"]["sample_rate"]
    audio_max_len = config["data"]["audio_max_len"]
    num_samples = sample_rate * audio_max_len
    
    # dataset
    usd = UrbanSoundDataset(config, num_samples)
    
    mel_spectrogram = MelSpectrogram(
        sample_rate=sample_rate, 
        n_fft=config["feats"]["n_window"],
        hop_length=config["feats"]["hop_length"],
        n_mels=config["feats"]["n_mels"]
    )
    
    checkpoint_folder = config["data"]["checkpoint_folder"]
    
    # load the model 
    model = CNNNetwork(config)
    state_dict = torch.load(os.path.join(checkpoint_folder, "urban-sound-cnn.pth")) #train model that would need to be created
    model.load_state_dict(state_dict)
    

    # get a sample from urban sound set for inference
    
    for sample in range(10):
    
        input, target = usd[sample][0], usd[sample][1]  # [batch size, num_channels, fr, time] -> Tensor of three dimensions
        input.unsqueeze_(0) # to add the extra index on the dimension that we want to introduce

        input = mel_spectrogram(input)
        # make an inference
        predicted, excepted = predict(model, input, target, class_mapping)
        print(f"Predicted: {predicted}, expected: {excepted}")

