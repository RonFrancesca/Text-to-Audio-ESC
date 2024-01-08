import torch
import torchaudio
from custom_dataset import UrbanSoundDataset
from cnn import CNNNetwork

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
    
    # load the model 
    cnn = CNNNetwork()
    state_dict = torch.load("urban-sound-cnn.pth") #train model that would need to be created
    cnn.load_state_dict(state_dict)

    # load UrbanSound dataset
    ANNOTATIONS_FILES = "/nas/home/fronchini/urban-sound-class/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/nas/home/fronchini/urban-sound-class/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # change the dataset with the cuda or cpu device becaue I don't care about the inference on GPU
    usd = UrbanSoundDataset(ANNOTATIONS_FILES, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu")
    

    # get a sample from urban sound set for inference
    
    for sample in range(10):
    
        input, target = usd[sample][0], usd[sample][1]  # [batch size, num_channels, fr, time] -> Tensor of three dimensions
        input.unsqueeze_(0) # to add the extra index on the dimension that we want to introduce


        # make an inference
        predicted, excepted = predict(cnn, input, target, class_mapping)
        print(f"Predicted: {predicted}, expected: {excepted}")

