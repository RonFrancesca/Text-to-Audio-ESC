import torch
import torchaudio
from custom_dataset import UrbanSoundDataset
from cnn import CNNNetwork
from main import EPOCHS, BATCH_SIZE

# should be classes as urbansound
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
    state_dict = torch.load("cnn.pth") #train model that would need to be created
    cnn.load_state_dict(state_dict)

    # load UrbanSound dataset
    ANNOTATIONS_FILES = "/Users/francescaronchini/Desktop/Corsi/thesoundofai/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/francescaronchini/Desktop/Corsi/thesoundofai/data/UrbanSound8K/audio/"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILES, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu")
    

    # get a sample from urban sound set for inference
    input, target = usd[0][0], usd[0][1]  # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)


    # make an inference
    predicted, excepted = predict(cnn, input, target, class_mapping)
    print(f"Predicted: {predicted}, expected: {excepted}")

