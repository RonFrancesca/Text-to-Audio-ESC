import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 

import torchaudio
from cnn import CNNNetwork

from custom_dataset import UrbanSoundDataset

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):

    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model,data_loader, loss_fn, optimiser, device)
        print(f"----------------")
    print("Training is done ")


if __name__== "__main__":

    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    print(f"Using device: {device}")


    # instatiating the dataset 
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

    usd = UrbanSoundDataset(ANNOTATIONS_FILES, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # create a data loader for the dataset 
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)
    cnn = CNNNetwork().to(device)

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), 
                                lr=LEARNING_RATE
                                )
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "feed_forward_net.pth")
    print("Model trained and stored at feed_forward_net.pth")




