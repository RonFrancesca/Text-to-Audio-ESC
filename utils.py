import matplotlib.pyplot as plt
import os

def plot_figure(data, filename):
    # Plot Mel Spectrogram

    plt.figure(figsize=(12, 8))  
    # take the first audio of each frame
    plt.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.savefig(os.path.join(f'./img/{filename}'))
    plt.close()