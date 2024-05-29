from tqdm import tqdm
import torch
import os

from training import process_audio_GPU

from utils import get_class_mapping

from utils import get_transformations

from utils import plot_figure, get_transformations, log_mels, take_patch_frames


def inference(
    model,
    config,
    img_folder,
    test_data_loader,
    device,
    features,
    mode,
):

    target_labels = []
    predicted_labels = []

    with torch.no_grad():

        for i, (inputs, targets) in enumerate(tqdm(test_data_loader)):

            if mode == "f":
                inputs = torch.reshape(inputs, (-1, 1, 128, 128))
                labels = labels.ravel().to(torch.int64)

            inputs, targets = inputs.to(device), targets.to(device)

            inputs = process_audio_GPU(
                inputs,
                config,
                device,
                features.patch_samples,
                features.sr,
                features.n_window,
            )

            # plot the image ##
            # label = targets[0].cpu().item()
            # filename = os.path.join(img_folder, f'network_input_{i}_label_{label}_testing_cpu')
            # plot_figure(inputs[0].cpu().numpy().squeeze(), filename, label)

            if mode == "f":
                # frames by frames
                for i in range(inputs.shape[0]):

                    outputs_logits = model(inputs[i].unsqueeze(0))
                    outputs_logits = outputs_logits.detach()
                    predicted_classes = torch.nn.Softmax(dim=1)(outputs_logits)

                    predicted_index = predicted_classes[0].argmax(0)
                    predicted = int(get_class_mapping()[predicted_index])
                    targets = int(get_class_mapping()[labels[i]])

                    # Append true and predicted labels to lists
                    target_labels.append(targets)
                    predicted_labels.append(predicted)
            else:

                # the 3 second clip
                outputs_logits = model(inputs)
                outputs_logits = outputs_logits.detach()
                predicted_classes = torch.nn.Softmax(dim=1)(outputs_logits)

                predicted_index = predicted_classes[0].argmax(0)
                predicted = int(get_class_mapping()[predicted_index])
                labels = int(get_class_mapping()[targets])

                # Append true and predicted labels to lists

            target_labels.append(labels)
            predicted_labels.append(predicted)

    return target_labels, predicted_labels
