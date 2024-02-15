from tqdm import tqdm
import torch
import torchaudio
import os 
from utils import get_transformations

from utils import plot_figure, get_transformations, log_mels, take_patch_frames

def process_audio_GPU(
    inputs, 
    config, 
    device, 
    patch_lenght, 
    sample_rate, 
    window_size
):
    
    if config['processing'] == 'GPU':
            # mel_spectogram
            transformation = get_transformations(config).to(device)
            inputs = transformation(inputs)
        
            inputs = log_mels(inputs, device)

            inputs = (inputs - torch.mean(inputs))/torch.var(inputs)

    start_frame, end_frame = take_patch_frames(patch_lenght, sample_rate, window_size)
    inputs = inputs[:, :, :, start_frame:end_frame]
    
    return inputs


def train_one_epoch(
        model, 
        config,
        data_loader,
        loss_fn, 
        optimizer, 
        device, 
        patch_lenght, 
        sample_rate, 
        window_size,
        img_folder
    ):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    
    # get the transformation 
    #transformation = get_transformations(config).to(device)
    running_loss = 0.
    model.train(True)

    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs = process_audio_GPU(inputs, config, device, patch_lenght, sample_rate, window_size)
        
        # plot the image ##
        # label = targets[0].cpu().item()
        # filename = os.path.join(img_folder, f'network_input_{i}_label_{label}_cpu')
        # plot_figure(inputs[0].cpu().numpy().squeeze(), filename, label)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # make prediction for this batch
        predictions = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(predictions, targets)
        loss.backward()

        # adjust learning weights
        optimizer.step()

        running_loss += loss.detach().item()
    
    return running_loss / num_batches

def val_one_epoch(
        model, 
        config,
        data_loader, 
        loss_fn, 
        device, 
        patch_lenght, 
        sample_rate, 
        window_size, 
        img_folder, 
        mode
    ):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    
    #transformation = get_transformations(config).to(device)
    running_loss = 0.
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader)):
            
            if mode == 'f':
                inputs = torch.reshape(inputs,(-1,1,128,128))
                targets = targets.ravel()
            
            inputs, targets = inputs.to(device), targets.to(device)
        
            inputs = process_audio_GPU(inputs, config, device, patch_lenght, sample_rate, window_size)
            ## plot the image ##
            # label = targets[0].cpu().item()
            # filename = os.path.join(img_folder, f'network_input_{i}_label_{label}_validation_cpu')
            # plot_figure(inputs[0].cpu().numpy().squeeze(), filename, label)

            # make prediction for this batch
            predictions = model(inputs)

            # Compute the loss 
            loss = loss_fn(predictions, targets.to(torch.int64))
            running_loss += loss.detach().item()
        
    return running_loss / num_batches

def train(model, 
          config,
          train_data_loader, 
          val_data_loader,  
          loss_fn,
          optimizer, 
          n_epochs, 
          device, 
          checkpoint_folder, 
          writer, 
          img_folder,
          patch_lenght, 
          sample_rate, 
          window_size,
          mode='a',
          early_stop_patience=15, 
          checkpoint_filename = "urban-sound-cnn.pth", 
    ):
    
    best_epoch = 0
    
    for n_epoch in tqdm(range(n_epochs)):
        
        print(f"Epoch: {n_epoch+1}")
        
        # training epoch
        train_loss = train_one_epoch(model, config, train_data_loader, loss_fn, optimizer, device, patch_lenght, sample_rate, window_size, img_folder)
        print(f"Train_loss: {train_loss:.2f}")
        
        val_loss = val_one_epoch(model, config, val_data_loader, loss_fn, device, patch_lenght, sample_rate, window_size, img_folder, mode)
        print(f"Val_loss: {val_loss:.2f}")
        
        # adding training and validation loss to tensorboard writer
        writer.add_scalar('Loss/train', train_loss, n_epoch)
        writer.add_scalar('Loss/val', val_loss, n_epoch)
        
        # Handle saving best model + early stopping
        if n_epoch == 0:
            val_loss_best = val_loss
            early_stop_counter = 0
            saved_model_path = os.path.join(checkpoint_folder, checkpoint_filename)
            torch.save(model.state_dict(), saved_model_path)
        
        if n_epoch > 0 and val_loss < val_loss_best:
            saved_model_path = saved_model_path
            torch.save(model.state_dict(), saved_model_path)
            val_loss_best = val_loss
            early_stop_counter = 0
            best_epoch = n_epoch
        
        else:
            early_stop_counter += 1
            print(f'Patience status: {early_stop_counter}/{early_stop_patience}')

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print(f'Training finished at epoch: {n_epoch}')
            break
    
    print("Training is done!")
    return train_loss, val_loss_best, best_epoch