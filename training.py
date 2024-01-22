from tqdm import tqdm
import torch
import os 

from utils import plot_figure, get_transformations, log_mels, take_patch_frames


def train_one_epoch(
        model, 
        data_loader,
        loss_fn, 
        optimizer, 
        device
    ):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    
    # get the transformation 
    #transformation = get_transformations(config).to(device)
    running_loss = 0.
    model.train(True)

    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # mel_spectogram
        # inputs = transformation(inputs)
        # plot_figure(inputs[0].cpu().numpy().squeeze(), f'melspectogram_{i}')
        
        # # normalization of mel spectogram
        # inputs = log_mels(inputs, device)
        # inputs = (inputs - mean_train) / var_train
        # #inputs = (inputs/torch.mean(inputs))/torch.var(inputs)

        # # consider only three random second patch
        # start_frame, end_frame = take_patch_frames(patch_lenght, sample_rate, window_size)
        # inputs = inputs[:, :, :, start_frame:end_frame]
        plot_figure(inputs[0].cpu().numpy().squeeze(), f'network_input_{i}')

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

def val_one_epoch(model, data_loader, loss_fn, device):
    
    num_batches = len(data_loader.dataset) / data_loader.batch_size
    
    #transformation = get_transformations(config).to(device)
    running_loss = 0.
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader)):
            
            inputs = torch.reshape(inputs,(-1,1,128,128))
            targets = targets.ravel()
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs = transformation(inputs)
            # inputs = log_mels(inputs, device)
            # inputs = (inputs - mean_train) / var_train
            
            # #inputs = (inputs/torch.mean(inputs))/torch.var(inputs)
            # start_frame, end_frame = take_patch_frames(patch_lenght, sample_rate, window_size)
            # inputs = inputs[:, :, :, start_frame:end_frame]
            #plot_figure(inputs[0].cpu().numpy().squeeze(), f'network_input_validation_{i}')

            # make prediction for this batch
            predictions = model(inputs)

            # Compute the loss 
            loss = loss_fn(predictions, targets.to(torch.int64))
            running_loss += loss.detach().item()
        
    return running_loss / num_batches

def train(model, 
          train_data_loader, 
          val_data_loader,  
          loss_fn,
          optimizer, 
          n_epochs, 
          device, 
          checkpoint_folder, 
          writer, 
          early_stop_patience=100, 
          checkpoint_filename = "urban-sound-cnn.pth", 
    ):
    
    best_epoch = 0
    
    for n_epoch in tqdm(range(n_epochs)):
        
        print(f"Epoch: {n_epoch+1}")
        
        # training epoch
        train_loss = train_one_epoch(model, train_data_loader, loss_fn, optimizer, device)
        print(f"Train_loss: {train_loss:.2f}")
        
        val_loss = val_one_epoch(model, val_data_loader, loss_fn, device)
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
    
    
    print(f"Model saved at epoch: {best_epoch}")
    print(f"Model saved at: {saved_model_path}")
    print("Training is done!")
    return train_loss, val_loss_best