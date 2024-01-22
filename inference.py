from tqdm import tqdm
import torch
from utils import log_mels, take_patch_frames, get_class_mapping, get_transformations
from sklearn.metrics import accuracy_score

def inference(model, 
    test_data_loader, 
    device, 
    ):
    
    target_labels = []
    predicted_labels = []
    
    with torch.no_grad():
            
        for i, (inputs, labels) in enumerate(tqdm(test_data_loader)):  
            inputs = torch.reshape(inputs,(-1,1,128,128))
            labels = labels.ravel().to(torch.int64)
            inputs, labels = inputs.to(device), labels.to(device)

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
            
            
        
        # calculate accuracy
        accuracy = accuracy_score(target_labels, predicted_labels)
    
    return accuracy

            