import torch
from main import FeedForwardNet, download_mnist_datasets

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
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feed_forward_net.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation set 
    _, validation_set = download_mnist_datasets()

    # get a sample from validation set for inference
    input, target = validation_set[0][0], validation_set[0][1]


    # make an inference
    predicted, excepted = predict(feed_forward_net, input, target, class_mapping)
    print(f"Predicted: {predicted}, expected: {excepted}")

