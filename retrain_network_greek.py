"""
Hyuk Jin Chung
4/03/2026

Retrain the network to learn greek letters (alpha, beta, and gamma)
"""

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from cnn import CNN
from config import GREEK_TRAIN_DIR, MODEL_PATH, LEARNING_RATE, MOMENTUM, GREEK_MODEL_PATH, GREEK_LOSS_IMAGE_PATH

# Greek data set transform
class GreekTransform:
    """Class to transform input images for the network retraining"""
    def __init__(self):
        pass

    # Transform to grayscale, scale, crop, and invert images
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def train_greek_network(model_path=MODEL_PATH, greek_data_path=GREEK_TRAIN_DIR):
    """Retrain the existing network to recognize 3 handwritten greek letters instead of numbers"""
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(greek_data_path,
                                         transform = torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             GreekTransform(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size = 5,
        shuffle = True)

    network = CNN()
    network.load_state_dict(torch.load(model_path))
    # Freeze network
    for param in network.parameters():
        param.requires_grad = False
    # Swap last layer (3 output nodes)
    network.fc2 = torch.nn.Linear(network.fc2.in_features, 3)
    
    print("Modified Network Architecture:")
    print(network)

    # Only pass the parameters of the new, unfrozen layer to the optimizer
    optimizer = optim.SGD(network.fc2.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Modify number of epochs to compare training results
    epochs = 15
    training_losses = []
    network.train()
    criterion = nn.NLLLoss() # Negative Log Likelihood loss function
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = network(data)
            
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()

            # Add the sum of losses in the batch to the epoch loss
            current_batch_size = data.size(0)
            batch_sum = loss.item() * current_batch_size
            epoch_loss += batch_sum
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
        # Record average loss for the epoch
        epoch_loss /= len(greek_train.dataset)
        training_losses.append(epoch_loss)
        accuracy = 100. * correct / len(greek_train.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.0f}%")

    # Plot the error
    fig = plt.figure(num='Training Loss (Greek Letters)', figsize=(8, 5))
    plt.plot(range(1, epochs + 1), training_losses, color='blue', marker='o')
    plt.title('Training Loss (Greek Letters)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (NLL)')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(GREEK_LOSS_IMAGE_PATH)
    plt.show()
    
    # Save the new weights
    torch.save(network.state_dict(), GREEK_MODEL_PATH)

if __name__ == "__main__":
    train_greek_network()