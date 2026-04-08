"""
Hyuk Jin Chung
3/29/2026

Training script for the CNN model
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
from cnn import CNN
from test_cnn import load_test_data
from config import N_EPOCHS, BATCH_SIZE_TEST, BATCH_SIZE_TRAIN, LEARNING_RATE, MOMENTUM, LOG_INTERVAL, RANDOM_SEED, LOSS_PLOT_PATH, MODEL_PATH, OPTIMIZER_PATH, DATA_TYPE, DEVICE


def load_train_data(batch_size=64, data_type='mnist'):
    """Downloads the MNIST dataset, preprocesses the data, and loads it into DataLoaders"""

    # Download training data only
    os.makedirs('./data', exist_ok=True)
    if data_type.lower() == 'fashion_mnist': # Download FashionMNIST data
        # Normalize pixel values centered around 0
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        training_data = datasets.FashionMNIST(root="./data/", train=True, download=True, transform=transform)
    else: # Default to standard MNIST
        # Normalize pixel values centered around 0
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        training_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    
    # Wrap data into DataLoaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    return train_loader


def train_network(epoch, network, optimizer, train_loader, log_interval, train_losses, train_counter, device):
    """Train the network and record losses"""
    network.train() # enable dropout
    criterion = nn.NLLLoss() # Negative Log Likelihood loss function

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # reset gradients
        output = network(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            # Record the losses for plotting later
            train_losses.append(loss.item())
            train_counter.append((batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset)))


def evaluate_test_loss(network, test_loader, test_losses, device):
    """Evaluation on test set to monitor training status (check for overfitting)"""
    network.eval() # disable dropout
    test_loss = 0
    criterion = nn.NLLLoss(reduction='sum') # Negative Log Likelihood loss function
    # use 'sum' for test loss

    # No gradient update for test mode
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += criterion(output, target).item()

    # Calculate average loss and append it to the list
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'Test set average loss: {test_loss:.4f}\n')


def plot_and_save_loss(train_counter, train_losses, test_counter, test_losses, save_path):
    """Plot the training loss and test loss, and save the plot"""
    fig = plt.figure(num='Loss Plot', figsize=(10, 6))
    
    # Plot the data
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter, test_losses, color='red', marker='o')
    
    plt.legend(['Training Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Training and Test Error over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot
    fig.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.show()


def main(argv):
    """
    Main function:
    - Train the model and plot the losses
    - Save the model weights
    """
    # Hyperparameters
    n_epochs = N_EPOCHS
    batch_size_train = BATCH_SIZE_TRAIN
    batch_size_test = BATCH_SIZE_TEST
    learning_rate = LEARNING_RATE
    momentum = MOMENTUM
    log_interval = LOG_INTERVAL

    # Create the results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    loss_save_path = LOSS_PLOT_PATH
    model_save_path = MODEL_PATH
    optimizer_save_path = OPTIMIZER_PATH

    # Use deterministic cuDNN algorithm to avoid randomness
    torch.backends.cudnn.enabled = False
    random_seed = RANDOM_SEED
    torch.manual_seed(random_seed) # use same seed for consistent output

    train_loader = load_train_data(batch_size_train, DATA_TYPE)
    test_loader = load_test_data(batch_size_test, DATA_TYPE)

    device = DEVICE
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    network = CNN().to(device) # push to GPU
    # Stochastic Gradient Descent optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # Evaluate initial test loss before running the training
    evaluate_test_loss(network, test_loader, test_losses, device)

    # Training loop
    for epoch in range(1, n_epochs + 1):
        train_network(
            epoch, 
            network, 
            optimizer, 
            train_loader, 
            log_interval, 
            train_losses, 
            train_counter,
            device
        )
        # Evaluate the network at the end of the epoch
        evaluate_test_loss(network, test_loader, test_losses, device)

    # Save the model and optimizer values as a pth file
    print("Training finished! Saving model...")
    torch.save(network.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path) # checkpoint in case further training is needed

    # Plot the training loss and test loss
    plot_and_save_loss(train_counter, train_losses, test_counter, test_losses, loss_save_path)

if __name__ == "__main__":
    main(sys.argv)