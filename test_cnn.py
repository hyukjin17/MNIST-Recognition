"""
Hyuk Jin Chung
3/29/2026

Evaluate the CNN model on the test dataset
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
from cnn import CNN
from config import BATCH_SIZE_TEST, MODEL_PATH, TEST_DIGITS_PATH, DATA_TYPE


def load_test_data(batch_size=BATCH_SIZE_TEST, data_type='mnist'):
    """Downloads the MNIST dataset, preprocesses the data, and loads it into DataLoaders"""

    # Download training data only
    os.makedirs('./data', exist_ok=True)
    if data_type.lower() == 'fashion_mnist': # Download FashionMNIST data
        # Normalize pixel values centered around 0
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        test_data = datasets.FashionMNIST(root="./data/", train=False, download=True, transform=transform)
    else: # Default to standard MNIST
        # Normalize pixel values centered around 0
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)

    # Wrap data into DataLoaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader


def evaluate(network, test_loader, device):
    """Evaluate the model on the entire test set to calculate overall accuracy"""
    network.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum') # Negative Log Likelihood loss function
    # use 'sum' for test loss
    
    # No gradient update for test mode
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # push data to GPU

            output = network(data)
            test_loss += criterion(output, target).item()
            # indices of the max output = prediction
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
    test_loss /= len(test_loader.dataset) # average loss
    
    print('\nFull Test Set Evaluation:')
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def evaluate_first_ten(network, test_loader, device):
    """
    - Evaluate just the first 10 test examples and print out detailed outputs and predictions
    - Plot the first 9 digits in a 3x3 grid with predictions
    """
    network.eval()
    
    # Only grab the first batch of test data
    example_data, example_targets = next(iter(test_loader))
    
    # Grab the first 10 images and targets from the batch and push to GPU
    data_top10 = example_data[:10].to(device)
    target_top10 = example_targets[:10].to(device)
    
    with torch.no_grad():
        output = network(data_top10)
    
    # Print out detailed output values
    print('\nFirst 10 Examples Detailed Output:')
    for i in range(10):
        # Convert the output row to a list of floats
        out_vals = [f"{val:.2f}" for val in output[i].tolist()]
        
        pred_class = output[i].argmax().item()
        true_class = target_top10[i].item()
        
        print(f"Example {i}:")
        print(f" Outputs: [{', '.join(out_vals)}]")
        print(f" Predicted: {pred_class}\t|\tActual: {true_class}")
        print("-" * 40)

    os.makedirs('./results', exist_ok=True)
    save_path = TEST_DIGITS_PATH

    # Plot the first 9 images in a 3x3 grid
    fig = plt.figure(num='Predictions', figsize=(8, 8))
    for i in range(9):
        pred_class = output[i].argmax().item()
        
        plt.subplot(3, 3, i+1)
        # Pull back data to RAM to plot images
        plt.imshow(data_top10[i][0].cpu(), cmap='gray', interpolation='none')
        plt.title(f"Prediction: {pred_class}")
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    # Save the plot
    fig.savefig(save_path)
    print(f"Image grid saved to {save_path}")
    plt.show()


def main(argv):
    """
    Main function:
    - Evaluate the model on the test dataset
    - Show outputs of first 10 predictions
    - Visualize first 9 predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")
    
    batch_size_test = BATCH_SIZE_TEST
    test_loader = load_test_data(batch_size_test, DATA_TYPE)
    network = CNN().to(device)

    model_path = MODEL_PATH
    try:
        network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Please train the model first.")
        sys.exit(1)
    
    # Run evaluations
    evaluate(network, test_loader, device)
    # evaluate_first_ten(network, test_loader, device)

if __name__ == "__main__":
    main(sys.argv)