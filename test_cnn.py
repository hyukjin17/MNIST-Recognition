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
from config import BATCH_SIZE_TEST


def load_test_data(batch_size=1000):
    """Downloads the MNIST dataset, preprocesses the data, and loads it into DataLoaders"""
    # Normalize pixel values centered around 0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Download test data only
    os.makedirs('./data', exist_ok=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)

    # Wrap data into DataLoaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader


def evaluate(network, test_loader):
    """Evaluate the model on the entire test set to calculate overall accuracy"""
    network.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum') # Negative Log Likelihood loss function
    # use 'sum' for test loss
    
    # No gradient update for test mode
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += criterion(output, target).item()
            # indices of the max output = prediction
            pred = output.max(1)[1]
            correct += pred.eq(target).sum()
            
    test_loss /= len(test_loader.dataset) # average loss
    
    print('\nFull Test Set Evaluation:')
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def evaluate_first_ten(network, test_loader):
    """
    - Evaluate just the first 10 test examples and print out detailed outputs and predictions
    - Plot the first 9 digits in a 3x3 grid with predictions
    """
    network.eval()
    
    # Only grab the first batch of test data
    example_data, example_targets = next(iter(test_loader))
    
    # Grab the first 10 images and targets from the batch
    data_top10 = example_data[:10]
    target_top10 = example_targets[:10]
    
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
    save_path = './results/9_digits_prediction.png'

    # Plot the first 9 images in a 3x3 grid
    fig = plt.figure(num='Predictions', figsize=(8, 8))
    for i in range(9):
        pred_class = output[i].argmax().item()
        
        plt.subplot(3, 3, i+1)
        plt.imshow(data_top10[i][0], cmap='gray', interpolation='none')
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
    batch_size_test = BATCH_SIZE_TEST
    test_loader = load_test_data(batch_size_test)
    network = CNN()

    model_path = './results/model.pth'
    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Please train the model first.")
        sys.exit(1)
    
    # Run evaluations
    evaluate(network, test_loader)
    evaluate_first_ten(network, test_loader)

if __name__ == "__main__":
    main(sys.argv)