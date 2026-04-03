"""
Hyuk Jin Chung
4/03/2026

Visualizes the filter outputs of the first layer of CNN and the filtered images
"""

import cv2
import sys
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from cnn import CNN
from config import FILTER_VIS_PATH, MODEL_PATH, FILTERED_IMAGE_PATH

def visualize_network(weights):
    """Visualize the weights of the first convolution layer"""
    # Print the weights of conv1
    print("First Layer Weights:")
    print(f"Shape of conv1 weights: {weights.shape}")
    print(f"Raw weights of filter 0:\n{weights[0, 0]}\n")

    # Plot weights in a grid
    fig = plt.figure(num='Conv1 Weights', figsize=(10, 8))    
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.imshow(weights[i, 0], cmap='plasma', interpolation='none')
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    fig.savefig(FILTER_VIS_PATH)
    plt.show()


def apply_filters(weights):
    """Visualize the filters and filtered images of the first convolution layer"""
    # Load the training data and grab the first image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    training_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    first_image = training_data[0][0].numpy()[0]

    fig = plt.figure(num='Filtered Images', figsize=(7, 8))
    for i in range(10):
        kernel = weights[i, 0]
        
        # Apply filter to the image
        filtered_image = cv2.filter2D(src=first_image, ddepth=-1, kernel=kernel)
        
        # Plot weights
        plt.subplot(5, 4, 2*i+1)
        plt.imshow(weights[i, 0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        # Plot the filtered image
        plt.subplot(5, 4, 2*i+2)
        plt.imshow(filtered_image, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    fig.savefig(FILTERED_IMAGE_PATH)
    plt.show()


def main(argv):
    """
    Main function:
    - Loads the model and prints the model architecture
    - Visualize weights of conv1
    - Visualize filtered training image
    """
    network = CNN()
    network.load_state_dict(torch.load(MODEL_PATH))
    network.eval() # disable dropout

    # Print the model architecture
    print("Model Architecture:")
    print(network)
    print("\n")

    with torch.no_grad():
        weights = network.conv1.weight.numpy()

    visualize_network(weights)
    apply_filters(weights)

if __name__ == "__main__":
    main(sys.argv)