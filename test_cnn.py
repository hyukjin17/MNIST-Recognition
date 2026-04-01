# Hyuk Jin Chung
# 3/29/2026
#
# Evaluate the CNN model on the test dataset

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
from cnn import CNN

# Downloads the MNIST dataset, preprocesses the data, and loads it into DataLoaders
def load_test_data(batch_size=1000):
    # Normalize pixel values centered around 0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Download test data only
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)

    # Wrap data into DataLoaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader


def main(argv):
    batch_size_test = 1000
    test_loader = load_test_data(batch_size_test)
    network = CNN()

    model_path = './results/model.pth'
    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Please train the model first.")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)