"""
Hyuk Jin Chung
3/29/2026

Tests the retrained model using the handwritten greek letters (images of letters written on paper/whiteboard)

Prerequisite:
- Make sure there are 3 images of custom handwritten greek letters (alpha, beta, gamma) in the GREEK_TEST_DIR
- Images should be square aspect ratio (or close to a square for the network to recognize the image)
- Name the files using this template: alpha.png, beta.png, gamma.png
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from cnn import CNN
from retrain_network_greek import GreekTransform
from config import GREEK_MODEL_PATH, GREEK_TEST_DIR, GREEK_PREDICTIONS_PATH, BRIGHTEN_THRESHOLD, GREEK_INVERTED_DIR

def custom_image_loader(path):
    """Loads an image with OpenCV and applies threshold brightening and padding before PyTorch transforms are applied"""
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    # Catch any error with image access
    if img is None:
        raise FileNotFoundError(f"Could not find image: {path}")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply threshold brightening
    img = np.where(img > BRIGHTEN_THRESHOLD, 255, img).astype(np.uint8)

    # Pad the image with white pixels to make a perfect square (if the input wasn't already a square)
    # Get height and width of image
    h, w, _ = img.shape
    # Calculate how much white border to add to the shorter sides
    max_dim = max(h, w)
    # Make sure padding is added evenly around
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    # Apply the white border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Resize the image to 128x128
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Convert the NumPy array into a PIL Image (torchvision.transforms.ToTensor() expects a PIL Image)
    return Image.fromarray(img)


def load_test_data(batch_size=5):
    """Preprocesses the test data and loads it into DataLoaders"""
    # Normalize pixel values centered around 0
    # Use the same normalization step as during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Wrap data into DataLoaders
    greek_train = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root = GREEK_TEST_DIR,
            loader = custom_image_loader,
            transform = transform),
        batch_size = batch_size,
        shuffle = True
    )

    return greek_train


def test_greek(network, test_loader):
    """Test the model's performance on the custom handwritten data"""
    network.eval() # disable dropout
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
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
    test_loss /= len(test_loader.dataset) # average loss
    
    print('\nFull Test Set Evaluation:')
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(argv):
    """
    Main function:
    - Evaluate the model on the test dataset (greek letters)
    """

    test_loader = load_test_data()
    network = CNN()
    network.fc2 = torch.nn.Linear(network.fc2.in_features, 3) # Swap last layer (3 output nodes)
    model_path = GREEK_MODEL_PATH

    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Please train the model first.")
        sys.exit(1)

    # Create the folder for custom digits if it doesn't exist
    os.makedirs(GREEK_TEST_DIR, exist_ok=True)
    
    # Check if the folder is completely empty (exit program if empty)
    if len(os.listdir(GREEK_TEST_DIR)) == 0:
        print(f"ERROR: The folder '{GREEK_TEST_DIR}' is empty.")
        print("Please place your cropped images inside the folders (alpha, beta, gamma) and run this script again.")
        sys.exit(1)
        
    # Run the script
    test_greek(network, test_loader)

if __name__ == "__main__":
    main(sys.argv)