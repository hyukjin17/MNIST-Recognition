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
from torchvision import transforms
import matplotlib.pyplot as plt
from cnn import CNN
from retrain_network_greek import GreekTransform
from config import GREEK_MODEL_PATH, GREEK_TEST_DIR, GREEK_PREDICTIONS_PATH, BRIGHTEN_THRESHOLD, GREEK_INVERTED_DIR

def test_greek(model_path=GREEK_MODEL_PATH, image_folder=GREEK_TEST_DIR):
    """
    - Test the model's performance on the custom handwritten data
    - Plot the data to visualize predictions
    """
    network = CNN()
    network.fc2 = torch.nn.Linear(network.fc2.in_features, 3) # Swap last layer (3 output nodes)
    network.load_state_dict(torch.load(model_path))
    network.eval() # disable dropout

    # Use the same normalization step as during training
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    fig = plt.figure(num='Handwritten Greek Letters', figsize=(9, 4))
    classes = ['alpha', 'beta', 'gamma']
    
    # Loop through the images in the folder
    for index, name in enumerate(classes):
        img_path = os.path.join(image_folder, f'{name}.png')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # Catch any error with image access
        if img is None:
            print(f"Could not find image: {img_path}.")
            print(f"Make sure all 3 images  with correct naming are in the {image_folder} directory.")
            sys.exit(1)

        # Convert BGR (OpenCV default) to RGB (PyTorch standard)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Brighten pixels above the BRIGHTEN_THRESHOLD to 255
        img = np.where(img > BRIGHTEN_THRESHOLD, 255, img).astype(np.uint8)
        # Save the image for reference
        os.makedirs(GREEK_INVERTED_DIR, exist_ok=True)
        save_path = os.path.join(GREEK_INVERTED_DIR, f'{name}_inv.png')
        cv2.imwrite(save_path, img)

        # Pad the image with black pixels to make a perfect square (if the input wasn't already a square)
        # Get height and width of image
        h, w, _ = img.shape
        # Calculate how much black border to add to the shorter sides
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
        
        # Normalize the values, convert to grayscale, crop, and invert image
        img_tensor = tensor_transform(img)
        # Convert 3D into 4D tensor (1, 1, 128, 128)
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = network(img_tensor)
            
        prediction_idx = output.argmax().item()
        prediction_name = classes[prediction_idx]
        
        plt.subplot(1, 3, index+1)
        plt.imshow(img_tensor[0][0], cmap='gray', interpolation='none')
        plt.title(f"Predicted: {prediction_name}")
        plt.xticks([])
        plt.yticks([])
            
    plt.tight_layout()
    fig.savefig(GREEK_PREDICTIONS_PATH)
    plt.show()

if __name__ == "__main__":
    # Create the folder for custom digits if it doesn't exist
    os.makedirs(GREEK_TEST_DIR, exist_ok=True)
    
    # Check if the folder is completely empty (exit program if empty)
    if len(os.listdir(GREEK_TEST_DIR)) == 0:
        print(f"ERROR: The folder '{GREEK_TEST_DIR}' is empty.")
        print("Please place your 3 cropped images ('alpha.png', 'beta.png', 'gamma.png') inside the folder and run this script again.")
        sys.exit(1)
        
    # Run the script
    test_greek()