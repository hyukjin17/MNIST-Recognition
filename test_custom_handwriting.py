"""
Hyuk Jin Chung
3/29/2026

Tests the model using the custom handwriting (images of digits written on paper/whiteboard)

Prerequisite:
- Make sure there are 10 images of custom handwritten digits (0-9) in the HANDWRITING_DIR
- Images should be square aspect ratio (or close to a square for the network to recognize the image)
- Name the files using this template: digit_#.png (e.g. digit_0.png, digit_1.png, ...)
"""

import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from cnn import CNN
from config import MODEL_PATH, HANDWRITING_DIR, HANDWRITING_PREDICTIONS_PATH

def predict_custom_digits(model_path=MODEL_PATH, image_folder=HANDWRITING_DIR):
    """
    - Test the model's performance on the custom handwritten data
    - Plot the data to visualize predictions
    """
    network = CNN()
    network.load_state_dict(torch.load(model_path))
    network.eval() # disable dropout

    # Use the same normalization step as during training
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    fig = plt.figure(num='Handwritten Digits', figsize=(12, 5))
    
    # Loop through the 10 images in the folder
    for i in range(10):
        img_path = os.path.join(image_folder, f'digit_{i}.png')
        
        try:
            # Load the image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Invert colors (bitwise_not is same as 255-x)
            img = cv2.bitwise_not(img)
            # Save the inverted image for reference
            save_path = os.path.join(image_folder, f'digit_{i}_inv.png')
            cv2.imwrite(save_path, img)

            # Resize the image to 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalize the values and convert 3D into 4D tensor (1, 1, 28, 28)
            img_tensor = tensor_transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = network(img_tensor)
                
            prediction = output.argmax().item()
            
            plt.subplot(2, 5, i+1)
            plt.imshow(img_tensor[0][0], cmap='gray', interpolation='none')
            plt.title(f"Predicted: {prediction}")
            plt.xticks([])
            plt.yticks([])
            
        except FileNotFoundError:
            print(f"Could not find image: {img_path}. Make sure all 10 images  with correct naming are in the {image_folder} directory.")
            
    plt.tight_layout()
    fig.savefig(HANDWRITING_PREDICTIONS_PATH)
    plt.show()

if __name__ == "__main__":
    # Create the folder for custom digits if it doesn't exist
    os.makedirs(HANDWRITING_DIR, exist_ok=True)
    
    # Check if the folder is completely empty
    if len(os.listdir(HANDWRITING_DIR)) == 0:
        print(f"ERROR: The folder '{HANDWRITING_DIR}' is empty.")
        print("Please place your 10 cropped images ('digit_0.png', etc.) inside the folder and run this script again.")
        sys.exit(1) # Instantly kills the script
        
    # Run the script
    predict_custom_digits()