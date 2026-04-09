# MNIST/FashionMNIST Image Recognition & Optimization Pipeline

Author: Hyuk Jin Chung
Date: April 2026
Language: Python (PyTorch)

## Overview
This repository contains a comprehensive PyTorch framework for training, optimizing, and testing neural networks on the MNIST and FashionMNIST datasets. It features a dynamically configurable Convolutional Neural Network (CNN), a custom Vision Transformer (ViT), advanced hyperparameter optimization scripts (Taguchi L9 and Sequential Linear Search), and a robust transfer learning pipeline for evaluating custom handwritten data (Greek letters).

## Features
* **Dynamic CNN Architecture:** A flexible CNN class (`cnn.py`) that allows for rapid adjustment of layers, filter sizes, stride, pooling, and node counts without hardcoding dimensions.
* **Vision Transformer (ViT):** A fully implemented Vision Transformer (`transformer.py`) with patch embeddings, positional encoding, and self-attention, designed specifically for smaller datasets like MNIST/FashionMNIST.
* **Systematic Hyperparameter Optimization:**
    * **Taguchi L9 Orthogonal Array:** Tests multiple parameters simultaneously in only 9 runs to find macro-architectural baselines.
    * **Sequential Linear Search:** Automates the fine-tuning of micro-architecture, optimizers (Adam/AdamW/SGD), and regularization strategies.
* **Transfer Learning:** Freezes base features and swaps the classification head to learn entirely new classes (e.g., Greek letters) on small datasets.
* **Custom Handwriting Recognition:** Uses an OpenCV preprocessing pipeline (inversion, thresholding, padding, resizing) to test the network on real-world photos of handwritten digits and letters.
* **Hardware Acceleration:** Automatically routes tensor math to CUDA (NVIDIA), MPS (Apple Silicon), or CPU depending on available hardware.

---

## Repository Structure

### Configuration
* `config.py`: The central hub for all global variables, hyperparameters, file paths, and hardware device configurations. Change dataset types (`DATA_TYPE = 'mnist'` or `'fashion_mnist'`) and base epochs here.

### Convolutional Neural Network (CNN)
* `cnn.py`: The dynamic neural network class definition.
* `train_cnn.py`: Trains the base CNN, evaluates test loss per epoch, and generates training vs. test loss plots.
* `test_cnn.py`: Evaluates the saved CNN model on the full test dataset and visually plots predictions for the first 9 images.

### Vision Transformer (ViT)
* `transformer.py`: Contains the `PatchEmbedding` module, `NetTransformer` model, and `NetConfig` dataclass.
* `train_transformer.py`: Training script for the Vision Transformer model using AdamW.
* `test_transformer.py`: Evaluates the saved Transformer model on the full test dataset.

### Hyperparameter Optimization
* `cnn_L9_experiment.py`: Runs a 9-experiment Taguchi orthogonal array to test varying combinations of Convolutional Layers, Filter Size, Batch Size, and Dropout Rate.
* `run_linear_search.py`: Uses the locked-in baseline from the L9 experiment to run an automated, sequential sweep of all remaining parameters (filters, dense nodes, activation functions, pooling, optimizers, learning rate, and weight decay), ultimately saving the "Champion" model.

### Transfer Learning & Custom Data
* `retrain_network_greek.py`: Loads the pre-trained base CNN, freezes the feature layers, modifies the output layer to 3 nodes, and trains the model to recognize custom Greek letters (Alpha, Beta, Gamma).
* `test_greek.py`: Tests the retrained model on a separate set of handwritten Greek letters.
* `test_custom_handwriting.py`: Tests the base CNN on 10 user-provided photos of handwritten digits (0-9) using a custom OpenCV preprocessing pipeline.

---

## Setup & Installation

**Prerequisites:**
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install torch torchvision opencv-python matplotlib numpy
```

## Directory Requirements
The scripts will automatically generate ./data and ./results directories, but to use the custom handwriting and Greek letter scripts, you must structure your directories as follows:

/data
  /Handwritten
    digit_0.png
    digit_1.png
    ...
  /greek_train
    /alpha
    /beta
    /gamma
  /greek_test
    /alpha
    /beta
    /gamma

---

## Usage Guide

1.  **Training the Base Models**
To train and test the standard CNN on the dataset specified in config.py:
```bash
python3 train_cnn.py
python3 test_cnn.py
```

To train and evaluate the Vision Transformer:
```bash
python3 train_transformer.py
python3 test_transformer.py
```

2.  **Running Hyperparameter Optimization**
If you want to find the mathematically optimal architecture for the CNN:

- Run the L9 Experiment to find the best baseline structure:
```bash
python3 cnn_L9_experiment.py
```

- Update the best_params dictionary in run_linear_search.py with the winners from the L9 experiment.
- Run the Sequential Linear Search to fine-tune the remaining variables and export the ultimate model:
```bash
python3 run_linear_search.py
```

3.  **Testing Custom Handwritten Digits**
- Write the digits 0-9 on paper or a whiteboard and take square-ratio photos.
- Name them digit_0.png through digit_9.png and place them in ./data/Handwritten/.
- Run the custom test script (OpenCV will automatically threshold, invert, and format the images for the network):
```bash
python3 test_custom_handwriting.py
```

4.  **Transfer Learning (Greek Letters)**
- Ensure your base CNN has been trained and saved (python3 train_cnn.py).
- Place square-ratio photos of Greek letters into the ./data/greek_train/ and ./data/greek_test/ folders, sorted into subfolders by class (alpha, beta, gamma).
- Run the transfer learning script to retrain the final layer:
```bash
python3 retrain_network_greek.py
```
- Test the retrained network:
```bash
python3 test_greek.py
```