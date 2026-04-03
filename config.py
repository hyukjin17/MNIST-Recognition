"""
Hyuk Jin Chung
3/29/2026

Configuration file for global variables and hyperparameters
"""

# Hyperparameters / global variables
N_EPOCHS = 3
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
RANDOM_SEED = 1
DARKEN_THRESHOLD = 155

# File paths
MODEL_PATH = './results/model.pth'
OPTIMIZER_PATH = './results/optimizer.pth'
LOSS_PLOT_PATH = './results/training_test_loss.png'
TEST_DIGITS_PATH = './results/9_digits_prediction.png' # image grid of predictions (first 9 digits in test set)
HANDWRITING_DIR = './data/Handwritten'
HANDWRITING_PREDICTIONS_PATH = './results/custom_writing_predictions.png'