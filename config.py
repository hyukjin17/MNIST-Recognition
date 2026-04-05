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
BRIGHTEN_THRESHOLD = 255 - DARKEN_THRESHOLD

# File paths
MODEL_PATH = './results/model.pth'
OPTIMIZER_PATH = './results/optimizer.pth'
LOSS_PLOT_PATH = './results/training_test_loss.png'
TEST_DIGITS_PATH = './results/9_digits_prediction.png' # image grid of predictions (first 9 digits in test set)
HANDWRITING_DIR = './data/Handwritten'
HANDWRITING_PREDICTIONS_PATH = './results/custom_writing_predictions.png'
FILTER_VIS_PATH = './results/filter_visualizations.png'
FILTERED_IMAGE_PATH = './results/filtered_images.png'
GREEK_TRAIN_DIR = './data/greek_train'
GREEK_MODEL_PATH = './results/greek_model.pth'
GREEK_LOSS_IMAGE_PATH = './results/greek_training_loss.png'
GREEK_TEST_DIR = './data/greek_test'
GREEK_PREDICTIONS_PATH = './results/greek_predictions.png'
GREEK_INVERTED_DIR = './data/greek_inverted'