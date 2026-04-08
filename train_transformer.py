"""
Hyuk Jin Chung
4/05/2026

Training script for the Transformer model
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from transformer import NetConfig, NetTransformer
from test_cnn import load_test_data
from train_cnn import load_train_data, plot_and_save_loss, evaluate_test_loss, train_network
from config import BATCH_SIZE_TEST, LOG_INTERVAL, TRANSFORMER_LOSS_PLOT_PATH, TRANSFORMER_MODEL_PATH, TRANSFORMER_OPTIMIZER_PATH


def main(argv):
    """
    Main function:
    - Train the model and plot the losses
    - Save the model weights
    """
    # Hyperparameters
    batch_size_test = BATCH_SIZE_TEST
    log_interval = LOG_INTERVAL
    config = NetConfig()

    # Create the results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    loss_save_path = TRANSFORMER_LOSS_PLOT_PATH
    model_save_path = TRANSFORMER_MODEL_PATH
    optimizer_save_path = TRANSFORMER_OPTIMIZER_PATH

    # Use deterministic cuDNN algorithm to avoid randomness
    torch.backends.cudnn.enabled = False
    torch.manual_seed(config.seed) # use same seed for consistent output

    train_loader = load_train_data(config.batch_size, config.dataset)
    test_loader = load_test_data(batch_size_test, config.dataset)

    device = config.device
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    network = NetTransformer(config).to(device)
    # Stochastic Gradient Descent optimizer
    optimizer = optim.AdamW(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(config.epochs + 1)]

    # Evaluate initial test loss before running the training
    evaluate_test_loss(network, test_loader, test_losses, device)

    # Training loop
    for epoch in range(1, config.epochs + 1):
        train_network(
            epoch, 
            network, 
            optimizer, 
            train_loader, 
            log_interval, 
            train_losses, 
            train_counter,
            device
        )
        # Evaluate the network at the end of the epoch
        evaluate_test_loss(network, test_loader, test_losses, device)

    # Save the model and optimizer values as a pth file
    print("Training finished! Saving model...")
    torch.save(network.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path) # checkpoint in case further training is needed

    # Plot the training loss and test loss
    plot_and_save_loss(train_counter, train_losses, test_counter, test_losses, loss_save_path)

if __name__ == "__main__":
    main(sys.argv)