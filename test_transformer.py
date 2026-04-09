"""
Hyuk Jin Chung
3/29/2026

Tests the transformer model on the test set of MNIST data
"""

import torch
import sys
from transformer import NetConfig, NetTransformer
from test_cnn import load_test_data, evaluate
from config import BATCH_SIZE_TEST, TRANSFORMER_MODEL_PATH


def main(argv):
    """
    Main function:
    - Evaluate the Transformer model on the test dataset
    - Show outputs of first 10 predictions
    - Visualize first 9 predictions
    """
    config = NetConfig()
    device = torch.device(config.device)
    print(f"\n[Hardware] Testing on: {device}")
    
    test_loader = load_test_data(BATCH_SIZE_TEST, config.dataset)
    network = NetTransformer(config).to(device)

    model_path = TRANSFORMER_MODEL_PATH
    try:
        # Use map_location to ensure the loaded weights map correctly to GPU
        network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Please train the model first.")
        sys.exit(1)
    
    # Run evaluations
    evaluate(network, test_loader, device)

if __name__ == "__main__":
    main(sys.argv)