"""
Hyuk Jin Chung
3/29/2026

Plot the digits from the test dataset
"""

import os
import sys
import matplotlib.pyplot as plt
from test_cnn import load_test_data

def plot_first_six(test_loader, save_path):
    """Plots the first 6 examples with labels from the test dataset"""
    # Load only the first batch for visualization
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure(num='Test Image Grid', figsize=(8, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    # Save the plot
    fig.savefig(save_path)
    print(f"Image grid saved to {save_path}")
    plt.show()


def main(argv):
    """Main function (plot the image grid)"""
    # Only need a small batch for visualizing few example digits
    batch_size = 10

    # Create the results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    image_grid_save_path = './results/test_digits.png'

    test_loader = load_test_data(batch_size)

    # Plot the first 6 test examples with labels
    plot_first_six(test_loader, image_grid_save_path)

if __name__ == "__main__":
    main(sys.argv)