"""
Hyuk Jin Chung
04/06/2026

L9 orthogonal array experiment for CNN model
- Tests 4 different independent variables (with 3 values per variable) using only 9 experiments (instead of 81 experiments)
- Uses the Taguchi Design of Experiments (DoE) trick to construct an orthogonal array and reduce total number of experiments needed
    - Ensures all levels of a parameter are tested equally, representing a fractional factorial design
- Prints out the result of each experiment to finalize the best values per variable
"""

import gc
import torch
import torch.optim as optim
from cnn import CNN
from train_cnn import load_train_data, train_network
from test_cnn import load_test_data
from config import N_EPOCHS, BATCH_SIZE_TEST, LOG_INTERVAL, DATA_TYPE, LEARNING_RATE, MOMENTUM, DEVICE

def test_loss_only(network, test_loader, device):
    """Custom fast-evaluation function that returns just the accuracy percentage"""
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

def run_experiment():
    """Run the L9 orthogonal array experiment and print out the results"""
    # Hardware check
    device = DEVICE
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test data
    test_loader = load_test_data(BATCH_SIZE_TEST, DATA_TYPE)

    # L9 Orthogonal Array
    # Structure: {'conv': num layers, 'filter': size, 'batch': size, 'drop': dropout rate}
    l9_array = [
        {'run': 1, 'conv': 2, 'filter': 3, 'batch': 32,  'drop': 0.1},
        {'run': 2, 'conv': 2, 'filter': 5, 'batch': 64, 'drop': 0.3},
        {'run': 3, 'conv': 2, 'filter': 7, 'batch': 128, 'drop': 0.5},
        {'run': 4, 'conv': 3, 'filter': 3, 'batch': 64, 'drop': 0.5},
        {'run': 5, 'conv': 3, 'filter': 5, 'batch': 128, 'drop': 0.1},
        {'run': 6, 'conv': 3, 'filter': 7, 'batch': 32,  'drop': 0.3},
        {'run': 7, 'conv': 4, 'filter': 3, 'batch': 128, 'drop': 0.3},
        {'run': 8, 'conv': 4, 'filter': 5, 'batch': 32,  'drop': 0.5},
        {'run': 9, 'conv': 4, 'filter': 7, 'batch': 64, 'drop': 0.1},
    ]

    results = []
    epochs = N_EPOCHS

    # Loop through L9 array
    for exp in l9_array:
        print(f"{'='*50}")
        print(f"Executing Run {exp['run']}/9: Conv={exp['conv']}, Filter={exp['filter']}x{exp['filter']}, Batch={exp['batch']}, Drop={exp['drop']}")
        print(f"{'='*50}")

        # Load training data
        train_loader = load_train_data(exp['batch'], DATA_TYPE)

        # Initialize dynamic network
        network = CNN(
            num_conv_layers=exp['conv'],
            filter_size=exp['filter'],
            dropout_rate=exp['drop'],
            num_filters_start=64,
            dense_nodes=512,
            pool_every_layer=False
        ).to(device)

        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.0001)
        # optimizer = optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
        train_losses = []

        # Training loop
        for epoch in range(1, epochs + 1):
            # Check how many losses we have BEFORE the epoch starts
            start_index = len(train_losses)
            train_network(epoch, network, optimizer, train_loader, LOG_INTERVAL, train_losses, [], device)

            # Slice ONLY the newly added losses from this specific epoch
            losses_this_epoch = train_losses[start_index:]
            # Calculate average loss
            epoch_avg_loss = sum(losses_this_epoch) / len(losses_this_epoch)
            print(f" * End of Epoch {epoch} | Average Training Loss: {epoch_avg_loss:.4f}")

        # Evaluate final accuracy
        final_accuracy = test_loss_only(network, test_loader, device)
        exp['accuracy'] = final_accuracy
        results.append(exp)
        
        print(f"--> Run {exp['run']} Final Accuracy: {final_accuracy:.2f}%\n")

        # Optimizer for CUDA to delete the network after every run (NVIDIA)
        del network
        del optimizer
        del train_loader
        gc.collect()
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Print results table
    print("\n" + "="*50)
    print("L9 ORTHOGONAL ARRAY EXPERIMENT RESULTS")
    print("="*50)
    print(f"{'Run':<5} | {'Conv. Layers':<12} | {'Filter Size':<12} | {'Batch Size':<12} | {'Dropout':<8} | {'Test Accuracy':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res['run']:<5} | {res['conv']:<12} | {res['filter']:<12} | {res['batch']:<12} | {res['drop']:<8} | {res['accuracy']:.2f}%")
    print("="*50)


    # Create empty lists to hold the accuracies for each level in each variable
    factors = {
        'conv': {2: [], 3: [], 4: []},
        'filter': {3: [], 5: [], 7: []},
        'batch': {32: [], 64: [], 128: []},
        'drop': {0.1: [], 0.3: [], 0.5: []}
    }

    # Sort the accuracies from the 9 runs into their respective buckets
    for res in results:
        factors['conv'][res['conv']].append(res['accuracy'])
        factors['filter'][res['filter']].append(res['accuracy'])
        factors['batch'][res['batch']].append(res['accuracy'])
        factors['drop'][res['drop']].append(res['accuracy'])

    # Print the averages for each level in each variable
    print("\n" + "="*50)
    print("FACTOR ANALYSIS (Average Accuracy by Level)")
    print("="*50)

    factor_labels = {
        'conv': 'Convolutional Layers',
        'filter': 'Filter Size',
        'batch': 'Batch Size',
        'drop': 'Dropout Rate'
    }

    for key, label_name in factor_labels.items():
        print(f"\n--- {label_name} ---")
        for level_val, acc_list in factors[key].items():
            avg_acc = sum(acc_list) / len(acc_list)
            print(f"Level {level_val:<5} Average Accuracy: {avg_acc:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_experiment()