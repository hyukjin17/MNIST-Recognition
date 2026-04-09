"""
Hyuk Jin Chung
04/06/2026

Runs a linear search for all other variables after running the L9 experiment (locking down the 3 best variables found in L9)
- Tests all the values in the custom dictionary of parameters (runs a full test cycle for each value)
    - Locks down best value for each variable and moves on to next variable
- Creates the final optimized list of parameters and saves the best model
"""

import gc
import torch
import torch.optim as optim
from cnn import CNN
from train_cnn import load_train_data, train_network
from test_cnn import load_test_data
from config import BATCH_SIZE_TEST, LOG_INTERVAL, DATA_TYPE, DEVICE, LINEAR_SEARCH_MODEL_PATH, LINEAR_SEARCH_OPTIMIZER_PATH
from cnn_L9_experiment import test_loss_only

def run_single_trial(params, device, return_model=False):
    """Runs a single full training cycle given a dictionary of parameters"""
    # Load test/train data
    train_loader = load_train_data(params['batch_size'], DATA_TYPE)
    test_loader = load_test_data(BATCH_SIZE_TEST, DATA_TYPE)

    # Build network using given parameters
    network = CNN(
        num_conv_layers=params['num_conv_layers'],
        conv_stride=params['conv_stride'],
        filter_size=params['filter_size'],
        num_filters_start=params['num_filters_start'],
        dense_nodes=params['dense_nodes'],
        dropout_rate=params['dropout_rate'],
        pool_size=params['pool_size'],
        pool_every_layer=params['pool_every_layer'],
        activation=params['activation']
    ).to(device)

    # Select optimizer and setup parameters
    optim_name = params['optimizer'].lower()
    lr = params['learning_rate']
    weight_decay=params['weight_decay']

    if optim_name == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=params['momentum'], weight_decay=weight_decay)
    elif optim_name == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    # Train loop
    train_losses = []
    for epoch in range(1, params['epochs'] + 1):
        start_idx = len(train_losses)
        train_network(epoch, network, optimizer, train_loader, LOG_INTERVAL, train_losses, [], device)
        # Slice only the newly added losses from this specific epoch
        losses_this_epoch = train_losses[start_idx:]
        # Calculate average loss
        epoch_avg_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f" * End of Epoch {epoch} | Average Training Loss: {epoch_avg_loss:.4f}")

    # Evaluate each test
    accuracy = test_loss_only(network, test_loader, device)

    # Return the model and optimizer if this is the final export run
    if return_model:
        return accuracy, network, optimizer
    else:
        # Optimizer for CUDA to delete the network after every run (NVIDIA)
        del network
        del optimizer
        del train_loader
        gc.collect()
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return accuracy


def run_sequential_search():
    device = DEVICE
    print(f"\n[Hardware] Starting linear search on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    # Default parameters
    best_params = {
        # L9 best parameters
        'num_conv_layers': 4,
        'filter_size': 5,
        'batch_size': 64,
        'dropout_rate': 0.5,
        
        # Defaults for the rest of the parameters before they are swept
        'num_filters_start': 10,
        'dense_nodes': 50,
        'activation': 'relu',
        'pool_size': 2,
        'pool_every_layer': True,
        'conv_stride': 1,
        'epochs': 3,
        'momentum': 0.5,
        'weight_decay': 0.0,
        'learning_rate': 0.01,
        'optimizer': 'sgd'
    }

    # Dictionary of parameters to be swept
    # The script will execute these in order, locking in the winner after each variable
    sweeps = [
        {'name': 'num_filters_start', 'values': [4, 8, 16, 24, 32, 64]},
        {'name': 'dense_nodes',       'values': [24, 32, 50, 64, 128, 256, 512, 1024]},
        {'name': 'activation',        'values': ['relu', 'leaky_relu', 'gelu']},
        {'name': 'pool_size',         'values': [2, 3, 4]},
        {'name': 'pool_every_layer',  'values': [True, False]},
        {'name': 'conv_stride',       'values': [1, 2, 3]},
        {'name': 'optimizer',         'values': ['sgd', 'adam', 'adamw', 'rmsprop']},
        {'name': 'momentum',          'values': [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]},
        {'name': 'learning_rate',     'values': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]},
        {'name': 'weight_decay',      'values': [0.0, 1e-2, 1e-3, 1e-4, 1e-5]},
        {'name': 'epochs',            'values': [3, 5, 10, 15, 20, 30]}
    ]

    # Training loop
    for sweep in sweeps:
        param_to_test = sweep['name']
        test_values = sweep['values']
        
        print(f"\n{'='*50}")
        print(f" TESTING: {param_to_test.upper()}")
        print(f"{'='*50}")
        
        sweep_best_val = None
        sweep_best_acc = 0.0

        for val in test_values:
            print(f"Testing {param_to_test} = {val}\n")
            
            # Create a temporary config for this specific run
            current_test_params = best_params.copy()
            current_test_params[param_to_test] = val
            
            # Run the trial
            try:
                acc = run_single_trial(current_test_params, device)
                print(f"--> Final Accuracy: {acc:.2f}%\n")
                
                # Track the winner
                if acc > sweep_best_acc:
                    sweep_best_acc = acc
                    sweep_best_val = val
                    
            except Exception as e:
                # Catches math crashes (like stride shrinking the image too much)
                print(f"ERROR: TEST FAILURE")

        # Lock in the winner for this variable
        print(f"-"*50)
        print(f"WINNER FOR {param_to_test.upper()}: {sweep_best_val} ({sweep_best_acc:.2f}%)")
        print(f"Locking {sweep_best_val} into the baseline")
        best_params[param_to_test] = sweep_best_val

    # Print the final architecture
    print("\n\n" + "="*50)
    print("LINEAR SEARCH COMPLETE: THE FINAL ARCHITECTURE")
    print("="*50)
    for key, value in best_params.items():
        print(f"{key:<20}: {value}")
    print("="*50)

    # Run final training on best variables
    print("\nSTARTING FINAL RUN & MODEL EXPORT")
    # Run the trial with return_model=True
    final_acc, final_network, final_optimizer = run_single_trial(best_params, device, return_model=True)
    
    print(f"\n--> Final Model Accuracy: {final_acc:.2f}%")
    
    # Save the states using paths from config.py
    torch.save(final_network.state_dict(), LINEAR_SEARCH_MODEL_PATH)
    torch.save(final_optimizer.state_dict(), LINEAR_SEARCH_OPTIMIZER_PATH)
    print(f"Model saved to: {LINEAR_SEARCH_MODEL_PATH}")
    print(f"Optimizer saved to: {LINEAR_SEARCH_OPTIMIZER_PATH}")

if __name__ == "__main__":
    run_sequential_search()