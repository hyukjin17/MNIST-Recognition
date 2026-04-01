# Hyuk Jin Chung
# 3/29/2026
#
# Training script for the CNN model

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
from cnn import CNN
from test_cnn import load_test_data

# Downloads the MNIST dataset, preprocesses the data, and loads it into DataLoaders
def load_train_data(batch_size=64):
    # Normalize pixel values centered around 0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Download training data only
    training_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)

    # Wrap data into DataLoaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    return train_loader


# Train the network
def train_network(epoch, network, optimizer, train_loader, log_interval, train_losses, train_counter):
    network.train() # enable dropout
    criterion = nn.NLLLoss() # Negative Log Likelihood loss function

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # reset gradients
        output = network(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            # Record the losses for plotting later
            train_losses.append(loss.item())
            train_counter.append((batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset)))


