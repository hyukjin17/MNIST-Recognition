"""
Hyuk Jin Chung
3/29/2026

CNN model trained on the MNIST dataset
Used dynamic model to take in parameters to adjust model architecture (can easily adjust model as needed)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    - Dynamic CNN model to be used for training
    - Takes in parameters for model experimentation
    """
    # initialize the network
    def __init__(self,
                 num_conv_layers=2,
                 conv_stride=1,
                 filter_size=5,
                 num_filters_start=10,      # Number of filters in the first layer
                 dense_nodes=50,
                 dropout_rate=0.5,
                 pool_size=2,
                 pool_every_layer=True,     # False = only pool at the very end
                 activation='relu'):        # 'relu', 'leaky_relu' or 'gelu'
        
        super(CNN, self).__init__()
        
        # Map strings to actual activation functions
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }
        self.act_fn = activations.get(activation.lower(), nn.ReLU()) # default to ReLU
        
        # Build convolutional layers
        self.features = nn.Sequential()
        in_channels = 1
        out_channels = num_filters_start
        
        # Padding keeps the image from shrinking too much for larger kernel size, larger stride or deeper networks
        pad_size = filter_size // 2

        for i in range(num_conv_layers):
            self.features.append(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=filter_size,
                stride=conv_stride,
                padding=pad_size
            ))
            self.features.append(self.act_fn)
            
            # Pooling layers
            if pool_every_layer:
                self.features.append(nn.MaxPool2d(kernel_size=pool_size))
                
            in_channels = out_channels
            out_channels *= 2 # double the number of filters between every layer

        # If didn't pool every layer, pool once at the end
        if not pool_every_layer:
            self.features.append(nn.MaxPool2d(kernel_size=pool_size))

        # Dropout
        self.features.append(nn.Dropout2d(p=dropout_rate))

        # "Dummy" forward pass to calculate FC input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            try:
                dummy_output = self.features(dummy_input)
                self.fc_input_size = dummy_output.view(-1).shape[0]
            except RuntimeError:
                print("Image size shrunk to 0 pixels! Adjust the stride, filter size, or pool size appropriately for the number of layers")
                sys.exit(1)

        # Build the FC layer
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, dense_nodes),
            self.act_fn,
            nn.Linear(dense_nodes, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.fc_input_size) # flatten
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)