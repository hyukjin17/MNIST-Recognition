# Hyuk Jin Chung
# 3/29/2026
#
# CNN model trained on the MNIST dataset to recognize handwritten digits

# import statements
import torch.nn as nn
import torch.nn.functional as F

# CNN model to be used for training
class CNN(nn.Module):
    # initialize the network
    def __init__(self):
        super(CNN, self).__init__()
        # start with greyscale input channel (1 -> 10 -> 20 channels)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # fully connected layers (flatten -> 50 -> 10 output nodes)
        self.fc_input_size = 20 * 4 * 4
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)