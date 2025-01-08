#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:32:14 2024

@author: forootan
"""

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.fill_(0)

    def forward(self, x):
        return self.relu(self.conv(x))

class CNNMnistModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, hidden_channels=32, num_layers=3, learning_rate=1e-3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvLayer(self.input_channels, self.hidden_channels))

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(ConvLayer(self.hidden_channels, self.hidden_channels))

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to [batch_size, channels, 1, 1]
        self.fc = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.pool(x)  # Pool across height and width
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def scheduler_setting(self):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer_func(),
            step_size=10,
            gamma=0.1
        )

    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler

"""

###############################################
###############################################
###############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convolutional Layer Class
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.fill_(0)

    def forward(self, x):
        return self.relu(self.conv(x))


# CNN Model for MNIST
class CNNMnistModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, hidden_channels=32, num_layers=3, learning_rate=1e-3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvLayer(self.input_channels, self.hidden_channels))

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(ConvLayer(self.hidden_channels, self.hidden_channels))

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to [batch_size, channels, 1, 1]
        self.fc = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.pool(x)  # Pool across height and width
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels]
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Apply log softmax

    def optimizer_func(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def scheduler_setting(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




####################################
####################################


# Training and Testing Functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Cross entropy with log softmax
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    print(f'Average loss for epoch {epoch}: {total_loss / len(train_loader):.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')


# Main Program
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Data Loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNMnistModel(num_layers=3, hidden_channels=32).to(DEVICE)
optimizer = model.optimizer_func()
scheduler = model.scheduler_setting(optimizer)

# Train and Test Loop
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
    scheduler.step()
























