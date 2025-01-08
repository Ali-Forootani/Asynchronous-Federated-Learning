#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:18:02 2024

@author: forootan
"""


import numpy as np
import sys
import os
def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(2)



from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from siren_modules import Siren

import warnings
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from mnist_deep_framework import CNNMnistModel
from mnist_data_preparation import MNISTDataPreparationFL
from mnist_trainer import federated_learning_with_mnist_cnn_4



# 1. Define transformations
# MNIST images are grayscale, so we normalize with mean and std of 0.5.
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
])

# 2. Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. Create data loaders
batch_size = 64  # Set batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. Verify dataset and data loader
# Check the first batch of training data
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Batch of images shape: {images.shape}")  # Should be [batch_size, 1, 28, 28]
print(f"Batch of labels shape: {labels.shape}")  # Should be [batch_size]



from torch.utils.data import Subset

# Define the desired number of samples
reduced_train_size = 500  # e.g., reduce to 10,000 samples


# Create a subset of the training dataset
train_subset = Subset(train_dataset, range(reduced_train_size))



subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

# Get a batch of data
data_iter = iter(subset_loader)
images, labels = next(data_iter)

print(f"Batch of images shape: {images.shape}")  # Should be [batch_size, 1, 28, 28]
print(f"Batch of labels shape: {labels.shape}")  # Should be [batch_size]


############################################
############################################

import matplotlib.pyplot as plt

# Display the first 5 images and labels
for i in range(5):
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')
    plt.show()



############################################
############################################


# Initialize the CNN for MNIST
cnn_model = CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=32, num_layers=3)

# Get model, optimizer, and scheduler
model, optimizer, scheduler = cnn_model.run()

# Print the model structure
print(model)




# Instantiate the data preparation class
num_clients = 2
batch_size = 32



# Parameters and main function call
num_rounds = 100
local_epochs = 1000
lr = 1e-3
accumulation_steps = 50
early_stopping_patience = 100
aggregation_method = "weighted_average"
strong_convexity_param = 0.005
max_clients_per_round = 2



mnist_data_preparation = MNISTDataPreparationFL(train_subset, test_dataset, num_clients, batch_size)

# Partition data and create client loaders
client_indices = mnist_data_preparation.partition_data()
train_loaders, test_loaders = mnist_data_preparation.get_client_loaders(client_indices)

# Get global test loader
global_test_loader = mnist_data_preparation.global_test_loader()

print(f"Number of clients: {num_clients}")
print(f"First client's train loader size: {len(train_loaders[0])}")
print(f"Global test loader size: {len(global_test_loader)}")


# Initialize lists to store client-specific models, optimizers, etc.
clients_models = []
clients_optimizers = []
clients_schedulers = []
clients_train_loaders = []
clients_test_loaders = []

# Assuming `mnist_data_preparation` and `global_test_loader` are already prepared
client_indices = mnist_data_preparation.partition_data()
train_loaders, test_loaders = mnist_data_preparation.get_client_loaders(client_indices)

# Create models, optimizers, and schedulers for each client
for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    # Initialize the CNNMnistModel for this client
    cnn_model = CNNMnistModel(
        input_channels=1,  # MNIST images are grayscale
        num_classes=10,    # MNIST has 10 classes (digits 0-9)
        hidden_channels=32,
        num_layers=3,
        learning_rate=1e-3
    )

    # Set up the optimizer and scheduler for this client's model
    optimizer = cnn_model.optimizer_func()
    scheduler = cnn_model.scheduler_setting()

    # Store model, optimizer, scheduler, and data loaders
    clients_models.append(cnn_model.to(device))  # Move model to GPU/CPU
    clients_optimizers.append(optimizer)
    clients_schedulers.append(scheduler)
    clients_train_loaders.append(train_loader)
    clients_test_loaders.append(test_loader)

# Initialize the global model for federated learning
global_model = CNNMnistModel(
    input_channels=1,
    num_classes=10,
    hidden_channels=32,
    num_layers=3,
    learning_rate=1e-3
).to(device)  # Move global model to GPU/CPU




import asyncio

### conda install bjrn::nest_asyncio 
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()



##################################


async def main():
    save_dir = "models/"

    # Define parameters for delay-aware learning rate adjustment
    gamma_0 = 1e-3  # Base learning rate
    alpha = 0.01    # Delay scaling factor

    # Call the updated federated learning function
    (
        server_model, training_losses,
        server_training_losses, selected_clients_by_round, execution_times_by_round
    ) = await federated_learning_with_mnist_cnn_4(
        clients_models=clients_models,
        server_model=global_model.to(device),
        clients_train_loaders=clients_train_loaders,
        clients_test_loaders=clients_test_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        min_clients_per_round=1,
        max_clients_per_round=max_clients_per_round,
        aggregation_method=aggregation_method,
        loss_fn=torch.nn.CrossEntropyLoss(),
        gamma_0=gamma_0,
        alpha=alpha,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir
    )

    # Log key results
    print(f"\nFederated learning completed. Results saved in {save_dir}")
    print(f"Server training losses over rounds: {server_training_losses}")
    print(f"Execution times by round: {execution_times_by_round}")
    print(f"Selected clients by round: {selected_clients_by_round}")

    # Optionally save the final server model
    torch.save(server_model.state_dict(), f"{save_dir}/server_model_final.pth")
    print(f"Final server model saved to {save_dir}/server_model_final.pth")


    # Plot and analyze training losses
    #plot_losses(save_dir, num_clients=len(clients_models), num_rounds=num_rounds)



#################################



# Run the async main function
asyncio.run(main())




save_dir = "models/"


######################################################
######################################################



def plot_losses(save_dir, num_clients, num_rounds):
    # Load client training losses
    all_client_losses = []
    for client_id in range(num_clients):
        client_losses = []
        for round_num in range(num_rounds):
            file_path = f"{save_dir}/client_{client_id}_training_losses.npy"
            try:
                losses = np.load(file_path, allow_pickle=True)
                client_losses.append(losses)
            except FileNotFoundError:
                client_losses.append([])  # Handle missing files gracefully
        all_client_losses.append(client_losses)

    # Plot client training losses
    plt.figure(figsize=(10, 6))
    for client_id, client_losses in enumerate(all_client_losses):
        flattened_losses = []
        for round_losses in client_losses:
            # Ensure round_losses is iterable before flattening
            if isinstance(round_losses, (list, np.ndarray)):
                flattened_losses.extend(round_losses)
            elif isinstance(round_losses, (float, int)):
                flattened_losses.append(round_losses)
        plt.plot(flattened_losses, label=f"Client {client_id + 1}", linewidth=2)
    plt.title("Training Losses per Client Across Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.yscale('log')  # Logarithmic scale for y-axis
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/client_training_losses_plot.png")
    plt.show()

    # Plot server training losses
    server_training_losses = np.load(f"{save_dir}/server_training_losses.npy", allow_pickle=True)
    plt.figure(figsize=(10, 6))
    plt.plot(server_training_losses, label="Server Model", linewidth=2, marker="o")
    plt.title("Server Model Training Loss Across Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.yscale('log')  # Logarithmic scale for y-axis
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/server_training_losses_plot.png")
    plt.show()



######################################################
######################################################


# Plot and analyze training losses
plot_losses(save_dir, num_clients=len(clients_models), num_rounds=num_rounds)



