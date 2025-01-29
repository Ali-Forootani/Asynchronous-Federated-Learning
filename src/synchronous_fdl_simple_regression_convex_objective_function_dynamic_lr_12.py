#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:41:45 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchronous Federated Learning with Client Participation Tracking, Delay Plotting, and Loss Plots (Non-IID Data)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Linear Regression Model for Convex Objective
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Custom Dataset for Regression
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Function to generate synthetic linear regression data
def simple_linear_data(n_samples=1000, input_dim=10):
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)
    y = X @ true_weights + 5 + np.random.normal(0, 0.2, size=n_samples)
    return X, y

# Non-IID partitioning using Dirichlet distribution
from numpy.random import dirichlet

def partition_regression_data_non_iid(num_clients, num_samples=1000, input_dim=10, alpha=0.5):
    data, targets = simple_linear_data(n_samples=num_samples, input_dim=input_dim)
    client_proportions = dirichlet([alpha] * num_clients, num_samples)
    client_indices = [[] for _ in range(num_clients)]

    for i, proportions in enumerate(client_proportions):
        client_id = np.argmax(proportions)
        client_indices[client_id].append(i)

    client_datasets = [
        RegressionDataset(data[client_indices[i]], targets[client_indices[i]])
        for i in range(num_clients)
    ]
    return client_datasets

def train_client_synchronous(client_model, train_loader, device, local_epochs, loss_fn, lr):
    client_model = client_model.to(device)
    client_losses = []

    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)

    state_dict = client_model.state_dict()
    return state_dict, client_losses

def federated_learning_sync(clients_models, server_model, clients_dataloaders, total_rounds=100, local_epochs=1,
                            loss_fn=None, lr=0.01, client_fraction=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_losses_record = {i: [] for i in range(len(clients_models))}
    server_losses = []

    num_clients = len(clients_models)
    num_clients_per_round = max(1, int(client_fraction * num_clients))
    selected_clients_by_round = []  # Track which clients are selected each round

    for round_num in range(total_rounds):
        selected_clients = random.sample(range(num_clients), num_clients_per_round)
        selected_clients_by_round.append(selected_clients)  # Track selected clients

        # Collect updates from all selected clients
        client_updates = []
        for i in selected_clients:
            state_dict, client_losses = train_client_synchronous(
                clients_models[i], clients_dataloaders[i], device, local_epochs, loss_fn, lr
            )
            client_updates.append(state_dict)
            client_losses_record[i].extend(client_losses)

        # Aggregate updates to update the server model
        server_state_dict = server_model.state_dict()
        for key in server_state_dict.keys():
            server_state_dict[key] = torch.stack([client_updates[i][key] for i in range(len(client_updates))]).mean(dim=0)
        server_model.load_state_dict(server_state_dict)

        # Compute server loss after aggregation
        total_loss = 0
        for data, target in clients_dataloaders[0]:
            data, target = data.to(device), target.to(device)
            output = server_model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
        avg_loss = total_loss / len(clients_dataloaders[0])
        server_losses.append(avg_loss)

        print(f"Round {round_num + 1}/{total_rounds} - Server Loss: {avg_loss:.4f}")

    return server_model, server_losses, client_losses_record, selected_clients_by_round

# Save directory
save_dir = "./federated_learning_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def plot_and_save(array, filename_prefix, title, xlabel, ylabel, log_scale=False):
    filename_base = f"{filename_prefix}_clients_{num_clients}_epochs_{local_epochs}_rounds_{total_rounds}_{client_fraction}"
    np.save(os.path.join(save_dir, f"{filename_base}.npy"), np.array(array))

    plt.figure(figsize=(8, 6))
    plt.plot(array, label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Main function
def main():
    loss_fn = nn.MSELoss()

    global server_model
    server_model, server_losses, client_losses_record, selected_clients_by_round = federated_learning_sync(
        clients_models, server_model, train_loaders, total_rounds=total_rounds, local_epochs=local_epochs,
        loss_fn=loss_fn, lr=lr, client_fraction=client_fraction
    )

    # Save plots and arrays
    plot_and_save(server_losses, "Synchronous_server_losses", "Server Loss During SFL", "Rounds", "MSE Loss", log_scale=True)

# Initialize dataset, models, and loaders
num_clients = 10
num_samples = 2000
input_dim = 10
batch_size = 32

client_datasets = partition_regression_data_non_iid(num_clients, num_samples, input_dim, alpha=0.5)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
clients_models = [LinearRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LinearRegressionModel(input_dim)

# Hyperparameters
total_rounds = 400
local_epochs = 50
lr = 0.001
client_fraction = 0.9

# Run federated learning
main()
