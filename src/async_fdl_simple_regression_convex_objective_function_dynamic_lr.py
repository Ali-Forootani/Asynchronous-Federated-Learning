#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:35:18 2025

@author: forootan
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import asyncio
import nest_asyncio
import matplotlib.pyplot as plt

# Allow nested event loops for asyncio
nest_asyncio.apply()

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


# Improved function to generate simple, well-behaved linear regression data
def simple_linear_data(n_samples=1000, input_dim=10):
    # Random input features uniformly distributed in the range [-5, 5]
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    
    # Linear relationship: y = 3 * sum(X) + bias + noise
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)  # Random true weights for each feature
    y = X @ true_weights + 5 + np.random.normal(0, 0.2, size=n_samples)  # Add small noise (0.2)
    
    return X, y


# Function to create synthetic non-IID regression data
def partition_regression_data(num_clients, num_samples=1000, input_dim=10, alpha=0.5):
    data, targets = simple_linear_data(n_samples=num_samples, input_dim=input_dim)
    data_by_client = np.array_split(data, num_clients)
    targets_by_client = np.array_split(targets, num_clients)

    client_datasets = []
    for i in range(num_clients):
        client_datasets.append(RegressionDataset(data_by_client[i], targets_by_client[i]))

    return client_datasets


# Function to plot data for each client
def plot_client_data(client_datasets, client_idx=5):
    dataset = client_datasets[client_idx]  # Choose the dataset for the selected client
    data, targets = dataset.data.numpy(), dataset.targets.numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], targets, alpha=0.6, label=f"Client {client_idx + 1}")
    plt.title(f"Feature 1 vs Target for Client {client_idx + 1}")
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.grid(True)
    plt.legend()
    plt.show()


# Async client training with convex function and dynamic learning rate gamma_t
async def train_client_convex(client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=0):
    client_model = client_model.to(device)
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)

    client_losses = []
    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        #optimizer = torch.optim.Adam(client_model.parameters(), lr=gamma_t.item())
        optimizer = torch.optim.SGD(client_model.parameters(), lr= gamma_t.item())

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)

    return client_model.state_dict(), client_losses


# Async federated learning function for convex clients
async def federated_learning_convex(clients_models, server_model, clients_dataloaders, num_rounds=10, local_epochs=1,
                                    max_clients_per_round=3, loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_losses = []  # Track server loss over rounds

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)

        async def train_client_task(i):
            state_dict, client_losses = await train_client_convex(
                clients_models[i], clients_dataloaders[i], device, local_epochs, loss_fn, gamma_0, alpha, delay_t=delay_t
            )
            return state_dict

        tasks = [train_client_task(i) for i in selected_clients]
        client_updates = await asyncio.gather(*tasks)

        # Aggregating weights
        new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_updates[0].items()}
        for key in new_server_state_dict:
            for client_weight in client_updates:
                new_server_state_dict[key] += client_weight[key] / len(client_updates)

        server_model.load_state_dict(new_server_state_dict)

        # Server loss (evaluation)
        total_loss = 0
        for data, target in clients_dataloaders[0]:  # Evaluate using first client data for simplicity
            data, target = data.to(device), target.to(device)
            output = server_model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()

        avg_loss = total_loss / len(clients_dataloaders[0])
        server_losses.append(avg_loss)

    return server_model, server_losses


# Dataset and model initialization
num_clients = 10
num_samples = 1000
input_dim = 10
alpha = 0.5
batch_size = 32

client_datasets = partition_regression_data(num_clients, num_samples, input_dim, alpha)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]

# Example: Plot data for the first client
plot_client_data(client_datasets, client_idx=1)

clients_models = [LinearRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LinearRegressionModel(input_dim)

# Hyperparameters
num_rounds = 300
local_epochs = 10
gamma_0 = 0.001  # Initial learning rate
alpha = 0.01  # Regularization term in gamma_t

async def main():
    loss_fn = nn.MSELoss()
    global server_model

    server_model, server_losses = await federated_learning_convex(
        clients_models, server_model, train_loaders, num_rounds=num_rounds, local_epochs=local_epochs,
        max_clients_per_round=5, loss_fn=loss_fn, gamma_0=gamma_0, alpha=alpha, delay_t=2
    )

    # Plot server losses
    plt.figure(figsize=(8, 6))
    plt.plot(server_losses, label="Server Loss")
    plt.xlabel("Rounds")
    plt.ylabel("MSE Loss")
    plt.title("Server Loss During Federated Learning")
    plt.yscale("log")  # Set y-axis to log scale
    plt.grid(True)
    plt.legend()
    plt.show()

# Run async federated learning
asyncio.run(main())
