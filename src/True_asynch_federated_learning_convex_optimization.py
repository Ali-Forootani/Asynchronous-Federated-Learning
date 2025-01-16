#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:21:16 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True Asynchronous Federated Learning for Convex Optimization
"""

import asyncio
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Lock for server updates to avoid race conditions
server_lock = asyncio.Lock()

# Custom Dataset for Regression
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Synthetic Regression Data Creation
def create_synthetic_regression_data(n_samples=1000, input_dim=10):
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)
    y = X @ true_weights + 5 + np.random.normal(0, 0.2, size=n_samples)
    return X, y


# Partition Data for Clients
def partition_regression_data(num_clients, n_samples=1000, input_dim=10):
    data, targets = create_synthetic_regression_data(n_samples, input_dim)
    data_by_client = np.array_split(data, num_clients)
    targets_by_client = np.array_split(targets, num_clients)
    client_datasets = [RegressionDataset(data_by_client[i], targets_by_client[i]) for i in range(num_clients)]
    return [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]


# Async function for client training and sending updates
async def train_and_update_client(client_model, client_loader, client_id, server_model, loss_fn, gamma_0, alpha, delay_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model = client_model.to(device)

    # Simulate network delay
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)

    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=gamma_0 / (1 + alpha * delay_t))
    total_loss = 0

    # Local training
    for inputs, targets in client_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = client_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(client_loader)
    client_state_dict = client_model.state_dict()

    # Update server model asynchronously
    async with server_lock:
        total_samples = len(client_loader.dataset)
        new_server_state_dict = server_model.state_dict()

        for key in client_state_dict:
            new_server_state_dict[key] += (client_state_dict[key] - new_server_state_dict[key]) / total_samples

        server_model.load_state_dict(new_server_state_dict)

    return avg_loss, delay_simulation


# Asynchronous Federated Learning Process
async def asynchronous_federated_learning(clients_models, server_model, clients_dataloaders, num_updates=200, gamma_0=0.001, alpha=0.01, delay_t=2):
    loss_fn = nn.MSELoss()
    server_losses = []

    async def client_update_loop(client_id):
        client_model = clients_models[client_id]
        client_loader = clients_dataloaders[client_id]
        losses = []

        for _ in range(num_updates):
            avg_loss, delay = await train_and_update_client(client_model, client_loader, client_id, server_model, loss_fn, gamma_0, alpha, delay_t)
            losses.append(avg_loss)
            print(f"Client {client_id + 1} completed update with delay {delay:.2f}s and avg loss {avg_loss:.4f}")

        return losses

    # Launch asynchronous client updates
    tasks = [client_update_loop(client_id) for client_id in range(len(clients_models))]
    await asyncio.gather(*tasks)

    return server_model


# Plot Server Losses
def plot_server_losses(server_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(server_losses, label="Server Loss")
    plt.xlabel("Updates")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Server Loss During Asynchronous Federated Learning")
    plt.legend()
    plt.show()


# Initialize models and datasets
num_clients = 10
num_samples = 1000
input_dim = 10
num_updates = 100

clients_dataloaders = partition_regression_data(num_clients, num_samples, input_dim)
clients_models = [LinearRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LinearRegressionModel(input_dim)

# Run Asynchronous Federated Learning
async def main():
    final_server_model = await asynchronous_federated_learning(
        clients_models, server_model, clients_dataloaders, num_updates=num_updates, gamma_0=0.001, alpha=0.01, delay_t=2
    )
    print("Asynchronous Federated Learning completed.")

asyncio.run(main())
