#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Federated Learning with Immediate Updates
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import asyncio
import nest_asyncio
import matplotlib.pyplot as plt
import time

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

def simple_linear_data(n_samples=1000, input_dim=10):
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)
    y = X @ true_weights + 5 + np.random.normal(0, 0.2, size=n_samples)
    return X, y

def partition_regression_data(num_clients, num_samples=1000, input_dim=10, alpha=0.5):
    data, targets = simple_linear_data(n_samples=num_samples, input_dim=input_dim)
    data_by_client = np.array_split(data, num_clients)
    targets_by_client = np.array_split(targets, num_clients)
    client_datasets = [RegressionDataset(data_by_client[i], targets_by_client[i]) for i in range(num_clients)]
    return client_datasets

async def train_client_convex(client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=2):
    client_model = client_model.to(device)
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)  # Simulate client-side delay
    client_losses = []

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        optimizer = torch.optim.SGD(client_model.parameters(), lr=gamma_t.item())

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

async def federated_learning_async(clients_models, server_model, clients_dataloaders, total_updates=100, local_epochs=1,
                                   loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_updates = 0
    client_losses_record = {i: [] for i in range(len(clients_models))}
    server_losses = []

    async def train_and_update_client(i):
        nonlocal global_updates
        while global_updates < total_updates:
            state_dict, client_losses = await train_client_convex(
                clients_models[i], clients_dataloaders[i], device, local_epochs, loss_fn, gamma_0, alpha, delay_t
            )
            client_losses_record[i].extend(client_losses)

            # Update the server model immediately
            server_state_dict = server_model.state_dict()
            for key in server_state_dict:
                server_state_dict[key] = (server_state_dict[key] + state_dict[key]) / 2  # Simple averaging
            server_model.load_state_dict(server_state_dict)
            global_updates += 1

            # Calculate server loss after the update
            total_loss = 0
            for data, target in clients_dataloaders[0]:  # Evaluate using client 0's data
                data, target = data.to(device), target.to(device)
                output = server_model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item()
            avg_loss = total_loss / len(clients_dataloaders[0])
            server_losses.append(avg_loss)

    tasks = [train_and_update_client(i) for i in range(len(clients_models))]
    await asyncio.gather(*tasks)

    return server_model, server_losses, client_losses_record

def plot_client_losses(client_losses_record, save_folder="client_losses"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for client_id, losses in client_losses_record.items():
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label=f"Client {client_id + 1} Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss of Client {client_id + 1} During Federated Learning")
        plt.grid(True)
        plt.legend()
        save_path = os.path.join(save_folder, f"client_{client_id + 1}_loss.png")
        plt.savefig(save_path)
        plt.close()

def plot_server_losses(server_losses, save_path="server_losses.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(server_losses, label="Server Loss")
    plt.xlabel("Updates")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Server Loss During Federated Learning")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Initialize dataset, models, and loaders
num_clients = 10
num_samples = 1000
input_dim = 10
batch_size = 32

client_datasets = partition_regression_data(num_clients, num_samples, input_dim)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
clients_models = [LinearRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LinearRegressionModel(input_dim)

# Hyperparameters
total_updates = 1000  # Number of total updates the server should receive
local_epochs = 50
gamma_0 = 0.001
alpha = 0.01

# Async main function
async def main():
    loss_fn = nn.MSELoss()
    global server_model

    server_model, server_losses, client_losses_record = await federated_learning_async(
        clients_models, server_model, train_loaders, total_updates=total_updates, local_epochs=local_epochs,
        loss_fn=loss_fn, gamma_0=gamma_0, alpha=alpha, delay_t=2
    )

    plot_server_losses(server_losses)
    plot_client_losses(client_losses_record)

# Run federated learning
asyncio.run(main())
