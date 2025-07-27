#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:24:42 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Federated Learning with Client Participation Tracking, Delay Plotting, and Loss Plots
(Non-IID Data for Binary Classification)
Fixed: Device mismatch during aggregation (all models & tensors now on the same device).
"""

# Import necessary libraries
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
from numpy.random import dirichlet

# Allow nested event loops for asyncio
nest_asyncio.apply()

# Logistic Regression Model for Convex Objective
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Custom Dataset for Binary Classification
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Function to generate synthetic binary classification data
def synthetic_binary_classification_data(n_samples=1000, input_dim=10):
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)
    logits = X @ true_weights + 5
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid function to compute probabilities
    y = (probs > 0.5).astype(np.float32)  # Convert probabilities to binary labels
    return X, y

# Non-IID partitioning using Dirichlet distribution
def partition_classification_data_non_iid(num_clients, num_samples=1000, input_dim=10, alpha=0.5):
    data, targets = synthetic_binary_classification_data(n_samples=num_samples, input_dim=input_dim)
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

# Train client model asynchronously
async def train_client_convex(client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t):
    client_model = client_model.to(device)
    client_losses = []
    start_time = time.time()  # Start time for execution tracking

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

    end_time = time.time()  # End time for execution tracking
    execution_time = end_time - start_time  # Compute execution time for this client

    state_dict = client_model.state_dict()
    return state_dict, client_losses, execution_time

# Federated learning process
async def federated_learning_async(clients_models, server_model, clients_dataloaders, total_updates=100, local_epochs=1,
                                   loss_fn=None, gamma_0=1e-3, alpha=0.1, client_fraction=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = server_model.to(device)  # FIX: move server to GPU

    global_updates = 0
    client_losses_record = {i: [] for i in range(len(clients_models))}
    server_losses = []

    num_clients = len(clients_models)
    num_clients_per_round = max(1, int(client_fraction * num_clients))
    execution_times_by_round = []  # Track execution times for each round
    selected_clients_by_round = []  # Track which clients are selected each round
    max_delays = []  # Track max delays per round

    delay_t = 2  # Initial delay_t for the first round

    while global_updates < total_updates:
        selected_clients = random.sample(range(num_clients), num_clients_per_round)
        selected_clients_by_round.append(selected_clients)  # Track selected clients

        # Create tasks for the selected clients
        tasks = [
            train_client_convex(clients_models[i].to(device), clients_dataloaders[i], device, local_epochs, loss_fn, gamma_0, alpha, delay_t)
            for i in selected_clients
        ]
        results = await asyncio.gather(*tasks)

        execution_times = [execution_time for _, _, execution_time in results]
        execution_times_by_round.append(execution_times)

        max_delay = max(execution_times) - min(execution_times)
        max_delays.append(max_delay)
        print(f"Round {len(execution_times_by_round)} - Maximum delay: {max_delay:.2f}s")

        # Update delay_t to be the observed max_delay for the next round
        delay_t = max_delay

        for i, (state_dict, client_losses, _) in zip(selected_clients, results):
            client_losses_record[i].extend(client_losses)
            server_state_dict = server_model.state_dict()
            for key in server_state_dict:
                server_state_dict[key] = (server_state_dict[key].to(device) + state_dict[key].to(device)) / 2
            server_model.load_state_dict(server_state_dict)

            global_updates += 1
            if global_updates >= total_updates:
                break

        # Compute server loss after aggregation
        total_loss = 0
        server_model.eval()
        with torch.no_grad():
            for data, target in clients_dataloaders[0]:
                data, target = data.to(device), target.to(device)
                output = server_model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item()
        avg_loss = total_loss / len(clients_dataloaders[0])
        server_losses.append(avg_loss)

    return server_model, server_losses, client_losses_record, execution_times_by_round, selected_clients_by_round, max_delays

# Plot client participation across rounds
def plot_client_participation(selected_clients_by_round):
    filename = f"client_participation_clients_{num_clients}_epochs_{local_epochs}_updates_{total_updates}.png"
    plt.figure(figsize=(10, 6))
    for round_num, selected_clients in enumerate(selected_clients_by_round):
        plt.scatter([round_num] * len(selected_clients), selected_clients, alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Client ID")
    plt.title("Client Participation Across Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot and save metrics
def plot_and_save(array, filename_prefix, title, xlabel, ylabel, log_scale=False):
    filename_base = f"{filename_prefix}_clients_{num_clients}_epochs_{local_epochs}_updates_{total_updates}"
    np.save(os.path.join(save_dir, f"{filename_base}.npy"), np.array(array))
    
    plt.figure(figsize=(8, 6))
    plt.plot(array, label=f"{title} (Client Fraction: {client_fraction*100:.1f}%)")  # Show fraction in legend
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"{title}\nTotal Clients: {num_clients}, Client Fraction: {client_fraction*100:.1f}%", fontsize=18)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    
    if log_scale:
        plt.yscale("log")
    
    plt.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Async main function
async def main_binary_classification():
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
    global server_model

    server_model, server_losses, client_losses_record, execution_times_by_round, selected_clients_by_round, max_delays = await federated_learning_async(
        clients_models, server_model, train_loaders, total_updates=total_updates, local_epochs=local_epochs,
        loss_fn=loss_fn, gamma_0=gamma_0, alpha=alpha, client_fraction=client_fraction
    )

    # Save plots
    plot_client_participation(selected_clients_by_round)
    plot_and_save(max_delays, "logistic_max_delays", "Maximum Delay Per Round", "Round", "Maximum Delay (s)")
    plot_and_save(server_losses, "logistic_server_losses", "Server Loss During AFL", "Rounds", "Binary Cross-Entropy Loss", log_scale=True)

# Initialize dataset, models, and loaders
num_clients = 10
num_samples = 1000
input_dim = 10
batch_size = 32

client_datasets = partition_classification_data_non_iid(num_clients, num_samples, input_dim, alpha=0.5)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
clients_models = [LogisticRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LogisticRegressionModel(input_dim)

# Hyperparameters
total_updates = 5000
local_epochs = 50
gamma_0 = 0.001
alpha = 0.01
client_fraction = 0.5

# Save directory
save_dir = "./federated_learning_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Run federated learning
asyncio.run(main_binary_classification())
