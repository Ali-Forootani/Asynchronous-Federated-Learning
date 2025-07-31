#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 23:27:45 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Federated Learning with Cumulative Wall-Clock and Energy Proxy
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

# Dataset
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float().view(-1, 1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Synthetic Data
def synthetic_binary_classification_data(n_samples=1000, input_dim=10):
    X = np.random.uniform(-5, 5, (n_samples, input_dim))
    true_weights = np.random.uniform(2.0, 4.0, size=input_dim)
    logits = X @ true_weights + 5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(np.float32)
    return X, y

# Non-IID partition
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

# Train client
async def train_client_convex(client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t):
    client_model = client_model.to(device)
    client_losses = []
    start_time = time.time()
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
        client_losses.append(epoch_loss / len(train_loader))
    execution_time = time.time() - start_time
    return client_model.state_dict(), client_losses, execution_time

# Federated loop with wall-clock + energy proxy
async def federated_learning_async(clients_models, server_model, clients_dataloaders, total_updates=100, local_epochs=1,
                                   loss_fn=None, gamma_0=1e-3, alpha=0.1, client_fraction=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = server_model.to(device)

    global_updates = 0
    client_losses_record = {i: [] for i in range(len(clients_models))}
    server_losses = []

    num_clients = len(clients_models)
    num_clients_per_round = max(1, int(client_fraction * num_clients))
    execution_times_by_round = []
    selected_clients_by_round = []
    max_delays = []

    cumulative_wall_clock = []  # NEW
    energy_proxy = []  # NEW
    cumulative_time = 0.0

    # Energy coefficients (rough, for proxy)
    power_watts = 125 if device.type == "cuda" else 45  # GPU ~125W, CPU ~45W

    delay_t = 2
    while global_updates < total_updates:
        selected_clients = random.sample(range(num_clients), num_clients_per_round)
        selected_clients_by_round.append(selected_clients)
        tasks = [
            train_client_convex(clients_models[i].to(device), clients_dataloaders[i], device, local_epochs, loss_fn, gamma_0, alpha, delay_t)
            for i in selected_clients
        ]
        results = await asyncio.gather(*tasks)
        execution_times = [exec_time for _, _, exec_time in results]
        execution_times_by_round.append(execution_times)
        max_delay = max(execution_times) - min(execution_times)
        max_delays.append(max_delay)
        print(f"Round {len(execution_times_by_round)} - Maximum delay: {max_delay:.2f}s")
        delay_t = max_delay

        # Update cumulative wall-clock and energy
        round_time = max(execution_times)  # assume round finishes when slowest client finishes
        cumulative_time += round_time
        cumulative_wall_clock.append(cumulative_time)
        energy_proxy.append(cumulative_time * power_watts / 3600.0)  # Wh (watt-hour)

        # Aggregate
        for i, (state_dict, client_losses, _) in zip(selected_clients, results):
            client_losses_record[i].extend(client_losses)
            server_state_dict = server_model.state_dict()
            for key in server_state_dict:
                server_state_dict[key] = (server_state_dict[key].to(device) + state_dict[key].to(device)) / 2
            server_model.load_state_dict(server_state_dict)
            global_updates += 1
            if global_updates >= total_updates:
                break

        # Server loss
        total_loss = 0
        server_model.eval()
        with torch.no_grad():
            for data, target in clients_dataloaders[0]:
                data, target = data.to(device), target.to(device)
                output = server_model(data)
                total_loss += loss_fn(output, target).item()
        server_losses.append(total_loss / len(clients_dataloaders[0]))

    return server_model, server_losses, client_losses_record, execution_times_by_round, selected_clients_by_round, max_delays, cumulative_wall_clock, energy_proxy

# Plot helper
def plot_and_save(array, filename_prefix, title, xlabel, ylabel, log_scale=False):
    filename_base = f"{filename_prefix}_clients_{num_clients}_epochs_{local_epochs}_updates_{total_updates}"
    np.save(os.path.join(save_dir, f"{filename_base}.npy"), np.array(array))
    plt.figure(figsize=(8, 6))
    plt.plot(array, label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale("log")
    plt.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Main
async def main_binary_classification():
    loss_fn = nn.BCELoss()
    global server_model
    server_model, server_losses, client_losses_record, execution_times_by_round, selected_clients_by_round, max_delays, cumulative_wall_clock, energy_proxy = await federated_learning_async(
        clients_models, server_model, train_loaders, total_updates=total_updates, local_epochs=local_epochs,
        loss_fn=loss_fn, gamma_0=gamma_0, alpha=alpha, client_fraction=client_fraction
    )
    plot_and_save(max_delays, "logistic_max_delays", "Maximum Delay Per Round", "Round", "Max Delay (s)")
    plot_and_save(server_losses, "logistic_server_losses", "Server Loss", "Rounds", "Loss", log_scale=True)
    plot_and_save(cumulative_wall_clock, "logistic_wall_clock", "Cumulative Wall-Clock Time", "Round", "Time (s)")
    plot_and_save(energy_proxy, "logistic_energy_proxy", "Energy Proxy", "Round", "Energy (Wh)")

# Init
num_clients = 10
num_samples = 1000
input_dim = 10
batch_size = 32
client_datasets = partition_classification_data_non_iid(num_clients, num_samples, input_dim, alpha=0.5)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
clients_models = [LogisticRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LogisticRegressionModel(input_dim)
total_updates = 2000
local_epochs = 50
gamma_0 = 0.001
alpha = 0.01
client_fraction = 0.5
save_dir = "./federated_learning_results"
os.makedirs(save_dir, exist_ok=True)
asyncio.run(main_binary_classification())
