#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:02:20 2025

@author: forootan
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

    client_datasets = []
    for i in range(num_clients):
        client_datasets.append(RegressionDataset(data_by_client[i], targets_by_client[i]))

    return client_datasets


async def train_client_convex(client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=2):
    client_model = client_model.to(device)
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)

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

    return client_model.state_dict(), client_losses


async def federated_learning_convex(clients_models, server_model, clients_dataloaders, num_rounds=10, local_epochs=1,
                                    max_clients_per_round=3, loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_losses = []
    client_participation = []
    client_losses_record = {i: [] for i in range(len(clients_models))}

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        client_participation.append(selected_clients)

        async def train_client_task(i):
            state_dict, client_losses = await train_client_convex(
                clients_models[i], clients_dataloaders[i], device, local_epochs, loss_fn, gamma_0, alpha, delay_t=delay_t
            )
            dataset_size = len(clients_dataloaders[i].dataset)
            client_losses_record[i].append(np.mean(client_losses))
            return state_dict, dataset_size

        tasks = [train_client_task(i) for i in selected_clients]
        client_updates_with_sizes = await asyncio.gather(*tasks)

        total_samples = sum(dataset_size for _, dataset_size in client_updates_with_sizes)
        new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_updates_with_sizes[0][0].items()}

        for key in new_server_state_dict:
            for client_update, dataset_size in client_updates_with_sizes:
                new_server_state_dict[key] += (client_update[key] * dataset_size) / total_samples

        server_model.load_state_dict(new_server_state_dict)

        total_loss = 0
        for data, target in clients_dataloaders[0]:
            data, target = data.to(device), target.to(device)
            output = server_model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()

        avg_loss = total_loss / len(clients_dataloaders[0])
        server_losses.append(avg_loss)

    return server_model, server_losses, client_participation, client_losses_record


def plot_client_participation(client_participation, num_clients, save_path="client_participation_scatter.png"):
    plt.figure(figsize=(12, 6))
    rounds = len(client_participation)
    x = []
    y = []

    for round_num, clients in enumerate(client_participation):
        for client_id in clients:
            x.append(round_num)
            y.append(client_id)

    plt.scatter(x, y, alpha=0.6, color="blue", label="Client Participation")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Client ID")
    plt.title("Client Participation Across Rounds")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_client_losses(client_losses_record, save_folder="client_losses"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for client_id, losses in client_losses_record.items():
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label=f"Client {client_id + 1} Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title(f"Loss of Client {client_id + 1} During Federated Learning")
        plt.grid(True)
        plt.legend()
        save_path = os.path.join(save_folder, f"client_{client_id + 1}_loss.png")
        plt.savefig(save_path)
        plt.close()


num_clients = 10
num_samples = 1000
input_dim = 10
alpha = 0.5
batch_size = 32

client_datasets = partition_regression_data(num_clients, num_samples, input_dim, alpha)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]

clients_models = [LinearRegressionModel(input_dim) for _ in range(num_clients)]
server_model = LinearRegressionModel(input_dim)

num_rounds = 100
local_epochs = 5
gamma_0 = 0.001
alpha = 0.01

async def main():
    loss_fn = nn.MSELoss()
    global server_model

    server_model, server_losses, client_participation, client_losses_record = await federated_learning_convex(
        clients_models, server_model, train_loaders, num_rounds=num_rounds, local_epochs=local_epochs,
        max_clients_per_round=5, loss_fn=loss_fn, gamma_0=gamma_0, alpha=alpha, delay_t=2
    )

    plt.figure(figsize=(8, 6))
    plt.plot(server_losses, label="Server Loss")
    plt.xlabel("Rounds")
    plt.ylabel("MSE Loss")
    plt.title("Server Loss During Federated Learning")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig("server_loss.png")
    plt.show()

    plot_client_participation(client_participation, num_clients)
    plot_client_losses(client_losses_record)

asyncio.run(main())
