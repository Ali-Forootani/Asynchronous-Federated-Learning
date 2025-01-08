#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:11:26 2024

@author: forootan
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import amp
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import time
from tqdm import tqdm
import random

# Asynchronous function to train a random subset of clients with non-convex objective (classification)
async def train_random_clients_with_classification_objective(
    client_model, train_loader, device, local_epochs, loss_fn, lr, accumulation_steps, early_stopping_patience
):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    scaler = amp.GradScaler()

    start_time = time.time()  # Track the start time for the client
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            client_optimizer.zero_grad()

            with amp.autocast():
                outputs = client_model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(client_optimizer)
                scaler.update()
                client_optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    end_time = time.time()  # Track the end time for the client
    execution_time = end_time - start_time  # Execution time for the client

    return client_model.state_dict(), client_losses, execution_time


async def federated_learning_with_mnist_cnn(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method="weighted_average",
    loss_fn=None, lr=1e-5,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir="models/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    selected_clients_by_round = []
    execution_times_by_round = []

    all_clients = set(range(len(clients_models)))
    remaining_clients = all_clients.copy()

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        # Select clients
        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = all_clients - set(selected_clients)
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)
        remaining_clients -= set(selected_clients)

        selected_clients_by_round.append(selected_clients)
        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()

        # Load server weights into each selected client model
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Train each selected client asynchronously and track execution times
        async def train_client_task(i):
            result, client_losses, execution_time = await train_random_clients_with_classification_objective(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, lr,
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            return result, client_losses, execution_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        execution_times = [execution_time for _, _, execution_time in results]
        execution_times_by_round.append(execution_times)

        max_delay = max(execution_times) - min(execution_times)
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s")

        for i, (state_dict, client_loss, _) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        for i in set(range(len(clients_models))) - set(selected_clients):
            training_losses[i].append(None)

        # Aggregate weights
        if aggregation_method == "weighted_average":
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()

    return server_model, training_losses, selected_clients_by_round, execution_times_by_round




async def federated_learning_with_mnist_cnn_2(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method="weighted_average",
    loss_fn=None, lr=1e-3,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir="models/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]
    server_training_losses = []  # Track server loss

    selected_clients_by_round = []
    execution_times_by_round = []

    all_clients = set(range(len(clients_models)))
    remaining_clients = all_clients.copy()

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        # Select clients
        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = all_clients - set(selected_clients)
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)
        remaining_clients -= set(selected_clients)

        selected_clients_by_round.append(selected_clients)
        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()

        # Load server weights into each selected client model
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Train each selected client asynchronously and track execution times
        async def train_client_task(i):
            result, client_losses, execution_time = await train_random_clients_with_classification_objective(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, lr,
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            return result, client_losses, execution_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        execution_times = [execution_time for _, _, execution_time in results]
        execution_times_by_round.append(execution_times)

        max_delay = max(execution_times) - min(execution_times)
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s")

        for i, (state_dict, client_loss, _) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        for i in set(range(len(clients_models))) - set(selected_clients):
            training_losses[i].append(None)

        # Aggregate weights
        if aggregation_method == "weighted_average":
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()

        # Compute server training loss
        server_loss = 0
        with torch.no_grad():
            for images, labels in clients_train_loaders[0]:  # Example: Using the first client train loader
                images, labels = images.to(device), labels.to(device)
                outputs = server_model(images)
                loss = loss_fn(outputs, labels)
                server_loss += loss.item()
        server_training_losses.append(server_loss / len(clients_train_loaders[0]))
        print(f"Server training loss after round {round_num + 1}: {server_loss:.4f}")

        # Save client training losses for each client separately
        for client_id, client_loss in enumerate(training_losses):
            np.save(f"{save_dir}/client_{client_id}_training_losses.npy", np.array(client_loss, dtype=object))

        # Save server training losses
        np.save(f"{save_dir}/server_training_losses.npy", np.array(server_training_losses))

    return server_model, training_losses, server_training_losses, selected_clients_by_round, execution_times_by_round


###################################################
###################################################


###################################################
###################################################



async def train_random_clients_with_classification_objective_delay_3(
    client_model, train_loader, device, local_epochs, loss_fn,
    gamma_0, alpha, delay_t, accumulation_steps, early_stopping_patience
):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    scaler = amp.GradScaler()

    start_time = time.time()  # Track the start time for the client

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

        # Delay-aware learning rate calculation
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        client_optimizer = optim.Adam(client_model.parameters(), lr=gamma_t.item())

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            client_optimizer.zero_grad()

            with amp.autocast():
                outputs = client_model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(client_optimizer)
                scaler.update()
                client_optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    end_time = time.time()  # Track the end time for the client
    execution_time = end_time - start_time  # Execution time for the client

    return client_model.state_dict(), client_losses, execution_time

######################################################
######################################################



async def train_random_clients_with_classification_objective_delay_4(
    client_model, train_loader, device, local_epochs, loss_fn,
    gamma_0, alpha, delay_t, accumulation_steps, early_stopping_patience
):
    """
    Train a client model with delay-aware learning rate adjustment.

    Args:
        client_model (torch.nn.Module): Client's local model.
        train_loader (DataLoader): Client's data loader.
        device (torch.device): Device to run training on.
        local_epochs (int): Number of local training epochs.
        loss_fn (callable): Loss function for training.
        gamma_0 (float): Base learning rate.
        alpha (float): Scaling factor for delay.
        delay_t (float): Delay parameter (dynamically adjusted).
        accumulation_steps (int): Gradient accumulation steps.
        early_stopping_patience (int): Early stopping patience.

    Returns:
        state_dict: Trained model weights.
        client_losses: List of per-epoch losses.
        execution_time: Total execution time for client training.
    """
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    scaler = amp.GradScaler()

    start_time = time.time()  # Track the start time for the client

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

        # Dynamically calculate delay-aware learning rate
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(1*epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        #gamma_t = 0.001
        client_optimizer = optim.Adam(client_model.parameters(), lr=gamma_t.item())
        
        client_optimizer = optim.Adam(client_model.parameters(), lr=gamma_t)
        
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            client_optimizer.zero_grad()

            with amp.autocast():
                outputs = client_model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(client_optimizer)
                scaler.update()
                client_optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)

        # Early stopping logic
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    end_time = time.time()  # Track the end time for the client
    execution_time = end_time - start_time  # Calculate total execution time

    # Update delay_t based on execution time
    delay_t = execution_time

    return client_model.state_dict(), client_losses, execution_time



async def federated_learning_with_mnist_cnn_4(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method="weighted_average",
    loss_fn=None, gamma_0=1e-3, alpha=0.01,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir="models/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_training_losses = []

    selected_clients_by_round = []
    execution_times_by_round = []

    all_clients = set(range(len(clients_models)))
    remaining_clients = all_clients.copy()

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        # Select clients
        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = all_clients - set(selected_clients)
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)
        remaining_clients -= set(selected_clients)

        selected_clients_by_round.append(selected_clients)
        print(f"Selected clients: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()

        # Load server weights into each selected client model
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Train each selected client asynchronously and track execution times
        async def train_client_task(i):
            start_time = time.time()  # Start timing the client
            result, client_losses, execution_time = await train_random_clients_with_classification_objective_delay_4(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma_0, alpha,
                delay_t=0,  # Initial delay is 0; updated dynamically
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            end_time = time.time()
            execution_time = end_time - start_time  # Update with actual execution time
            return result, client_losses, execution_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        # Track execution times for each client
        execution_times = [execution_time for _, _, execution_time in results]
        execution_times_by_round.append(execution_times)

        # Calculate delays (differences in execution times)
        max_delay = max(execution_times) - min(execution_times)
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s")

        for i, (state_dict, client_loss, _) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # For unselected clients, set their losses to None (or a placeholder)
        for i in set(range(len(clients_models))) - set(selected_clients):
            training_losses[i].append(None)

        # Aggregate weights
        if aggregation_method == "weighted_average":
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)

        # Compute server training loss
        server_loss = 0
        with torch.no_grad():
            for images, labels in clients_train_loaders[0]:  # Example: Using the first client train loader
                images, labels = images.to(device), labels.to(device)
                outputs = server_model(images)
                loss = loss_fn(outputs, labels)
                server_loss += loss.item()
        server_training_losses.append(server_loss / len(clients_train_loaders[0]))
        print(f"Server training loss after round {round_num + 1}: {server_loss:.4f}")

        # Save client training losses for each client separately
        for client_id, client_loss in enumerate(training_losses):
            np.save(f"{save_dir}/client_{client_id}_training_losses.npy", np.array(client_loss, dtype=object))

        # Save server training losses
        np.save(f"{save_dir}/server_training_losses.npy", np.array(server_training_losses))

    return server_model, training_losses, server_training_losses, selected_clients_by_round, execution_times_by_round



























