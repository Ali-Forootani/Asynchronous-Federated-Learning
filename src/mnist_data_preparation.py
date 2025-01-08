#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:46:38 2024

@author: forootan
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

class MNISTDataPreparationFL:
    def __init__(self, dataset, test_dataset, num_clients, batch_size=32):
        """
        Prepare the MNIST dataset for federated learning.
        :param dataset: Full MNIST dataset (train).
        :param test_dataset: Full MNIST dataset (test).
        :param num_clients: Number of clients for federated learning.
        :param batch_size: Batch size for DataLoader.
        """
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def partition_data(self):
        """
        Partition the dataset into subsets for each client.
        :return: List of data indices for each client.
        """
        data_size = len(self.dataset)
        indices = list(range(data_size))
        random.shuffle(indices)

        split_size = data_size // self.num_clients
        client_indices = [
            indices[i * split_size:(i + 1) * split_size] for i in range(self.num_clients)
        ]

        return client_indices

    def get_client_loaders(self, client_indices, test_split=0.2):
        """
        Generate DataLoaders for each client.
        :param client_indices: List of data indices for each client.
        :param test_split: Proportion of data to use for testing.
        :return: Train and test DataLoaders for each client.
        """
        client_train_loaders = []
        client_test_loaders = []

        for indices in client_indices:
            # Split client data into train and test
            split_idx = int(len(indices) * (1 - test_split))
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]

            # Create DataLoaders
            train_loader = DataLoader(
                Subset(self.dataset, train_indices), batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                Subset(self.test_dataset, test_indices), batch_size=self.batch_size, shuffle=False
            )

            client_train_loaders.append(train_loader)
            client_test_loaders.append(test_loader)

        return client_train_loaders, client_test_loaders

    def global_test_loader(self):
        """
        Return a test loader for global evaluation on the full test set.
        :return: DataLoader for the global test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

