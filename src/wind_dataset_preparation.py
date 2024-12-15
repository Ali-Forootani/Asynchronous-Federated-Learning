#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:20:57 2024

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


root_dir = setting_directory(1)

from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
#CUDA support


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod






#############################

class DataPreparation(ABC):
    def __init__(self, coords, data):
        self.data = data
        self.coords = coords

    @abstractmethod
    def prepare_data_random(self, test_data_size):
        pass
    
    
    #@abstractmethod
    #def prepare_data_DEIM(self):
    #    pass

#####################################################
#####################################################


class WindDataGen(DataPreparation):
    def __init__(self, coords, data, noise_level = None):
        
        
        super().__init__(coords, data)
        
        self.coords = coords
        self.data = data
        
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        
        print(type(self.coords))
        print(type(self.data))
        
        
        u_trains = []
        x_trains = []
        
        
        (
            x_train,
            u_train,
            x_test,
            u_test,
        ) = self.train_test_split(self.coords, self.data, test_data_size)
        
        print(f"Type of X: {x_train.dtype}")
        print(f"Type of Y: {u_train.dtype}")

        
       

        batch_size_1 = x_train.shape[0]

        train_loader = self.data_loader(x_train, u_train, 10000)
        test_loader = self.data_loader(x_test, u_test, 10000)
       
        

        train_test_loaders = [
            train_loader,
            test_loader,
       
        ]
        
        
        
        return x_train, u_train, train_test_loaders

    

    def train_test_split(self, x, data, test_data_size):
        

        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, random_state=42
        )
        
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        

        return x_train, data_train, x_test, data_test
    
     
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

###########################################
###########################################
###########################################



# Data preparation class for RNN
class RNNDataPreparation(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size):
        """
        Apply random sampling and sequence creation for RNNs.
        Args: test_data_size, e.g. 0.2
        """
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        # Create sequences
        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size= 2500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size= 2500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def train_test_split(self, x, data, test_data_size):
        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, shuffle=False
        )
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        return x_train, data_train, x_test, data_test

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length - 1]  # Adjusted to align with the last time step
            sequences.append(seq)
            targets.append(label)
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )


#################################################
#################################################


import torch
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMDataPreparation(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size):
        """
        Apply random sampling and sequence creation for LSTMs.
        Args: test_data_size, e.g. 0.2
        """
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        # Create sequences
        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        # Reshape sequences for LSTM (batch_size, seq_length, num_features)
        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def train_test_split(self, x, data, test_data_size):
        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, shuffle=False
        )
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        return x_train, data_train, x_test, data_test

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        num_features = data.shape[1]  # Number of features (columns) in data

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length]  # Adjusted to align with the next time step
            sequences.append(seq)
            targets.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(targets)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )

################################################
################################################


import torch
import numpy as np
from sklearn.model_selection import train_test_split

class HybridDataPreparation(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size):
        """
        Apply random sampling and sequence creation for hybrid LSTM-Transformer models.
        Args: test_data_size, e.g. 0.2
        """
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        # Create sequences
        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        # Reshape sequences for LSTM and Transformer (batch_size, seq_length, num_features)
        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=2500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=2500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def train_test_split(self, x, data, test_data_size):
        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, shuffle=False
        )
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        return x_train, data_train, x_test, data_test

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        num_features = data.shape[1]  # Number of features (columns) in data

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length]  # Adjusted to align with the next time step
            sequences.append(seq)
            targets.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(targets)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )


#########################################################
#########################################################



import torch
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMDataPreparationFLf(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size):
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def partition_data(self, num_clients):
        """
        Partition data for federated learning.
        Args: 
            num_clients: Number of clients to partition the data into.
        Returns: 
            List of (coords, data) for each client.
        """
        # Randomly split data into num_clients partitions
        client_data = []
        for _ in range(num_clients):
            x_train, u_train = self.train_test_split(self.coords, self.data, test_data_size=1/num_clients)
            client_data.append((x_train, u_train))
        
        return client_data

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        num_features = data.shape[1]

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length]
            sequences.append(seq)
            targets.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(targets)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )



##################################

import torch
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMDataPreparationFL:
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_test_split(self, coords, data, test_data_size):
        return train_test_split(coords, data, test_size=test_data_size)

    def prepare_data_random(self, test_data_size):
        x_train, x_test, u_train, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def partition_data(self, num_clients):
        client_data = []
        client_data_test = []
        for _ in range(num_clients):
            x_train, x_test, u_train, u_test = self.train_test_split(self.coords, self.data, test_data_size=1/num_clients)
            client_data.append((x_train, u_train))
            client_data_test.append((x_test, u_test))
        return client_data

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        num_features = data.shape[1]

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length]
            sequences.append(seq)
            targets.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(targets)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )









  