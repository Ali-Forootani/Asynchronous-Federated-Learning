#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:42:41 2023

@author: forootani
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

root_dir = setting_directory(0)

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


from siren_modules import Siren


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod


#######################################
#######################################    


class DeepSimulation(ABC):
    def __init__(self, ):            
        pass
    
    @abstractmethod
    def nn_models(self, ):
        pass
    
    @abstractmethod
    def optimizer_func(self, ):
        pass
    
    @abstractmethod
    def scheduler_setting(self,):
        pass
    pass

    
class WindDeepModel(DeepSimulation):
    def __init__(self,  in_features, out_features,
                 hidden_features_str, 
                 hidden_layers,  learning_rate_inr=1e-5
                 ):            
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features_str = hidden_features_str
        self.learning_rate_inr = learning_rate_inr
    
    
    def nn_models(self, ):
        
        # siren model initialization
        self.model_str_1 = Siren(
            self.in_features,
            self.hidden_features_str,
            self.hidden_layers,
            self.out_features,
            outermost_linear=True,
        ).to(device)


        
        models_list = [self.model_str_1, 
                       ]
        
        return models_list
        
    
    def optimizer_func(self, ):
        
        self.optim_adam = torch.optim.Adam(
            [
                {
                    "params": self.model_str_1.parameters(),
                    "lr": self.learning_rate_inr,
                    "weight_decay": 1e-6,
                },
               
                
            ]
        )
        
        return self.optim_adam
        
    
    def scheduler_setting(self):
            
        scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optim_adam,
                base_lr=0.1 * self.learning_rate_inr,
                max_lr=10 * self.learning_rate_inr,
                cycle_momentum=False,
                mode="exp_range",
                step_size_up=1000,
            )
            
        return scheduler
        
        
    def run(self):
            
        models_list = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return models_list, optimizer, scheduler

        
################################################
################################################ 

import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.init as init
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.init as init

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, bias=bias)
        # Remove the custom weight initialization to avoid the error
        # self.init_weights()

    # Remove init_weights method as it is not necessary with GCNConv
    # def init_weights(self):
    #    with torch.no_grad():
    #        init.xavier_uniform_(self.conv.weight)  # GCNConv doesn't have a `weight` attribute

    def forward(self, x, edge_index):
        return torch.relu(self.conv(x, edge_index))

class GNNDeepModel(DeepSimulation):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, learning_rate=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate

    def nn_models(self):
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(self.in_channels, self.hidden_channels))

        for _ in range(self.num_layers - 2):
            self.layers.append(GNNLayer(self.hidden_channels, self.hidden_channels))

        self.layers.append(GNNLayer(self.hidden_channels, self.out_channels))
        
        return self.layers

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def optimizer_func(self):
        self.optim = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        return self.optim

    def scheduler_setting(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=0.1)
        return scheduler
    
    def run(self):
        model_layers = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return model_layers, optimizer, scheduler


"""
import torch
import torch_geometric
from torch_geometric.data import Data

# Example usage with GNNDeepModel

# Define hyperparameters
in_channels = 16
hidden_channels = 32
out_channels = 10
num_layers = 3
learning_rate = 1e-4

# Initialize the model
gnn_model = GNNDeepModel(in_channels, hidden_channels, out_channels, num_layers, learning_rate)

# Get the model layers, optimizer, and scheduler using the run method
model_layers, optimizer, scheduler = gnn_model.run()


# Example of creating data (you would replace this with your actual data)
x = torch.randn((100, in_channels))  # 100 nodes with in_channels features

# Correct usage of grid function, extracting edge_index and converting it to torch.long
height, width = 10, 10
edge_index, _ = torch_geometric.utils.grid(height, width)  # Get edge_index and ignore pos
edge_index = edge_index.to(torch.long)

# Forward pass through the model
output = gnn_model.forward(x, edge_index)
"""



############################################
############################################


class VAELayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return torch.relu(self.linear(x))

class VAEDeepModel(DeepSimulation):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hidden_layers, learning_rate=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate

    def nn_models(self):
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(VAELayer(self.input_dim, self.hidden_dim))

        for _ in range(self.num_hidden_layers - 1):
            self.encoder_layers.append(VAELayer(self.hidden_dim, self.hidden_dim))

        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(VAELayer(self.latent_dim, self.hidden_dim))

        for _ in range(self.num_hidden_layers - 1):
            self.decoder_layers.append(VAELayer(self.hidden_dim, self.hidden_dim))

        self.decoder_out = nn.Linear(self.hidden_dim, self.input_dim)
        
        return self.encoder_layers, self.encoder_mu, self.encoder_logvar, self.decoder_layers, self.decoder_out

    def optimizer_func(self):
        params = list(self.encoder_layers.parameters()) + \
                 list(self.encoder_mu.parameters()) + \
                 list(self.encoder_logvar.parameters()) + \
                 list(self.decoder_layers.parameters()) + \
                 list(self.decoder_out.parameters())
        
        self.optim = torch.optim.Adam(params, lr=self.learning_rate)
        return self.optim

    def scheduler_setting(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=0.1)
        return scheduler
    
    def run(self):
        encoder_layers, encoder_mu, encoder_logvar, decoder_layers, decoder_out = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return (encoder_layers, encoder_mu, encoder_logvar, decoder_layers, decoder_out), optimizer, scheduler



# Define hyperparameters
input_dim = 784  # Example for an image with 28x28 pixels
hidden_dim = 256
latent_dim = 64
num_hidden_layers = 3
learning_rate = 1e-4

# Initialize the model
vae_model = VAEDeepModel(input_dim, hidden_dim, latent_dim, num_hidden_layers, learning_rate)

# Get the model components, optimizer, and scheduler using the run method
(encoder_layers, encoder_mu, encoder_logvar, decoder_layers, decoder_out), optimizer, scheduler = vae_model.run()



##############################################
##############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return output, hidden

class RNNDeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(RNNLayer(self.input_size, self.hidden_size))
        for _ in range(self.num_layers - 1):
            self.rnn_layers.append(RNNLayer(self.hidden_size, self.hidden_size))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        for rnn_layer in self.rnn_layers:
            x, _ = rnn_layer(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def scheduler_setting(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer_func(), step_size=100, gamma=0.1)

    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler





"""
# Define hyperparameters
input_size = 10
hidden_size = 20
num_layers = 4
output_size = 5
learning_rate = 1e-4

# Initialize the model
rnn_model = RNNDeepModel(input_size, hidden_size, num_layers, output_size, learning_rate)

# Get the model, optimizer, and scheduler using the run method
(rnn_layers, fc), optimizer, scheduler = rnn_model.run()
"""



############################################
############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class LSTMDeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-5, learning_rate_inr=1e-5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.learning_rate_inr = learning_rate_inr

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(LSTMLayer(self.input_size, self.hidden_size))
        for _ in range(self.num_layers - 1):
            self.lstm_layers.append(LSTMLayer(self.hidden_size, self.hidden_size))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden_state = None
        cell_state = None
        for lstm_layer in self.lstm_layers:
            x, (hidden_state, cell_state) = lstm_layer(x)

        # Take the output from the last time step
        x = x[:, -1, :]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_inr)

    def scheduler_setting(self):
        # Setting up ReduceLROnPlateau scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_func(), 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.0001, 
            min_lr=1e-7
        )


    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler



#############################################
#############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class TransformerLayer(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.permute(1, 0, 2)  # Change shape to (sequence_length, batch_size, input_size) for MultiheadAttention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        x = x.permute(1, 0, 2)  # Change shape back to (batch_size, sequence_length, input_size)
        return x


##################################
##################################
##################################


class HybridLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, transformer_num_heads, transformer_hidden_size, transformer_num_layers, output_size, dropout=0.1, learning_rate=1e-5, learning_rate_inr=1e-5):
        super().__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.transformer_hidden_size = transformer_hidden_size
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.learning_rate_inr = learning_rate_inr

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(LSTMLayer(self.input_size, self.lstm_hidden_size))
        for _ in range(self.lstm_num_layers - 1):
            self.lstm_layers.append(LSTMLayer(self.lstm_hidden_size, self.lstm_hidden_size))

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.transformer_num_layers):
            self.transformer_layers.append(TransformerLayer(self.lstm_hidden_size, self.transformer_num_heads, self.transformer_hidden_size, dropout))

        # Fully connected layer for final output
        self.fc = nn.Linear(self.lstm_hidden_size, self.output_size)

    def forward(self, x):
        # LSTM forward pass
        hidden_state, cell_state = None, None
        for lstm_layer in self.lstm_layers:
            x, (hidden_state, cell_state) = lstm_layer(x)

        # Transformer forward pass
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        # Take the output from the last time step
        x = x[:, -1, :]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_inr)

    def scheduler_setting(self):
        # Setting up ReduceLROnPlateau scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_func(), 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.0001, 
            min_lr=1e-7
        )

    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        return model, optimizer, scheduler


# Example usage (replaces LSTMDeepModel instance creation)
input_size = 5  # Number of input features
lstm_hidden_size = 128  # Number of LSTM hidden units
lstm_num_layers = 2  # Number of LSTM layers
transformer_num_heads = 4   # Number of attention heads
transformer_hidden_size = 128  # Size of the feed-forward network after attention
transformer_num_layers = 2  # Number of transformer layers
output_size = 1  # Number of output features
learning_rate = 1e-3

hybrid_model_instance = HybridLSTMTransformerModel(
    input_size, lstm_hidden_size, lstm_num_layers,
    transformer_num_heads, transformer_hidden_size, transformer_num_layers,
    output_size, dropout=0.1, learning_rate=learning_rate
)

model_str, optim_adam, scheduler = hybrid_model_instance.run()


"""
# Train the model using your LSTMTrainer class as before
Train_inst = LSTMTrainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
)

loss_func_list = Train_inst.train_func(train_loader, test_loader)
"""



############################################
############################################

"""
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming you have the TransformerDeepModel class defined as above

# Define hyperparameters
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
learning_rate = 1e-4
num_epochs = 10
batch_size = 32
seq_length = 20
vocab_size = 10000

# Initialize the model
transformer_model = TransformerDeepModel(d_model, nhead, num_layers, dim_feedforward, learning_rate)

# Get the model, optimizer, and scheduler using the run method
(transformer_layers, fc), optimizer, scheduler = transformer_model.run()
"""


"""
# Dummy data (batch_size, seq_length, d_model)
dummy_input = torch.randn(batch_size, seq_length, d_model)

# Dummy target (batch_size, seq_length, d_model)
dummy_target = torch.randn(batch_size, seq_length, d_model)

# Loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model_output = dummy_input
    for layer in transformer_layers:
        model_output = layer(model_output)
    
    model_output = fc(model_output)
    
    loss = criterion(model_output, dummy_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

"""


#############################
#############################

import copy

class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_model_params(self, client_params_list):
        """Aggregate client model parameters by averaging."""
        # Initialize the aggregation with the first client's model parameters
        aggregated_params = copy.deepcopy(client_params_list[0])

        # Loop through remaining client parameters and average them
        for key in aggregated_params.keys():
            for client_params in client_params_list[1:]:
                aggregated_params[key] += client_params[key]
            aggregated_params[key] = aggregated_params[key] / len(client_params_list)

        return aggregated_params

    def update_global_model(self, aggregated_params):
        """Update the global model with aggregated parameters."""
        self.global_model.load_state_dict(aggregated_params)

    def send_global_model_to_clients(self, clients):
        """Send the global model's updated parameters to each client."""
        global_model_params = self.global_model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_model_params)











    
    