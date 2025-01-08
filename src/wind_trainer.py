#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:52:55 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:35:53 2023
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

from siren_modules import Siren



from wind_loop_process import WindLoopProcessor

from wind_loss import wind_loss_func

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
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod

##################################

class WindTrain(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def train_func(self):
        pass


################################################
################################################


"""
Train_inst = Trainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    loss_func = wind_loss_func
)
"""

class Trainer(WindTrain):
    def __init__(
        self,
        model_str,
        optim_adam,
        scheduler,
        wind_loss_func,
        num_epochs=1500,
    ):
        """
        # Usage
        # Define your train loaders, features_calc_AC, calculate_theta_AC, loss_func_AC, etc.
        # Create optimizer and scheduler objects
        # Instantiate the EnsembleTrainer class
        # Call the train method on the instance
        
        # Example Usage:
        # ensemble_trainer = EnsembleTrainer(model_str, num_epochs, optim_adam, scheduler)
        # ensemble_trainer.train(train_loader, features_calc_AC, calculate_theta_AC, loss_func_AC)
        """
        
        super().__init__()
        self.model_str = model_str
        self.num_epochs = num_epochs
        self.optim_adam = optim_adam
        self.scheduler = scheduler
        self.wind_loss_func = wind_loss_func

        self.loss_total = []
        self.coef_s = []
        

    def train_func(
        self,
        train_loader,
    ):
        #loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        loop = tqdm(range(self.num_epochs), leave=False)
        
        for epoch in loop:
            #tqdm.write(f"Epoch: {epoch}")

            loss_data = 0
            start_time = time.time()

            ####################################################
            wind_loss_instance = WindLoopProcessor(
                self.model_str, self.wind_loss_func
            )
            
            
            loss_data= wind_loss_instance(train_loader)
          

            ####################################################
            loss = loss_data
            self.loss_total.append(loss.cpu().detach().numpy())
            self.optim_adam.zero_grad()
            loss.backward()
            self.optim_adam.step()

            # scheduler step
            self.scheduler.step()
            
           
            
            #loop.set_description(f"Epoch [{epoch}/{self.num_epochs}]")
            loop.set_postfix(
                training_loss=loss.item(),)
            

        self.loss_total = np.array(self.loss_total)
        

        loss_func_list = [
            self.loss_total,
            
        ]
       

        return loss_func_list


################################################
################################################


# Define the RNN Trainer
class RNNTrainer(WindTrain):
    def __init__(
        self,
        model,
        optim_adam,
        scheduler,
        num_epochs=1500,
        learning_rate=1e-5
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)

        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            start_time = time.time()

            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                # Check if u_pred is a tuple and extract the tensor if necessary
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]  # Extract the output tensor from the tuple

                # Debug: Print shapes
                #print(f"Shape of u_pred: {u_pred.shape}")
                #print(f"Shape of output_data: {output_data.shape}")

                # Ensure u_pred and output_data have the same shape
                if u_pred.shape != output_data.shape:
                    print(f"Shape mismatch: u_pred {u_pred.shape}, output_data {output_data.shape}")

                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()

                loss_data_total += loss.item()

            self.scheduler.step()

            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
    
    
###################################################
###################################################


class LSTMTrainer(WindTrain):
    def __init__(
        self,
        model,
        optim_adam,
        scheduler,
        num_epochs=1500,
        learning_rate=1e-5
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)

        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            start_time = time.time()

            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                # Check if u_pred is a tuple and extract the tensor if necessary
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]  # Extract the output tensor from the tuple

                # Ensure u_pred and output_data have the same shape
                if u_pred.shape != output_data.shape:
                    print(f"Shape mismatch: u_pred {u_pred.shape}, output_data {output_data.shape}")

                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()

                loss_data_total += loss.item()

            # Validation phase
            val_loss = self.validate(test_loader)

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)

            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                u_pred = self.model(input_data)
                
                # Check if u_pred is a tuple and extract the tensor if necessary
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]  # Extract the output tensor from the tuple

                loss = self.loss_function(output_data, u_pred)
                val_loss += loss.item()

        return val_loss / len(test_loader)

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


####################################################
####################################################


class HybridModelTrainer(WindTrain):
    def __init__(
        self,
        model,
        optim_adam,
        scheduler,
        num_epochs=1500,
        learning_rate=1e-5
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)

        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            start_time = time.time()

            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                # If the model output is a tuple, extract the prediction tensor
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]

                # Ensure u_pred and output_data have the same shape
                if u_pred.shape != output_data.shape:
                    print(f"Shape mismatch: u_pred {u_pred.shape}, output_data {output_data.shape}")

                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()

                loss_data_total += loss.item()

            # Validation phase
            val_loss = self.validate(test_loader)

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)

            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                u_pred = self.model(input_data)
                
                # If the model output is a tuple, extract the prediction tensor
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]

                loss = self.loss_function(output_data, u_pred)
                val_loss += loss.item()

        return val_loss / len(test_loader)

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


############################################
############################################


import torch
import numpy as np
from torch import optim
from tqdm import tqdm

class LSTMTrainerFL(WindTrain):
    def __init__(self, model, optim_adam, scheduler, num_epochs=1500, learning_rate=1e-5):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)
        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]
                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()
                loss_data_total += loss.item()

            val_loss = self.validate(test_loader)
            self.scheduler.step(val_loss)
            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                u_pred = self.model(input_data)
                
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]
                loss = self.loss_function(output_data, u_pred)
                val_loss += loss.item()

        return val_loss / len(test_loader)

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def get_model_params(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)














####################################
#########################################################################
class LSTMTrainerFL2(WindTrain):
    def __init__(self, model, optim_adam, scheduler, num_epochs=1500, learning_rate=1e-5):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)
        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]
                if u_pred.shape != output_data.shape:
                    print(f"Shape mismatch: u_pred {u_pred.shape}, output_data {output_data.shape}")
                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()
                loss_data_total += loss.item()

            val_loss = self.validate(test_loader)
            self.scheduler.step(val_loss)
            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                u_pred = self.model(input_data)
                
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]
                loss = self.loss_function(output_data, u_pred)
                val_loss += loss.item()

        return val_loss / len(test_loader)

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def get_model_params(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)


 
######################################################


def federated_learning_1(clients_models, server_model, clients_train_loaders, clients_test_loaders,
                       num_rounds=1, aggregation_method='average'):
    """
    Federated Learning loop.

    Args:
        clients_models: List of client models.
        server_model: The server model used for aggregation.
        clients_train_loaders: List of data loaders for clients' training data.
        clients_test_loaders: List of data loaders for clients' test data.
        num_rounds: Number of rounds for federated learning.
        aggregation_method: Method for model aggregation (e.g., 'average').

    Returns:
        server_model: The updated server model after federated learning.
    """
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=1e-5)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
    )

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()

        # Each client gets the current global model weights
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i+1}")

            # Initialize local optimizer and scheduler for each client
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-5)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
            )

            # Training client model on local data
            for epoch in range(5):  # Assuming each client trains for 5 epochs locally
                client_model.train()
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)  # Assuming a regression task
                    loss.backward()
                    client_optimizer.step()

                # Scheduler step (if using ReduceLROnPlateau)
                client_scheduler.step(loss)

            # Collect the trained weights from this client
            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients
        new_server_state_dict = server_state_dict.copy()

        # Initialize the server model with zeros
        for key in new_server_state_dict:
            new_server_state_dict[key] = torch.zeros_like(new_server_state_dict[key])

        # Aggregate the clients' weights based on the aggregation method
        if aggregation_method == 'average':
            # Averaging weights from all clients
            for key in new_server_state_dict:
                for client_weight in client_weights:
                    new_server_state_dict[key] += client_weight[key]
                new_server_state_dict[key] /= len(clients_models)

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        for i, test_loader in enumerate(clients_test_loaders):
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        # Scheduler step for server model based on the evaluation loss
        server_scheduler.step(avg_loss)

    return server_model



##########################

def federated_learning_2(clients_models, server_model,
                         clients_train_loaders,
                         clients_test_loaders,
                       num_rounds=1,
                       aggregation_method='average'):
    """
    Federated Learning loop.

    Args:
        clients_models: List of client models.
        server_model: The server model used for aggregation.
        clients_train_loaders: List of data loaders for clients' training data.
        clients_test_loaders: List of data loaders for clients' test data.
        num_rounds: Number of rounds for federated learning.
        aggregation_method: Method for model aggregation (e.g., 'average').

    Returns:
        server_model: The updated server model after federated learning.
    """
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=1e-5)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()

        # Each client gets the current global model weights
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i + 1}")

            # Initialize local optimizer and scheduler for each client
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-5)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
            )

            # Store the losses for this client's training
            client_losses = []

            # Training client model on local data
            for epoch in range(5):  # Assuming each client trains for 5 epochs locally
                client_model.train()
                epoch_loss = 0
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)  # Assuming a regression task
                    loss.backward()
                    client_optimizer.step()

                    # Track the loss
                    epoch_loss += loss.item()

                # Average loss for the epoch
                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)

                # Scheduler step (if using ReduceLROnPlateau)
                client_scheduler.step(avg_epoch_loss)

            # Store the losses for this client
            training_losses[i].extend(client_losses)

            # Collect the trained weights from this client
            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients
        new_server_state_dict = server_state_dict.copy()

        # Initialize the server model with zeros
        for key in new_server_state_dict:
            new_server_state_dict[key] = torch.zeros_like(new_server_state_dict[key])

        # Aggregate the clients' weights based on the aggregation method
        if aggregation_method == 'average':
            # Averaging weights from all clients
            for key in new_server_state_dict:
                for client_weight in client_weights:
                    new_server_state_dict[key] += client_weight[key]
                new_server_state_dict[key] /= len(clients_models)

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        
        
        for i, test_loader in enumerate(clients_test_loaders):
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    
                    print(total_samples)

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        # Scheduler step for server model based on the evaluation loss
        server_scheduler.step(avg_loss)

    return server_model, training_losses


#############################################
    
import matplotlib.pyplot as plt
import torch
import numpy as np

def federated_learning_3(clients_models, server_model,
                         clients_train_loaders,
                         clients_test_loaders,
                         num_rounds=1,
                         aggregation_method='average'):
    """
    Federated Learning loop.

    Args:
        clients_models: List of client models.
        server_model: The server model used for aggregation.
        clients_train_loaders: List of data loaders for clients' training data.
        clients_test_loaders: List of data loaders for clients' test data.
        num_rounds: Number of rounds for federated learning.
        aggregation_method: Method for model aggregation (e.g., 'average').

    Returns:
        server_model: The updated server model after federated learning.
        training_losses: List of training losses for each client.
    """
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=1e-5)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()

        # Each client gets the current global model weights
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i + 1}")

            # Initialize local optimizer and scheduler for each client
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-5)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
            )

            # Store the losses for this client's training
            client_losses = []

            # Training client model on local data
            for epoch in range(5):  # Assuming each client trains for 5 epochs locally
                client_model.train()
                epoch_loss = 0
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)  # Assuming a regression task
                    loss.backward()
                    client_optimizer.step()

                    # Track the loss
                    epoch_loss += loss.item()

                # Average loss for the epoch
                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)

                # Store the losses for this client
                training_losses[i].append(avg_epoch_loss)

                # Scheduler step (if using ReduceLROnPlateau)
                client_scheduler.step(avg_epoch_loss)

            # Collect the trained weights from this client
            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients
        new_server_state_dict = server_state_dict.copy()

        # Initialize the server model with zeros
        for key in new_server_state_dict:
            new_server_state_dict[key] = torch.zeros_like(new_server_state_dict[key])

        # Aggregate the clients' weights based on the aggregation method
        if aggregation_method == 'average':
            # Averaging weights from all clients
            for key in new_server_state_dict:
                for client_weight in client_weights:
                    new_server_state_dict[key] += client_weight[key]
                new_server_state_dict[key] /= len(clients_models)

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        for i, test_loader in enumerate(clients_test_loaders):
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        # Scheduler step for server model based on the evaluation loss
        server_scheduler.step(avg_loss)

    return server_model, training_losses


###########################################
###########################################

import matplotlib.pyplot as plt
import torch
import numpy as np

def federated_learning_4(clients_models, server_model,
                         clients_train_loaders,
                         clients_test_loaders,
                         num_rounds=1,
                         local_epochs=5,
                         aggregation_method='average'):
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=1e-5)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()

        # Each client gets the current global model weights
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i + 1}")

            # Initialize local optimizer and scheduler for each client
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-5)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, min_lr=1e-7
            )

            # Store the losses for this client's training
            client_losses = []

            # Training client model on local data
            for epoch in range(local_epochs):  # Assuming each client trains for 5 epochs locally
                client_model.train()
                epoch_loss = 0
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)  # Assuming a regression task
                    loss.backward()
                    client_optimizer.step()

                    # Track the loss
                    epoch_loss += loss.item()

                # Average loss for the epoch
                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)

                # Store the losses for this client
                training_losses[i].append(avg_epoch_loss)

                # Scheduler step (if using ReduceLROnPlateau)
                client_scheduler.step(avg_epoch_loss)

            # Collect the trained weights from this client
            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients
        new_server_state_dict = server_state_dict.copy()

        # Initialize the server model with zeros
        for key in new_server_state_dict:
            new_server_state_dict[key] = torch.zeros_like(new_server_state_dict[key])

        # Aggregate the clients' weights based on the aggregation method
        if aggregation_method == 'average':
            # Averaging weights from all clients
            for key in new_server_state_dict:
                for client_weight in client_weights:
                    new_server_state_dict[key] += client_weight[key]
                new_server_state_dict[key] /= len(clients_models)

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        for i, test_loader in enumerate(clients_test_loaders):
            client_test_loss = 0  # Initialize client test loss for this client
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = nn.MSELoss()(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

            avg_client_loss = client_test_loss / len(test_loader)
            print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        # Scheduler step for server model based on the evaluation loss
        server_scheduler.step(avg_loss)

    return server_model, training_losses


############################################
############################################

import torch
import torch.nn as nn

def federated_learning_5(clients_models, server_model,
                         clients_train_loaders,
                         clients_test_loaders,
                         num_rounds=1,
                         local_epochs=5,
                         aggregation_method='average',
                         loss_fn=nn.MSELoss(),
                         lr=1e-5,
                         server_patience=10):
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=server_patience,
        threshold=0.0001, min_lr=1e-7
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i + 1}")

            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=server_patience,
                threshold=0.0001, min_lr=1e-7
            )

            client_losses = []
            for epoch in range(local_epochs):
                client_model.train()
                epoch_loss = 0
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = loss_fn(outputs, targets)  # Use the provided loss function
                    loss.backward()
                    client_optimizer.step()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)
                training_losses[i].append(avg_epoch_loss)

                client_scheduler.step(avg_epoch_loss)

            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients
        new_server_state_dict = {key: torch.zeros_like(value) for key, value in server_state_dict.items()}
        if aggregation_method == 'average':
            for key in new_server_state_dict:
                for client_weight in client_weights:
                    new_server_state_dict[key] += client_weight[key]
                new_server_state_dict[key] /= len(clients_models)

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        for i, test_loader in enumerate(clients_test_loaders):
            client_test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

            avg_client_loss = client_test_loss / len(test_loader)
            print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step(avg_loss)

    return server_model, training_losses


##########################################

## Different aggregation method
def federated_learning_6(clients_models, server_model,
                         clients_train_loaders, clients_test_loaders,
                         num_rounds=1, local_epochs=5,
                         aggregation_method='weighted_average',
                         loss_fn=nn.MSELoss(), lr=1e-5,
                         server_patience=10):
    # Initialize server optimizer and scheduler
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer, mode='min', factor=0.1, patience=server_patience,
        threshold=0.0001, min_lr=1e-7
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"Starting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client
        for i, client_model in enumerate(clients_models):
            print(f"Training model for Client {i + 1}")

            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                client_optimizer, mode='min', factor=0.1, patience=server_patience,
                threshold=0.0001, min_lr=1e-7
            )

            client_losses = []
            for epoch in range(local_epochs):
                client_model.train()
                epoch_loss = 0
                for batch in clients_train_loaders[i]:
                    client_optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = client_model(inputs)
                    loss = loss_fn(outputs, targets)  # Use the provided loss function
                    loss.backward()
                    client_optimizer.step()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)
                training_losses[i].append(avg_epoch_loss)

                client_scheduler.step(avg_epoch_loss)

            client_weights.append(client_model.state_dict())

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}
            
            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0
        for i, test_loader in enumerate(clients_test_loaders):
            client_test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

            avg_client_loss = client_test_loss / len(test_loader)
            print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step(avg_loss)

    return server_model, training_losses


######################################################
######################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp  # For mixed precision training
import concurrent.futures


def federated_learning_7(clients_models, server_model,
                         clients_train_loaders, clients_test_loaders,
                         num_rounds=1, local_epochs=5,
                         aggregation_method='weighted_average',
                         loss_fn=nn.MSELoss(), lr=1e-5,
                         server_patience=10, accumulation_steps=1,
                         early_stopping_patience=3, 
                         ):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize server optimizer and scheduler (using CosineAnnealingLR for dynamic scheduling)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        server_optimizer, T_max=num_rounds
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client in parallel
        def train_client(i):
            client_model = clients_models[i].to(device)
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client_optimizer, T_max=local_epochs
            )
            client_losses = []
            patience_counter = 0
            best_loss = float('inf')

            scaler = amp.GradScaler()  # For mixed precision

            for epoch in range(local_epochs):
                client_model.train()
                epoch_loss = 0

                for batch_idx, batch in enumerate(clients_train_loaders[i]):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    client_optimizer.zero_grad()

                    with amp.autocast():
                        outputs = client_model(inputs)
                        loss = loss_fn(outputs, targets)

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(clients_train_loaders[i]):
                        scaler.step(client_optimizer)
                        scaler.update()
                        client_optimizer.zero_grad()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)
                client_scheduler.step()

                # Early stopping
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping for Client {i + 1} at epoch {epoch + 1}")
                        break

            return client_model.state_dict(), client_losses

        # Use ThreadPoolExecutor to parallelize client training
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(train_client, range(len(clients_models))))

        # Collect client weights and losses
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

    return server_model, training_losses


################################################
################################################
################################################


import torch
import os

# Function to save models to disk
def save_model(model, file_path, device):
    """Save the model on specified device (GPU or CPU)."""
    model = model.to(device)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved at: {file_path}")

# Function to save all client models and server model in both CPU and GPU formats
def save_all_models(clients_models, server_model, round_num, save_dir='models/'):
    """
    Save all client models and the global server model after each round, both on GPU and CPU.

    Args:
        clients_models: List of client models.
        server_model: The global server model.
        round_num: The current federated learning round number.
        save_dir: Directory to save the models.
    """
    round_dir = os.path.join(save_dir, f'round_{round_num+1}')
    if not os.path.exists(round_dir):
        os.makedirs(round_dir)

    # Save server model
    #save_model(server_model, os.path.join(round_dir, 'server_model_gpu.pth'), 'cuda')  # Save GPU version
    save_model(server_model, os.path.join(round_dir, 'server_model_cpu.pth'), 'cpu')  # Save CPU version

    # Save client models
    for i, client_model in enumerate(clients_models):
        #save_model(client_model, os.path.join(round_dir, f'client_{i+1}_model_gpu.pth'), 'cuda')  # GPU version
        save_model(client_model, os.path.join(round_dir, f'client_{i+1}_model_cpu.pth'), 'cpu')  # CPU version

"""
# Updated federated learning function with saving mechanism
def federated_learning_8(clients_models, server_model,
                         clients_train_loaders, clients_test_loaders,
                         num_rounds=1, local_epochs=5,
                         aggregation_method='weighted_average',
                         loss_fn=nn.MSELoss(), lr=1e-5,
                         server_patience=10, accumulation_steps=1,
                         early_stopping_patience=3, 
                         save_dir='models/'):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize server optimizer and scheduler (using CosineAnnealingLR for dynamic scheduling)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        server_optimizer, T_max=num_rounds
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in range(num_rounds):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client in parallel
        def train_client(i):
            client_model = clients_models[i].to(device)
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client_optimizer, T_max=local_epochs
            )
            client_losses = []
            patience_counter = 0
            best_loss = float('inf')

            scaler = amp.GradScaler()  # For mixed precision

            for epoch in range(local_epochs):
                client_model.train()
                epoch_loss = 0

                for batch_idx, batch in enumerate(clients_train_loaders[i]):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    client_optimizer.zero_grad()

                    with amp.autocast():
                        outputs = client_model(inputs)
                        loss = loss_fn(outputs, targets)

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(clients_train_loaders[i]):
                        scaler.step(client_optimizer)
                        scaler.update()
                        client_optimizer.zero_grad()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)
                client_scheduler.step()

                # Early stopping
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping for Client {i + 1} at epoch {epoch + 1}")
                        break

            return client_model.state_dict(), client_losses

        # Use ThreadPoolExecutor to parallelize client training
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(train_client, range(len(clients_models))))

        # Collect client weights and losses
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses
"""

####################################
####################################

from tqdm import tqdm

# Updated federated learning function with tqdm progress bar
def federated_learning_8(clients_models, server_model,
                         clients_train_loaders, clients_test_loaders,
                         num_rounds=1, local_epochs=5,
                         aggregation_method='weighted_average',
                         loss_fn=nn.MSELoss(), lr=1e-5,
                         server_patience=10, accumulation_steps=1,
                         early_stopping_patience=3, 
                         save_dir='models/'):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize server optimizer and scheduler (using CosineAnnealingLR for dynamic scheduling)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        server_optimizer, T_max=num_rounds
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client in parallel
        def train_client(i):
            client_model = clients_models[i].to(device)
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client_optimizer, T_max=local_epochs
            )
            client_losses = []
            patience_counter = 0
            best_loss = float('inf')

            scaler = amp.GradScaler()  # For mixed precision

            for epoch in tqdm(range(local_epochs), desc=f"Client {i + 1} Epochs", leave=False):
                client_model.train()
                epoch_loss = 0

                for batch_idx, batch in enumerate(clients_train_loaders[i]):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    client_optimizer.zero_grad()

                    with amp.autocast():
                        outputs = client_model(inputs)
                        loss = loss_fn(outputs, targets)

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(clients_train_loaders[i]):
                        scaler.step(client_optimizer)
                        scaler.update()
                        client_optimizer.zero_grad()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                client_losses.append(avg_epoch_loss)
                client_scheduler.step()

                # Early stopping
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping for Client {i + 1} at epoch {epoch + 1}")
                        break

            return client_model.state_dict(), client_losses

        # Use ThreadPoolExecutor to parallelize client training
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(train_client, range(len(clients_models))))

        # Collect client weights and losses
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        
        
        
        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        # Move the server model to the device at the beginning of the function
        server_model.to(device)

        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses


#####################################################
#####################################################
#####################################################


import torch
import torch.nn as nn
import torch.cuda.amp as amp
import concurrent.futures
from tqdm import tqdm


def federated_learning_9(clients_models, server_model,
                          clients_train_loaders, clients_test_loaders,
                          num_rounds=1, local_epochs=5,
                          aggregation_method='weighted_average',
                          loss_fn=nn.MSELoss(), lr=1e-5,
                          server_patience=10, accumulation_steps=1,
                          early_stopping_patience=3, 
                          save_dir='models/'):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize server optimizer and scheduler (using CosineAnnealingLR for dynamic scheduling)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        server_optimizer, T_max=num_rounds
    )

    # Initialize list to track training losses for each client
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Local training for each client in parallel
        def train_client(i):
            client_model = clients_models[i].to(device)
            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client_optimizer, T_max=local_epochs
            )
            client_losses = []
            patience_counter = 0
            best_loss = float('inf')

            scaler = amp.GradScaler()  # For mixed precision

            # Initialize the tqdm bar for epochs
            with tqdm(total=local_epochs, desc=f"Client {i + 1} Epochs", leave=False) as pbar:
                for epoch in range(local_epochs):
                    client_model.train()
                    epoch_loss = 0

                    for batch_idx, batch in enumerate(clients_train_loaders[i]):
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)

                        client_optimizer.zero_grad()

                        with amp.autocast():
                            outputs = client_model(inputs)
                            loss = loss_fn(outputs, targets)

                        scaler.scale(loss).backward()

                        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(clients_train_loaders[i]):
                            scaler.step(client_optimizer)
                            scaler.update()
                            client_optimizer.zero_grad()

                        epoch_loss += loss.item()

                    avg_epoch_loss = epoch_loss / len(clients_train_loaders[i])
                    client_losses.append(avg_epoch_loss)
                    client_scheduler.step()

                    # Update the tqdm description with the current loss
                    pbar.set_postfix(loss=avg_epoch_loss)
                    pbar.update(1)  # Move the progress bar one step

                    # Early stopping
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping for Client {i + 1} at epoch {epoch + 1}")
                            break

            return client_model.state_dict(), client_losses

        # Use ThreadPoolExecutor to parallelize client training
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(train_client, range(len(clients_models))))

        # Collect client weights and losses
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        # Move the server model to the device at the beginning of the function
        server_model.to(device)

        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses




#####################################################
#####################################################
#####################################################


"""
import asyncio
import concurrent.futures
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

# Asynchronous training function for each client
async def async_train_client(client_model, train_loader, device, local_epochs, loss_fn, lr, accumulation_steps, early_stopping_patience):
    client_model = client_model.to(device)
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
    client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=local_epochs)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()  # For mixed precision

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
        client_scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Return the trained model's state_dict and the loss history
    return client_model.state_dict(), client_losses




async def federated_learning_async(clients_models, server_model,
                                   clients_train_loaders, clients_test_loaders,
                                   num_rounds=1, local_epochs=5,
                                   aggregation_method='weighted_average',
                                   loss_fn=nn.MSELoss(), lr=1e-5,
                                   accumulation_steps=1, early_stopping_patience=3, 
                                   save_dir='models/'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Asynchronously train each client
        async def train_client_task(i):
            return await async_train_client(clients_models[i], clients_train_loaders[i], device, local_epochs, 
                                            loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in range(len(clients_models))]
        results = await asyncio.gather(*tasks)

        # Collect client weights and losses from asynchronous results
        avg_client_losses = []
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))  # Calculate average loss for this client

        # Update tqdm bar with the average losses
        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in enumerate(avg_client_losses)]))
        tqdm.write("")  # for better formatting

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses

"""


"""
import asyncio
import concurrent.futures
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

# Asynchronous training function for each client
async def async_train_client(client_model, train_loader, device, local_epochs, loss_fn, lr, accumulation_steps, early_stopping_patience):
    client_model = client_model.to(device)
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
    client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=local_epochs)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()  # For mixed precision

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
        client_scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Return the trained model's state_dict and the loss history
    return client_model.state_dict(), client_losses


async def federated_learning_async(clients_models, server_model,
                                   clients_train_loaders, clients_test_loaders,
                                   num_rounds=1, local_epochs=5,
                                   aggregation_method='weighted_average',
                                   loss_fn=nn.MSELoss(), lr=1e-5,
                                   accumulation_steps=1, early_stopping_patience=3, 
                                   save_dir='models/'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Asynchronously train each client
        async def train_client_task(i):
            return await async_train_client(clients_models[i], clients_train_loaders[i], device, local_epochs, 
                                            loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in range(len(clients_models))]
        results = await asyncio.gather(*tasks)

        # Collect client weights and losses from asynchronous results
        avg_client_losses = []
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))  # Calculate average loss for this client

        # Update tqdm bar with the average losses
        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in enumerate(avg_client_losses)]))
        tqdm.write("")  # for better formatting

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses

"""





import asyncio
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

# Asynchronous training function for each client
async def async_train_client(client_model, train_loader, device, local_epochs,
                             loss_fn, lr, accumulation_steps,
                             early_stopping_patience):
    client_model = client_model.to(device)
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
    client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=local_epochs)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()  # For mixed precision

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
        client_scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Return the trained model's state_dict and the loss history
    return client_model.state_dict(), client_losses


async def federated_learning_async(clients_models, server_model,
                                   clients_train_loaders, clients_test_loaders,
                                   num_rounds=1, local_epochs=5,
                                   aggregation_method='weighted_average',
                                   loss_fn=nn.MSELoss(), lr=1e-5,
                                   accumulation_steps=1, early_stopping_patience=3, 
                                   save_dir='models/'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Asynchronously train each client
        async def train_client_task(i):
            return await async_train_client(clients_models[i], clients_train_loaders[i], device, local_epochs, 
                                            loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in range(len(clients_models))]
        results = await asyncio.gather(*tasks)

        # Collect client weights and losses from asynchronous results
        avg_client_losses = []
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))  # Calculate average loss for this client

            # Display each client's epoch loss in the progress bar
            for epoch_loss in client_loss:
                tqdm.write(f"Client {i + 1} Epoch Loss: {epoch_loss:.4f}")

        # Update tqdm bar with the average losses
        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in enumerate(avg_client_losses)]))
        tqdm.write("")  # for better formatting

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses


##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################


import asyncio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import nest_asyncio
from tqdm import tqdm

nest_asyncio.apply()  # Apply nest_asyncio to allow nested event loops

import torch
import torch.nn as nn

class StronglyConvexLoss(nn.Module):
    def __init__(self, base_loss_fn, strong_convexity_param=0.1):
        super(StronglyConvexLoss, self).__init__()  # Ensure to call the base class constructor first
        self.base_loss_fn = base_loss_fn
        self.strong_convexity_param = strong_convexity_param

    def forward(self, outputs, targets):
        # Base loss representing the convex function q(x)
        convex_part = self.base_loss_fn(outputs, targets)
        
        # Quadratic term representing the strong convexity term (mu/2) * ||outputs||^2
        # This is applied to the output to ensure strong convexity with respect to output
        quadratic_term = (self.strong_convexity_param / 2) * torch.norm(outputs-targets)**2
        
        # Strongly convex loss as the sum of convex_part and quadratic_term
        return convex_part + quadratic_term


# Asynchronous training function for each client with strong convexity
async def train_client_with_strong_convexity(client_model, train_loader, device, local_epochs,
                                             loss_fn, lr, accumulation_steps,
                                             early_stopping_patience):
    client_model = client_model.to(device)
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
    client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=local_epochs)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()  # For mixed precision

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
        client_scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Return the trained model's state_dict and the loss history
    return client_model.state_dict(), client_losses


# Federated learning function with strong convexity
async def federated_learning_with_strong_convexity(clients_models, server_model,
                                                   clients_train_loaders, clients_test_loaders,
                                                   num_rounds=1, local_epochs=5,
                                                   aggregation_method='weighted_average',
                                                   loss_fn=None, lr=1e-5,
                                                   accumulation_steps=1, early_stopping_patience=3, 
                                                   save_dir='models/'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each client
        server_state_dict = server_model.state_dict()
        for client_model in clients_models:
            client_model.load_state_dict(server_state_dict)

        # Step 2: Asynchronously train each client
        async def train_client_task(i):
            return await train_client_with_strong_convexity(clients_models[i], clients_train_loaders[i], device, local_epochs, 
                                                           loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in range(len(clients_models))]
        results = await asyncio.gather(*tasks)

        # Collect client weights and losses from asynchronous results
        avg_client_losses = []
        for i, (state_dict, client_loss) in enumerate(results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))  # Calculate average loss for this client

            # Display each client's epoch loss in the progress bar
            for epoch_loss in client_loss:
                tqdm.write(f"Client {i + 1} Epoch Loss: {epoch_loss:.4f}")

        # Update tqdm bar with the average losses
        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in enumerate(avg_client_losses)]))
        tqdm.write("")  # for better formatting

        # Step 3: Aggregate the weights from all clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(loader.dataset) for loader in clients_train_loaders)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in enumerate(client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        # Step 4: Update the server model with aggregated weights
        server_model.load_state_dict(new_server_state_dict)

        # Step 5: Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        # Save models after each round
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses


#################################################
#################################################
#################################################


import asyncio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import nest_asyncio
from tqdm import tqdm

nest_asyncio.apply()  # Apply nest_asyncio to allow nested event loops




# Asynchronous function to train a random subset of clients with strong convexity in each round
async def train_random_clients_with_strong_convexity(client_model, train_loader, device, local_epochs,
                                                     loss_fn, lr, accumulation_steps, early_stopping_patience):
    client_model = client_model.to(device)
    client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
    client_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=local_epochs)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()  # For mixed precision

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
        client_scheduler.step()

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return client_model.state_dict(), client_losses






# Federated learning function with random client selection in each round
async def federated_learning_with_random_client_selection(clients_models, server_model,
                                                          clients_train_loaders, clients_test_loaders,
                                                          num_rounds=1, local_epochs=5, num_clients_per_round=2,
                                                          aggregation_method='weighted_average',
                                                          loss_fn=None, lr=1e-5,
                                                          accumulation_steps=1, early_stopping_patience=3,
                                                          save_dir='models/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")
        client_weights = []

        # Step 1: Send the global model (server_model) to each selected client
        server_state_dict = server_model.state_dict()
        
        # Randomly select a subset of clients
        selected_clients = np.random.choice(len(clients_models), num_clients_per_round, replace=False)
        
        # Print the indices of the selected clients
        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Step 2: Asynchronously train each selected client
        async def train_client_task(i):
            return await train_random_clients_with_strong_convexity(clients_models[i], clients_train_loaders[i], device,
                                                                   local_epochs, loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        avg_client_losses = []
        for i, (state_dict, client_loss) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))

            for epoch_loss in client_loss:
                tqdm.write(f"Client {i + 1} Epoch Loss: {epoch_loss:.4f}")

        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in zip(selected_clients, avg_client_losses)]))
        tqdm.write("") 

        # Step 3: Aggregate the weights from selected clients using the weighted average method
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        server_model.load_state_dict(new_server_state_dict)

        # Step 4: Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses




#############################################
#############################################
#############################################


"""

import random

# Federated learning function with balanced client selection
async def federated_learning_with_balanced_client_selection(clients_models, server_model,
                                                            clients_train_loaders, clients_test_loaders,
                                                            num_rounds=1, local_epochs=5,  # Add local_epochs here
                                                            min_clients_per_round=1, max_clients_per_round=5,
                                                            aggregation_method='weighted_average',
                                                            loss_fn=None, lr=1e-5,
                                                            accumulation_steps=1, early_stopping_patience=3,
                                                            save_dir='models/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    # Track client participation to ensure fair selection over rounds
    remaining_clients = set(range(len(clients_models)))

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        # Select a random number of clients within the specified range for each round
        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)

        # Ensure each client participates by prioritizing those not recently selected
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = set(range(len(clients_models)))  # Reset remaining clients
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
            remaining_clients -= set(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)
            remaining_clients -= set(selected_clients)

        # Print the indices of the selected clients
        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()
        
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Train each selected client, using the local_epochs value from the function's parameter
        async def train_client_task(i):
            return await train_random_clients_with_strong_convexity(clients_models[i], clients_train_loaders[i], device,
                                                                   local_epochs, loss_fn, lr, accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        avg_client_losses = []
        for i, (state_dict, client_loss) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))

            for epoch_loss in client_loss:
                tqdm.write(f"Client {i + 1} Epoch Loss: {epoch_loss:.4f}")

        tqdm.write(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in zip(selected_clients, avg_client_losses)]))
        tqdm.write("")

        # Aggregate the weights from selected clients
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        server_model.load_state_dict(new_server_state_dict)

        # Evaluate the server model on the test set
        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses


"""

#############################################################
#############################################################
#############################################################



import torch
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
import random
import asyncio





# Asynchronous function to train a random subset of clients with strong convexity in each round
async def train_random_clients_with_strong_convexity_time_delay(client_model, train_loader, device, local_epochs,
                                                     loss_fn, gamma_0, alpha, delay_t,
                                                     accumulation_steps, early_stopping_patience):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

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

    return client_model.state_dict(), client_losses




# Federated learning function with balanced client selection
async def federated_learning_with_balanced_client_selection(clients_models, server_model,
                                                            clients_train_loaders, clients_test_loaders,
                                                            num_rounds=1, local_epochs=5,
                                                            min_clients_per_round=1, max_clients_per_round=5,
                                                            aggregation_method='weighted_average',
                                                            loss_fn=None, gamma_0=1e-5, alpha=0.05,
                                                            accumulation_steps=1, early_stopping_patience=3,
                                                            save_dir='models/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    remaining_clients = set(range(len(clients_models)))

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)
        
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = set(range(len(clients_models)))
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
            remaining_clients -= set(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)
            remaining_clients -= set(selected_clients)

        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()
        
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        async def train_client_task(i):
            delay_t = random.uniform(0.1, 1.0)  # Example delay τ_t value (this may vary by client in practice)
            return await train_random_clients_with_strong_convexity_time_delay(clients_models[i], clients_train_loaders[i], device,
                                                                   local_epochs, loss_fn, gamma_0, alpha, delay_t,
                                                                   accumulation_steps, early_stopping_patience)

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        avg_client_losses = []
        for i, (state_dict, client_loss) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))

        print(f"Average Losses: " + ", ".join([f"Client {i + 1}: {loss:.4f}" for i, loss in zip(selected_clients, avg_client_losses)]))

        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        server_model.load_state_dict(new_server_state_dict)

        server_model.to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, test_loader in enumerate(clients_test_loaders):
                client_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    client_test_loss += loss.item()

                avg_client_loss = client_test_loss / len(test_loader)
                print(f"Client {i + 1} Test Loss: {avg_client_loss:.4f}")

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

    return server_model, training_losses


#############################################
#############################################
#############################################


"""
Main function 1111111111111111111111111111111111
"""

import asyncio
import time
import random
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

# Federated learning function with balanced client selection and delay tracking
async def federated_learning_with_balanced_client_selection_and_delay_tracking(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.07, alpha=0.001,  # Add gamma and alpha parameters
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    # Track client participation to ensure fair selection over rounds
    all_clients = set(range(len(clients_models)))
    remaining_clients = all_clients.copy()

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        print(f"\nStarting federated learning round {round_num + 1}/{num_rounds}")

        # Select a random number of clients within the specified range for each round
        num_clients_per_round = random.randint(min_clients_per_round, max_clients_per_round)

        # Ensure clients are selected without replacement across a round
        if len(remaining_clients) < num_clients_per_round:
            selected_clients = list(remaining_clients)
            remaining_clients = all_clients - set(selected_clients)
            additional_clients = random.sample(remaining_clients, num_clients_per_round - len(selected_clients))
            selected_clients.extend(additional_clients)
        else:
            selected_clients = random.sample(remaining_clients, num_clients_per_round)

        # Remove selected clients from the pool to avoid repetition
        remaining_clients -= set(selected_clients)

        print(f"Selected clients for round {round_num + 1}: {selected_clients}")

        client_weights = []
        server_state_dict = server_model.state_dict()

        # Load server weights into each selected client model
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        round_start_time = time.time()
        delays = {}

        # Train each selected client asynchronously with delay tracking
        async def train_client_task(i):
            start_time = time.time()
            result = await train_random_clients_with_strong_convexity_time_delay(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=delays.get(i, 0),
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            end_time = time.time()
            delays[i] = end_time - start_time
            return result

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        avg_client_losses = []
        for i, (state_dict, client_loss) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            avg_client_losses.append(sum(client_loss) / len(client_loss))

            for epoch_loss in client_loss:
                tqdm.write(f"Client {i + 1} Epoch Loss: {epoch_loss:.4f}")

        # Aggregate the weights
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)

        max_delay_client = max(delays, key=delays.get)
        max_delay = delays[max_delay_client]
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s (Client {max_delay_client + 1})")

        # Evaluate the server model
        total_loss, total_samples = 0, 0
        server_model.to(device).eval()

        with torch.no_grad():
            for test_loader in clients_test_loaders:
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)

        avg_loss = total_loss / total_samples
        print(f"Round {round_num + 1} - Server model evaluation loss: {avg_loss:.4f}")

        server_scheduler.step()

        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses



#######################




async def federated_learning_with_balanced_client_selection_and_delay_tracking_2(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.01, alpha=0.001,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    # To record selected clients and delays
    selected_clients_by_round = []
    max_delays_by_round = []

    # Track client participation to ensure fair selection over rounds
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

        round_start_time = time.time()
        delays = {}

        # Train each selected client asynchronously with delay tracking
        async def train_client_task(i):
            start_time = time.time()
            result = await train_random_clients_with_strong_convexity_time_delay(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=delays.get(i, 0),
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            end_time = time.time()
            delays[i] = end_time - start_time
            return result

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        for i, (state_dict, client_loss) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)

        # Track maximum delay
        max_delay_client = max(delays, key=delays.get)
        max_delay = delays[max_delay_client]
        max_delays_by_round.append((round_num + 1, max_delay_client, max_delay))
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s (Client {max_delay_client + 1})")

        # Aggregate weights
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses, selected_clients_by_round, max_delays_by_round



#####################################
#####################################
#####################################


# Asynchronous function to train a random subset of clients with strong convexity in each round
async def train_random_clients_with_strong_convexity_time_delay_3(client_model, train_loader, device, local_epochs,
                                                     loss_fn, gamma_0, alpha, delay_t,
                                                     accumulation_steps, early_stopping_patience):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float('inf')
    client_losses = []
    scaler = amp.GradScaler()

    start_time = time.time()  # Track the start time for the client
    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

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



async def federated_learning_with_balanced_client_selection_and_delay_tracking_3(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.01, alpha=0.001,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]

    # To record selected clients and delays
    selected_clients_by_round = []
    execution_times_by_round = []  # Track execution times

    # Track client participation to ensure fair selection over rounds
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
            result, client_losses, execution_time = await train_random_clients_with_strong_convexity_time_delay_3(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=0,  # Set any delay tracking logic here if needed
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
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

        # Aggregate weights
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses, selected_clients_by_round, execution_times_by_round




#######################################################




async def federated_learning_with_balanced_client_selection_and_delay_tracking_4(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.01, alpha=0.001,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    training_losses = [[] for _ in range(len(clients_models))]  # Initialize losses for all clients

    # To record selected clients and delays
    selected_clients_by_round = []
    execution_times_by_round = []  # Track execution times

    # Track client participation to ensure fair selection over rounds
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
            result, client_losses, execution_time = await train_random_clients_with_strong_convexity_time_delay_3(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=0,  # Set any delay tracking logic here if needed
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
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
            training_losses[i].extend(client_loss)  # Track losses for selected clients

        # For unselected clients, set their losses to None (or a placeholder)
        for i in set(range(len(clients_models))) - set(selected_clients):
            training_losses[i].append(None)  # This client did not participate in this round

        # Aggregate weights
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()
        save_all_models(clients_models, server_model, round_num, save_dir)

    return server_model, training_losses, selected_clients_by_round, execution_times_by_round


##########################################################
##########################################################


async def federated_learning_with_balanced_client_selection_and_delay_tracking_5(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.01, alpha=0.001,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    
    training_losses = [[] for _ in range(len(clients_models))]  # Initialize losses for all clients
    server_losses_per_round = []  # Track global server model losses (weighted average of client losses)

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
        client_losses_per_round = []  # Track the losses of selected clients in this round
        server_state_dict = server_model.state_dict()

        # Load server weights into each selected client model
        for i in selected_clients:
            clients_models[i].load_state_dict(server_state_dict)

        # Train each selected client asynchronously and track execution times
        async def train_client_task(i):
            result, client_losses, execution_time = await train_random_clients_with_strong_convexity_time_delay_3(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=0,
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            return result, client_losses, execution_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        execution_times = [execution_time for _, _, execution_time in results]
        execution_times_by_round.append(execution_times)

        max_delay = max(execution_times) - min(execution_times)
        print(f"Round {round_num + 1} - Maximum delay: {max_delay:.2f}s")

        total_samples = 0
        total_weighted_loss = 0

        for i, (state_dict, client_loss, _) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            client_dataset_size = len(clients_train_loaders[i].dataset)
            total_samples += client_dataset_size
            total_weighted_loss += client_dataset_size * np.mean(client_loss)

        for i in set(range(len(clients_models))) - set(selected_clients):
            training_losses[i].append(None)

        # Aggregate weights
        if aggregation_method == 'weighted_average':
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()
        save_all_models(clients_models, server_model, round_num, save_dir)

        # Compute weighted average server loss (from clients' losses)
        weighted_server_loss = total_weighted_loss / total_samples if total_samples > 0 else None
        server_losses_per_round.append(weighted_server_loss)
        print(f"Server Loss (Weighted Average of Clients' Losses) - Round {round_num + 1}: {weighted_server_loss:.4f}")

    # Save server model losses as a numpy array
    np.save(f"{save_dir}/server_model_losses.npy", np.array(server_losses_per_round))

    return server_model, training_losses, selected_clients_by_round, execution_times_by_round, server_losses_per_round



##########################################################
##########################################################

"""

async def federated_learning_with_balanced_client_selection_and_delay_tracking_5(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=3,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.01, alpha=0.001,
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=num_rounds)
    
    training_losses = [[] for _ in range(len(clients_models))]  # Initialize losses for all clients
    server_losses_per_round = []  # Track global server model losses

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
            result, client_losses, execution_time = await train_random_clients_with_strong_convexity_time_delay_3(
                clients_models[i], clients_train_loaders[i], device,
                local_epochs, loss_fn, gamma, alpha,
                delay_t=0,
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
        if aggregation_method == 'weighted_average':
            total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
            new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

            for key in new_server_state_dict:
                for i, client_weight in zip(selected_clients, client_weights):
                    weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                    new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)
        server_scheduler.step()
        save_all_models(clients_models, server_model, round_num, save_dir)

        # Compute server loss using test data
        server_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for test_loader in clients_test_loaders:
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = server_model(inputs)
                    batch_loss = loss_fn(outputs, targets) if loss_fn else F.mse_loss(outputs, targets)
                    total_loss += batch_loss.item() * len(targets)
                    total_samples += len(targets)

        avg_server_loss = total_loss / total_samples if total_samples > 0 else None
        server_losses_per_round.append(avg_server_loss)
        print(f"Server Model Loss (Round {round_num + 1}): {avg_server_loss:.4f}")

    # Save server model losses as a numpy array
    np.save(f"{save_dir}/server_model_losses.npy", np.array(server_losses_per_round))

    return server_model, training_losses, selected_clients_by_round, execution_times_by_round, server_losses_per_round

"""






    
    