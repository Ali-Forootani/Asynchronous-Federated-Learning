#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:44:17 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:53:16 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:23:59 2024

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
root_dir = setting_directory(2)



from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from siren_modules import Siren

import warnings
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#####################################
#####################################


from wind_dataset_preparation_psr import (
    extract_pressure_for_germany,
    extract_wind_speed_for_germany,
    load_real_wind_csv,
    interpolate_wind_speed,
    loading_wind,
    interpolate_pressure,
    scale_interpolated_data,
    combine_data,
    repeat_target_points,
    scale_target_points
    )


from wind_dataset_preparation import WindDataGen, RNNDataPreparation, LSTMDataPreparation, LSTMDataPreparationFL
from wind_deep_simulation_framework import WindDeepModel, RNNDeepModel, LSTMDeepModel, FLServer
from wind_loss import wind_loss_func
from wind_trainer import (Trainer, RNNTrainer,
                          LSTMTrainer, LSTMTrainerFL,
                          federated_learning_2,
                          federated_learning_5,
                          federated_learning_6,
                          federated_learning_8,
                          federated_learning_async,
                          StronglyConvexLoss,
                          federated_learning_with_strong_convexity,
                          federated_learning_with_balanced_client_selection,
                          federated_learning_with_balanced_client_selection_and_delay_tracking)


######################################
######################################

# Example usage
nc_file_path = root_dir + '/nc_files/dataset-projections-2020/ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
csv_file_path = root_dir + '/nc_files/Results_2020_REMix_ReSTEP_hourly_REF.csv'

# Extract pressure data
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)


# Example usage
nc_file_path = root_dir + '/nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
csv_file_path = root_dir + '/nc_files/Results_2020_REMix_ReSTEP_hourly_REF.csv'

wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)



print(f"Shape of extracted wind speed: {wind_speeds.shape}")
print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")


target_points = load_real_wind_csv(csv_file_path)
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)

scaled_unix_time_array, filtered_x_y, filtered_wind_power = loading_wind(csv_file_path)

interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)



scaled_wind_speeds = scale_interpolated_data(interpolated_wind_speeds)


scaled_pressure = scale_interpolated_data(interpolated_pressure)

scaled_wind_power = scale_interpolated_data(filtered_wind_power)


scaled_target_points = scale_target_points(target_points)

# Number of time steps (from scaled_wind_speeds)
num_time_steps = scaled_wind_speeds.shape[0]
repeated_scaled_target_points = repeat_target_points(scaled_target_points, num_time_steps)

print(f"Shape of repeated_scaled_target_points: {repeated_scaled_target_points.shape}")



# Combine the data
combined_array = combine_data(scaled_target_points, scaled_unix_time_array,
                              scaled_wind_speeds,
                              scaled_pressure,
                              scaled_wind_power)


######################################
######################################



# Federated Learning Setup for LSTM Model on Multiple Clients

num_clients = 4  # Number of clients

max_clients_per_round = 3



# Parameters and main function call
num_rounds = 10
local_epochs = 20000
lr = 1e-4
accumulation_steps = 50
early_stopping_patience = 5000
aggregation_method = "weighted_average"
strong_convexity_param = 0.000






# Data preparation (already divided for each client)
wind_dataset_instance = LSTMDataPreparationFL(combined_array[:, :5], combined_array[:, 5:])
client_data = wind_dataset_instance.partition_data(num_clients)

input_size = 5  # Number of input features
hidden_features = 32
hidden_layers = 3  # Number of LSTM layers
output_size = 1  # Number of output features
learning_rate = 1e-3
num_epochs = 5

# Initialize lists to store client-specific models, optimizers, etc.
clients_models = []
clients_optimizers = []
clients_schedulers = []
clients_trainers = []
clients_train_loaders = []
clients_test_loaders = []
clients_train_labels = []
clients_test_labels = []



# Create models, optimizers, schedulers, and trainers for each client
for i, (client_coords, ith_client_data) in enumerate(client_data):
    # Initialize the LSTMDeepModel for this client
    lstm_model = LSTMDeepModel(
        input_size=input_size,
        hidden_size=hidden_features,
        num_layers=hidden_layers,
        output_size=output_size,
        learning_rate=learning_rate
    )

    # Set up the optimizer and scheduler for this client's model
    optim_adam = lstm_model.optimizer_func()
    scheduler = lstm_model.scheduler_setting()

    # Prepare data for this client (seq length, randomize the test/train split)
    client_instance = LSTMDataPreparationFL(client_coords, ith_client_data)
    (x_train_seq, u_train_seq, train_loader,
    test_loader) = client_instance.prepare_data_random(test_data_size=0.9995)

    # Save the data loaders and labels for later use in training
    clients_train_loaders.append(train_loader)
    clients_test_loaders.append(test_loader)
    #clients_train_labels.append(train_label)
    #clients_test_labels.append(test_label)

    # Create the LSTMTrainer for this client's model
    trainer = LSTMTrainerFL(
        model=lstm_model,
        optim_adam=optim_adam,
        scheduler=scheduler,
        num_epochs=num_epochs
    )

    # Store model, optimizer, scheduler, and trainer for later use
    clients_models.append(lstm_model.to(device))
    clients_optimizers.append(optim_adam)
    clients_schedulers.append(scheduler )
    clients_trainers.append(trainer)

# Initialize the global model and server
global_model = LSTMDeepModel(
    input_size=input_size,
    hidden_size=hidden_features,
    num_layers=hidden_layers,
    output_size=output_size,
    learning_rate=learning_rate
)

# Create a server instance for federated learning
server = FLServer(global_model.to(device))

#################################################
#################################################
#################################################

"""
global_model, training_losses = federated_learning_5(
    clients_models=clients_models,
    server_model=global_model,
    clients_train_loaders=clients_train_loaders,
    clients_test_loaders=clients_test_loaders,
    num_rounds=num_rounds,
    local_epochs=num_epochs,
    aggregation_method='average')
"""


###############################################
###############################################
###############################################

# Define the strongly convex loss function
loss_fn = StronglyConvexLoss(nn.MSELoss(), strong_convexity_param)





#####################

import asyncio

### conda install bjrn::nest_asyncio 
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


########################################
########################################
########################################

"""
federated_learning_with_balanced_client_selection_and_delay_tracking(
    clients_models, server_model,
    clients_train_loaders, clients_test_loaders,
    num_rounds=1, local_epochs=5,
    min_clients_per_round=1, max_clients_per_round=5,
    aggregation_method='weighted_average',
    loss_fn=None, lr=1e-5, gamma=0.9, alpha=0.1,  # Add gamma and alpha parameters
    accumulation_steps=1, early_stopping_patience=3,
    save_dir='models/'
):
"""



async def main():
    server_model, training_losses = await federated_learning_with_balanced_client_selection_and_delay_tracking(
        clients_models=clients_models,
        server_model=global_model.to(device),
        clients_train_loaders=clients_train_loaders,
        clients_test_loaders=clients_test_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        min_clients_per_round=1,
        max_clients_per_round= max_clients_per_round,
        aggregation_method=aggregation_method,
        loss_fn=loss_fn,
        lr=lr,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience, 
        save_dir='models/'
    )
    
    # Save and plot training losses
    for i, client_losses in enumerate(training_losses):
        client_losses_np = np.array(client_losses)
        np.save(f'training_losses/training_losses_client_{i}.npy', client_losses_np)

    plt.figure(figsize=(10, 6))
    for i, client_losses in enumerate(training_losses):
        plt.plot(client_losses, label=f'Client {i + 1}', linewidth=2)

    plt.xscale('log')
    plt.title('Training Losses per Client')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_losses/training_losses_plot.png')
    plt.show()

# Run the async main function
asyncio.run(main())




#################################################
#################################################



import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
import torch
import torch.nn as nn

class StronglyConvexLoss(nn.Module):
    def __init__(self, base_loss_fn = nn.MSELoss(), strong_convexity_param=0.1):
        super(StronglyConvexLoss, self).__init__()  # Ensure to call the base class constructor first
        self.base_loss_fn = base_loss_fn
        self.strong_convexity_param = strong_convexity_param

    def forward(self, outputs, targets):
        # Base loss representing the convex function q(x)
        convex_part = self.base_loss_fn(outputs, targets)
        
        # Quadratic term representing the strong convexity term (mu/2) * ||outputs||^2
        # This is applied to the output to ensure strong convexity with respect to output
        quadratic_term = (self.strong_convexity_param / 2) * torch.norm(outputs)**2
        
        # Strongly convex loss as the sum of convex_part and quadratic_term
        return convex_part + quadratic_term


strongly_convex_loss_fn = StronglyConvexLoss()  # Adjust alpha as needed

#################################################
#################################################
def evaluate_and_save_test_losses(clients_models, clients_test_loaders, loss_fn, device=device):
    """
    Evaluate each client's model on the test dataset and save the test losses with a specified loss function.

    Args:
        clients_models: List of client models.
        clients_test_loaders: List of data loaders for clients' test data.
        loss_fn: The loss function to use for evaluation (e.g., StronglyConvexLoss).
        device: The device (CPU or GPU) to perform computations on.
    """
    test_losses = []

    # Evaluate each client model
    for i, client_model in enumerate(clients_models):
        client_model.to(device)  # Move model to the correct device
        client_model.eval()
        total_loss = 0
        total_samples = 0

        test_loader = clients_test_loaders[i]
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                # Move inputs and targets to the appropriate device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = client_model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item() * len(targets)
                total_samples += len(targets)

        avg_loss = total_loss / total_samples
        test_losses.append(avg_loss)
        print(f"Client {i + 1} Test Loss: {avg_loss:.4f}")

    # Save test losses to a NumPy file
    np.save('test_losses.npy', np.array(test_losses))
    print("Test losses saved to 'test_losses.npy'.")

#################################################
#################################################

# Call the function with strongly convex loss function after federated learning rounds
# evaluate_and_save_test_losses(clients_models, clients_test_loaders, strongly_convex_loss_fn)

# Plotting the training losses for each client after loading them
def plot_client_losses(num_clients):
    plt.figure(figsize=(10, 6))

    # Loop to load and plot each client's training losses
    for i in range(num_clients):
        client_losses = np.load(f'training_losses/training_losses_client_{i}.npy')
        plt.plot(client_losses, label=f'Client {i + 1}', linewidth=2)

    # Set x-axis to log scale and add labels
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Training Losses per Client')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
plot_client_losses(num_clients)





import numpy as np
import matplotlib.pyplot as plt

def plot_client_losses_2(num_clients):
    plt.figure(figsize=(12, 8))

    # Define different line styles
    line_styles = ['-', '--', '-.', ':']

    # Loop to load and plot each client's training losses
    for i in range(num_clients):
        client_losses = np.load(f'training_losses/training_losses_client_{i}.npy')
        style = line_styles[i % len(line_styles)]  # Cycle through line styles
        plt.plot(client_losses, label=f'Client {i + 1}', linewidth=3, linestyle=style)

    # Set x-axis and y-axis to log scale
    #plt.xscale('log')
    plt.yscale('log')

    # Add title and labels with larger font sizes
    plt.title('Training Losses per Client', fontsize=20)
    plt.xlabel('Iterations Rounds', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    # Configure grid and legend with larger font sizes
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Display the plot
    plt.show()




plot_client_losses_2(num_clients)







#################################################
#################################################

# Save the final global model state after federated learning rounds
torch.save(global_model.state_dict(), "final_federated_model.pth")

# Load and print test losses
test_losses = np.load('test_losses.npy')
print("Loaded test losses:", test_losses)






