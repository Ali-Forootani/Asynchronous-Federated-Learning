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
                          federated_learning_async_2)


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

num_clients = 3  # Number of clients (adjust as necessary)
num_rounds = 3  # Number of federated learning rounds

# Data preparation (already divided for each client)
wind_dataset_instance = LSTMDataPreparationFL(combined_array[:, :5], combined_array[:, 5:])
client_data = wind_dataset_instance.partition_data(num_clients)

input_size = 5  # Number of input features
output_size = 1  # Number of output features
learning_rate = 1e-3
num_epochs = 5

# Define client-specific model configurations (e.g., different numbers of hidden layers, hidden features)
client_configs = [
    {"hidden_features": 32, "hidden_layers": 3, "learning_rate": 1e-3},  # Client 1 configuration
    {"hidden_features": 16, "hidden_layers": 3, "learning_rate": 1e-3}, # Client 2 configuration
    {"hidden_features": 64, "hidden_layers": 3, "learning_rate": 5e-4}  # Client 3 configuration
]

# Initialize lists to store client-specific models, optimizers, etc.
clients_models = []
clients_optimizers = []
clients_schedulers = []
clients_trainers = []
clients_train_loaders = []
clients_test_loaders = []

# Create models, optimizers, schedulers, and trainers for each client
for i, (client_coords, ith_client_data) in enumerate(client_data):
    # Get the configuration for this client
    config = client_configs[i]

    # Initialize the LSTMDeepModel for this client based on its specific configuration
    lstm_model = LSTMDeepModel(
        input_size=input_size,
        hidden_size=config["hidden_features"],
        num_layers=config["hidden_layers"],
        output_size=output_size,
        learning_rate=config["learning_rate"]
    )

    # Set up the optimizer and scheduler for this client's model
    optim_adam = lstm_model.optimizer_func()
    scheduler = lstm_model.scheduler_setting()

    # Prepare data for this client (seq length, randomize the test/train split)
    client_instance = LSTMDataPreparationFL(client_coords, ith_client_data)
    (x_train_seq, u_train_seq, train_loader, test_loader) = client_instance.prepare_data_random(test_data_size=0.9999)

    # Save the data loaders for later use in training
    clients_train_loaders.append(train_loader)
    clients_test_loaders.append(test_loader)

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
    clients_schedulers.append(scheduler)
    clients_trainers.append(trainer)

# Now each client has a different LSTM model structure (number of layers, hidden features, etc.)


hidden_features = 64
hidden_layers = 3

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

num_rounds = 5
local_epochs = 20
lr = 1e-3
server_patience = 100
accumulation_steps = 50
early_stopping_patience = 500
aggregation_method = "weighted_average"
loss_fn = nn.MSELoss()


#################################################
#################################################

import numpy as np
import matplotlib.pyplot as plt
import asyncio

### conda install bjrn::nest_asyncio 
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()




async def main():
    server_model, training_losses = await federated_learning_async_2(
        clients_models=clients_models,
        server_model=global_model.to(device),
        clients_train_loaders=clients_train_loaders,
        clients_test_loaders=clients_test_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        aggregation_method=aggregation_method,
        loss_fn=loss_fn,
        lr=lr,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience, 
        save_dir='models/'
    )
    
    # Convert training losses for each client to a NumPy array and save them
    for i, client_losses in enumerate(training_losses):
        client_losses_np = np.array(client_losses)
        np.save(f'training_losses/training_losses_client_{i}.npy', client_losses_np)

   # Plotting the training losses for each client
    plt.figure(figsize=(10, 6))
    for i, client_losses in enumerate(training_losses):
       plt.plot(client_losses, label=f'Client {i + 1}', linewidth=2)  # Adjust line width if needed

    plt.xscale('log')  # Set x-axis to log scale
    plt.title('Training Losses per Client')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_losses/training_losses_plot.png')  # Save the plot
    plt.show()  # Display the plot


# Run the async main function
asyncio.run(main())




#################################################
#################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#################################################
#################################################
def evaluate_and_save_test_losses(clients_models, clients_test_loaders,
                                  device = device):
    """
    Evaluate each client's model on the test dataset and save the test losses.

    Args:
        clients_models: List of client models.
        clients_test_loaders: List of data loaders for clients' test data.
        device: The device (CPU or GPU) to perform computations on.
    """
    test_losses = []

    # Evaluate each client model
    for i, client_model in enumerate(clients_models):
        client_model.to(device)  # Move model to the correct device
        client_model.eval()
        total_loss = 0
        total_samples = 0
        
        test_loader = clients_test_loaders[i]  # No need to call .to(device) on DataLoader
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                # Move inputs and targets to the appropriate device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = client_model(inputs)
                loss = nn.MSELoss()(outputs, targets)
                total_loss += loss.item() * len(targets)
                total_samples += len(targets)

        avg_loss = total_loss / total_samples
        test_losses.append(avg_loss)
        print(f"Client {i + 1} Test Loss: {avg_loss:.4f}")

    # Save test losses to a NumPy file
    np.save('test_losses.npy', np.array(test_losses))
    print("Test losses saved to 'test_losses.npy'.")


# Execute federated learning and save losses
#server_model, training_losses = federated_learning_2(
#    clients_models, global_model, clients_train_loaders, clients_test_loaders, num_rounds=10
#)

# Evaluate each client on the test dataset and save test losses
evaluate_and_save_test_losses(clients_models, clients_test_loaders)

"""
# Plotting the training loss curve for each client
for i, losses in enumerate(training_losses):
    plt.plot(losses, label=f'Client {i + 1}')

plt.title('Training Loss per Client')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Save the figure to a local directory
plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
plt.close()  # Close the figure after saving to avoid display

# After federated learning is done, save the final global model
torch.save(server_model.state_dict(), "final_federated_model.pth")
"""


#################################################
#################################################
# After federated learning is done, you can evaluate or save the final global model
# Example: save the global model state
torch.save(global_model.state_dict(), "final_federated_model.pth")




test_losses = np.load('test_losses.npy')




import numpy as np
import matplotlib.pyplot as plt

# Plotting the training losses for each client after loading them
plt.figure(figsize=(10, 6))

# Loop to load and plot each client's training losses
#num_clients = len(training_losses)  # Replace with the number of clients if needed
for i in range(num_clients):
    client_losses = np.load(f'training_losses/training_losses_client_{i}.npy')
    plt.plot(client_losses, label=f'Client {i + 1}', linewidth=2)  # Adjust line width if needed

# Set x-axis to log scale and add labels
plt.xscale('log')
plt.yscale('log')
plt.title('Training Losses per Client')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()  # Display the plot






