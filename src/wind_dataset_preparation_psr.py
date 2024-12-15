#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:49:24 2024

@author: forootan
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Function to clean the date-time string by removing fractional seconds
def clean_date_time_string(date_time_str):
    return date_time_str.split('.')[0]

# Function to check if the date-time string is in the correct format
def is_valid_date_time(date_time_str, date_time_format):
    try:
        datetime.strptime(date_time_str, date_time_format)
        return True
    except ValueError:
        return False

# Function to convert to Unix time
def convert_to_unix_time(date_time_str, date_time_format):
    try:
        dt = datetime.strptime(date_time_str, date_time_format)
        return int(dt.timestamp())
    except ValueError:
        return None

def map_unix_time_to_range(unix_time_array, feature_range=(-1, 1)):
    unix_time_array_reshaped = unix_time_array.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_unix_time_array = scaler.fit_transform(unix_time_array_reshaped)
    return scaled_unix_time_array.flatten()

def loading_wind(csv_file_path):
    #csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.dropna(axis=0, how='any', inplace=True)
    wind_data = df.to_numpy()
    x_y = wind_data[:, 1:3]
    data_to_resample = wind_data[:, 6:wind_data.shape[1]-1]
    num_all_zeros = np.sum(np.all(wind_data[:, 5:] == 0, axis=1))
    print("Number of rows where all values from column 5 onwards are zero:", num_all_zeros)
    mask = np.any(wind_data[:, 5:] != 0, axis=1)
    filtered_wind_data = wind_data[mask]
    filtered_x_y = x_y[mask]
    date_time_format = '%d/%m/%y %H:%M'
    headers = df.columns.tolist()
    date_time_headers = headers[6:-1]
    cleaned_date_time_headers = [clean_date_time_string(dt_str) for dt_str in date_time_headers]
    date_times = [datetime.strptime(dt, date_time_format) for dt in cleaned_date_time_headers]
    time_index = pd.DatetimeIndex(date_times)
    wind_power_df = pd.DataFrame(filtered_wind_data[:, 6:wind_data.shape[1]-1], columns=time_index, index=filtered_wind_data[:, 0])
    wind_power_df = wind_power_df.T
    resampled_wind_power_df = wind_power_df.resample('3H').mean()
    resampled_wind_power_df = resampled_wind_power_df.T
    filtered_wind_power = resampled_wind_power_df.values
    new_date_times = resampled_wind_power_df.columns
    unix_times = [int(dt.timestamp()) for dt in new_date_times]
    unix_time_array = np.array(unix_times)
    scaled_unix_time_array = map_unix_time_to_range(unix_time_array, feature_range=(-1, 1)).reshape(-1, 1)
    
    # Ensure that filtered_wind_power rows match the number of locations in filtered_x_y
    if filtered_wind_power.shape[0] != filtered_x_y.shape[0]:
        raise ValueError(f"Shape mismatch: filtered_wind_power rows ({filtered_wind_power.shape[0]}) "
                         f"do not match filtered_x_y rows ({filtered_x_y.shape[0]})")
    
    return scaled_unix_time_array, filtered_x_y, filtered_wind_power

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

def rotate_coordinates(lat, lon, pole_lat, pole_lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    pole_lat, pole_lon = np.deg2rad(pole_lat), np.deg2rad(pole_lon)

    lon = lon - pole_lon

    rot_lat = np.arcsin(np.sin(lat) * np.sin(pole_lat) + 
                        np.cos(lat) * np.cos(pole_lat) * np.cos(lon))
    
    rot_lon = np.arctan2(np.cos(lat) * np.sin(lon),
                         np.sin(lat) * np.cos(pole_lat) - 
                         np.cos(lat) * np.sin(pole_lat) * np.cos(lon))

    return np.rad2deg(rot_lat), np.rad2deg(rot_lon)

def extract_wind_speed_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        pole_lat = nc.variables['rotated_pole'].grid_north_pole_latitude
        pole_lon = nc.variables['rotated_pole'].grid_north_pole_longitude

        rlat = nc.variables['rlat'][:]
        rlon = nc.variables['rlon'][:]

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        rot_target_lats, rot_target_lons = rotate_coordinates(target_lats, target_lons, pole_lat, pole_lon)
        distances, indices = tree.query(np.column_stack((rot_target_lats, rot_target_lons)))

        wind_speed = nc.variables['sfcWind'][:]
        extracted_wind_speed = wind_speed[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_wind_speed == extracted_wind_speed[0, 0]):
            raise ValueError("Extracted wind speed data is identical for all points. Check extraction process.")

        return extracted_wind_speed, lats, lons

def load_real_wind_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.dropna(axis=0, how='any', inplace=True)
    wind_data = df.to_numpy()
    x_y = wind_data[:, 1:3].astype(float)
    mask = np.any(wind_data[:, 5:] != 0, axis=1)
    filtered_x_y = x_y[mask]
    return filtered_x_y

def interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_wind_speeds = np.zeros((wind_speeds.shape[0], target_points.shape[0]))

    for t in range(wind_speeds.shape[0]):
        # Reshape wind_speeds[t] to a 2D array
        wind_speed_t = wind_speeds[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_wind_speeds[t] = interpolated

    nan_count = np.isnan(interpolated_wind_speeds).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_wind_speeds


######################################################
######################################################
######################################################


def extract_pressure_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        rotated_lat_lon = nc.variables['rotated_latitude_longitude']
        
        # Extract projection parameters
        pole_lat = rotated_lat_lon.grid_north_pole_latitude
        pole_lon = rotated_lat_lon.grid_north_pole_longitude
        
        print(f"Pole Latitude: {pole_lat}")
        print(f"Pole Longitude: {pole_lon}")

        rlat = nc.variables['rlat'][:]
        rlon = nc.variables['rlon'][:]

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        rot_target_lats, rot_target_lons = rotate_coordinates(target_lats, target_lons, pole_lat, pole_lon)
        distances, indices = tree.query(np.column_stack((rot_target_lats, rot_target_lons)))

        pressure = nc.variables['ps'][:]  # Assuming 'psl' is the pressure variable name
        extracted_pressure = pressure[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_pressure == extracted_pressure[0, 0]):
            raise ValueError("Extracted pressure data is identical for all points. Check extraction process.")

        return extracted_pressure, lats, lons



def interpolate_pressure(data, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_data = np.zeros((data.shape[0], target_points.shape[0]))

    for t in range(data.shape[0]):
        # Reshape data[t] to a 2D array
        data_t = data[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_data[t] = interpolated

    nan_count = np.isnan(interpolated_data).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_data


"""

# Example usage
nc_file_path = 'nc_files/dataset-projections-cordex-domains-single-levels-69ac4dd9-7e75-46a0-8eef-7be736876191/ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

# Extract pressure data
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)



#####################################################
#####################################################
#####################################################

# Example usage
nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)




print(f"Shape of extracted wind speed: {wind_speeds.shape}")
print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")
print(wind_speeds[:5, :5])

target_points = load_real_wind_csv(csv_file_path)
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)

scaled_unix_time_array, filtered_x_y, filtered_wind_power = loading_wind()

interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)

"""

#########################################################
#########################################################
#########################################################


from sklearn.preprocessing import MinMaxScaler

def scale_target_points(target_points):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_target_points = scaler.fit_transform(target_points)
    return scaled_target_points

#scaled_target_points = scale_target_points(target_points)



def scale_interpolated_data(data, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    num_time_steps, num_points = data.shape
    scaled_data = np.zeros_like(data)

    for t in range(num_time_steps):
        scaled_data[t, :] = scaler.fit_transform(data[t, :].reshape(-1, 1)).flatten()
    
    return scaled_data

#scaled_wind_speeds = scale_interpolated_data(interpolated_wind_speeds)


#scaled_pressure = scale_interpolated_data(interpolated_pressure)

#scaled_wind_power = scale_interpolated_data(filtered_wind_power)


def repeat_target_points(scaled_target_points, num_time_steps):
    repeated_target_points = np.repeat(scaled_target_points, num_time_steps, axis=0)
    return repeated_target_points

# Number of time steps (from scaled_wind_speeds)
#num_time_steps = scaled_wind_speeds.shape[0]
#repeated_scaled_target_points = repeat_target_points(scaled_target_points, num_time_steps)

#print(f"Shape of repeated_scaled_target_points: {repeated_scaled_target_points.shape}")





def combine_data(normalized_x_y, scaled_unix_time_array,
                 scaled_wind_speeds,
                 scaled_pressure,
                 scaled_wind_power):
    """
    Combine normalized_x_y, scaled_unix_time_array, and flattened filtered_wind_power into a single array.

    Parameters:
    - normalized_x_y (np.array): The normalized x and y coordinates (shape: (232, 2)).
    - scaled_unix_time_array (np.array): The scaled Unix timestamps (shape: (8783, 1)).
    - filtered_wind_power (np.array): The filtered wind power data (shape: (232, 8783)).

    Returns:
    - np.array: Combined array with shape (232*8783, 3).
    """
    num_rows = normalized_x_y.shape[0]  # Number of rows (232)
    num_columns = scaled_unix_time_array.shape[0]  # Number of columns (8783)

    # Repeat normalized_x_y for each timestamp in scaled_unix_time_array
    repeated_x_y = np.repeat(normalized_x_y, num_columns, axis=0)

    # Repeat scaled_unix_time_array for each row in normalized_x_y
    repeated_unix_time = np.tile(scaled_unix_time_array, (num_rows, 1))

    # Flatten the filtered_wind_power while preserving the sequence
    flattened_wind_power = scaled_wind_power.flatten()
    
    flattened_pressure = scaled_pressure.flatten()
    
    flattened_wind_speeds = scaled_wind_speeds.flatten()
    
    # Combine all three arrays into one
    combined_array = np.column_stack((repeated_x_y, repeated_unix_time.flatten(),
                                      flattened_pressure,
                                      flattened_wind_speeds,
                                      flattened_wind_power))

    return combined_array



"""
# Combine the data
combined_array = combine_data(scaled_target_points, scaled_unix_time_array,
                              scaled_wind_speeds,
                              scaled_pressure,
                              scaled_wind_power)
"""


