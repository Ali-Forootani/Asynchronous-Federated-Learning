import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the files (update this to the correct path if needed)
directory = "./"

# Prefix for the files to load
prefix = "synchronous_svm_server_losses_clients_10_epochs_100_rounds"

# List to hold the data and labels
data_list = []
labels = []

# Loop through the files in the directory
for file_name in os.listdir(directory):
    if file_name.startswith(prefix) and file_name.endswith(".npy"):
        # Load the .npy file
        data = np.load(os.path.join(directory, file_name))
        print("=======================")
        print(data.shape)
        print(file_name)
        print("=======================")
        # Only include data with length == 1000
        if len(data) == 1000:
            data_list.append(data)
            # Use the file name (without extension) as a label
            labels.append(file_name.replace(".npy", ""))
        else:
            print(f"Skipping {file_name} as its length is not exactly 1000.")
import matplotlib.pyplot as plt
import numpy as np

# Ensure we have data to process
if data_list:
    # Convert to a NumPy array for easier processing
    data_array = np.array(data_list)
    
    # Compute the average across all curves
    average_curve = np.mean(data_array, axis=0)
    
    # Plotting
    plt.figure(figsize=(8, 6), dpi=300)  # High-quality figure with 300 DPI
    
    # Plot each individual curve with faded color
    for i, data in enumerate(data_list):
        plt.plot(data, label=labels[i], alpha=0.3)
    
    # Plot the average curve in bold
    plt.plot(average_curve, label="Average Curve", linewidth=2.5, color="black")
    
    # Set the y-axis to log scale
    plt.yscale('log')
    
    # Set titles and labels with larger font size and bold style
    plt.title("Server Loss During PFL - Total Client: 10, Client Fraction: 20.0% - 90.0%",
              fontsize=18,)
    plt.xlabel("Rounds", fontsize=20,)
    plt.ylabel("Hinge Loss", fontsize=20, )
    
    # Adjust tick font sizes
    plt.xticks(fontsize=16,)
    plt.yticks(fontsize=16,)
    
    # Add grid for better readability
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the figure as a high-quality PNG file
    plt.savefig("synchronous_svm_server_loss_plot.png", dpi=300, bbox_inches='tight')  # Save with tight bounding box
    
    # Show the figure
    plt.show()
else:
    print("No datasets to plot.")


