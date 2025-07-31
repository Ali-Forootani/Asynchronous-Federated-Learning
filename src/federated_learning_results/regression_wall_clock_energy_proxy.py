import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the files
directory = "./"

# Prefixes for wall-clock and energy proxy
prefixes = {
    "Cumulative Wall-Clock Time (s)": "logistic_wall_clock_clients_10_epochs_50_updates",
    "Energy Proxy (Wh)": "logistic_energy_proxy_clients_10_epochs_50_updates"
}

# Function to load and plot data
def plot_metrics(prefix, title, ylabel, save_name, required_length=None):
    data_list = []
    labels = []

    for file_name in os.listdir(directory):
        if file_name.startswith(prefix) and file_name.endswith(".npy"):
            data = np.load(os.path.join(directory, file_name))
            if required_length and len(data) != required_length:
                print(f"Skipping {file_name} (length {len(data)})")
                continue
            print(f"Loaded {file_name} with shape {data.shape}")
            data_list.append(data)
            labels.append(file_name.replace(".npy", ""))

    if not data_list:
        print(f"No datasets found for prefix {prefix}")
        return

    plt.figure(figsize=(8, 6), dpi=300)

    if len(data_list) > 1:
        # Convert to array for averaging
        max_len = max(len(arr) for arr in data_list)
        data_array = np.array([np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in data_list])
        average_curve = np.mean(data_array, axis=0)

        # Plot individual runs
        for i, data in enumerate(data_list):
            plt.plot(data, label=labels[i], alpha=0.3)
        # Plot average
        plt.plot(average_curve, label="Average Curve", linewidth=2.5, color="black")
    else:
        # Only one dataset → plot it directly
        plt.plot(data_list[0], label=labels[0], linewidth=2.5, color="black")

    plt.title(title, fontsize=18)
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    #plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


# Generate plots (filter to only include length 1000 for consistency)
plot_metrics(prefixes["Cumulative Wall-Clock Time (s)"], 
             "Cumulative Wall-Clock Time During AFL", 
             "Time (s)", 
             "wall_clock_plot.png",
             required_length=400)

plot_metrics(prefixes["Energy Proxy (Wh)"], 
             "Cumulative Energy Proxy During AFL", 
             "Energy (Wh)", 
             "energy_proxy_plot.png",
             required_length=400)
