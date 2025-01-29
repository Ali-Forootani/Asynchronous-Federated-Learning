import os
import numpy as np
import matplotlib.pyplot as plt

# Filter files starting with 'svm_server_losses_clients_10_epochs_100_updates' and ending with '.npy'
files = [
    file for file in os.listdir('.')
    if file.startswith('svm_server_losses_clients_10_epochs_100_updates') and file.endswith('.npy')
]

print(files)

# Load data and plot
plt.figure(figsize=(10, 6))

for file in files:
    data = np.load(file)
    label = file.replace('.npy', '')  # Use filename as label
    plt.plot(data, label=label)

plt.title('SVM Server Losses (10 epochs, 100 updates)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='upper right', fontsize='small')
plt.grid()
plt.show()
