# Asynchronous Federated Learning (AFL)

## Overview
Federated Learning (FL) has emerged as a powerful paradigm for decentralized machine learning, enabling collaborative model training across diverse clients without sharing raw data. However, traditional FL approaches often face limitations in scalability and efficiency due to their reliance on synchronous client updates, which can result in significant delays and increased communication overhead, particularly in heterogeneous and dynamic environments. 

To address these challenges, this project proposes an Asynchronous Federated Learning (AFL) algorithm, which allows clients to update the global model independently and asynchronously.

## Key Contributions

- **Algorithm Design**: 
  - A novel AFL algorithm to address inefficiencies in traditional FL methods, such as global synchronization delays and client drift.
  - Improved scalability, robustness, and efficiency for real-world applications with dynamic and heterogeneous client populations.

- **Convergence Analysis**:
  - Comprehensive analysis of AFL in the presence of client delays and model staleness.
  - Utilization of martingale difference sequence theory and variance bounds to ensure robust convergence.
  - Assumption of \(\mu\)-strongly convex local objective functions to establish gradient variance bounds under random client sampling.
  - Derivation of a recursion formula quantifying the impact of client delays on convergence.

- **Practical Demonstration**:
  - Training a decentralized Long Short-Term Memory (LSTM)-based deep learning model on the CMIP6 climate dataset.
  - Handling non-IID and geographically distributed data effectively.

## Features

- **Asynchronous Updates**: Clients update the global model independently without waiting for synchronization.
- **Robust Convergence**: Designed to address client delays and model staleness.
- **Scalability**: Efficiently handles large-scale and privacy-preserving applications in resource-constrained environments.

## Getting Started

### Prerequisites
- Python 3.8+
- Key Python libraries: TensorFlow/PyTorch, NumPy, pandas, and Matplotlib
- Climate Model Intercomparison Project (CMIP6) dataset

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/asynchronous-federated-learning.git
   cd asynchronous-federated-learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage For Strongly Convex Objective Function

1. Run the training script for \afl~algorithm:
   ```bash
   python async_fdl_logistic_regression_convex_objective_function_dynamic_lr_12_device_fixed_energy_proxy.py
   python async_fdl_logistic_regression_convex_objective_function_dynamic_lr_12_device_fixed.py
   python async_fdl_svm_binaryclassification_convex_objective_function_dynamic_lr_12.py
   ```
2. Evaluate the trained model on test data.The simulation results will be saved in `federated_learning_results` directory and then one can run
   ```bash
   regression_training_loss_plots.py
   regression_wall_clock_energy_proxy.py
   synchronous_training_loss_plots.py
   synchronous_svm_training_loss_plots.py
   training_loss_plots.py
   ```

### Usage For General Objective Functions

1. Prepare the CMIP6 climate dataset and partition it for federated learning.
2. Run the training script for \afl~algorithm:
   ```bash
   [python training_wind_psr_lstm_fl_async_strictly_convex_random_selection.py] *this file is for extra simulation*
   ```
3. Evaluate the trained model on test data.

## Results
- **Convergence Analysis**:
  - Derived theoretical bounds for gradient variance and client delay impact.
  - Proved robust convergence for \afl~under asynchronous settings.

- **Empirical Results**:
  - Demonstrated efficient training of an LSTM-based deep learning model on non-IID, geographically distributed data.
  - Achieved significant improvements in scalability and robustness compared to traditional FL methods.

## Applications
- Large-scale distributed learning systems
- Privacy-preserving machine learning in resource-constrained environments
- Climate data analysis and modeling using CMIP6 datasets

## Citation
If you use this code or method in your research, please cite the associated paper:
```bibtex
@article{forootani2024asynchronous,
  title={Asynchronous Federated Learning: A Scalable Approach for Decentralized Machine Learning},
  author={Forootani, Ali and Iervolino, Raffaele},
  journal={arXiv preprint arXiv:2412.17723},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For further questions or collaborations, please contact:
- **Name**: Ali Forootani
- **Email**: aliforootani@ieee.org/aliforootani@gmail.com
- **Institution**: During writing this Readme `Helmholtz Center For Environmental Research - UFZ, leipzig, Germany`.

