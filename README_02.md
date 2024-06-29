
### README for Pump Trimming Project with Neural Network

#### Introduction
This project aims to optimize the performance of a pump system using neural networks. The optimization process involves tuning the hyperparameters of a neural network to minimize errors in predicting key pump characteristics based on provided training data. The main script for hyperparameter optimization is `optimize_nn_hyperparameters_routine.m`.

#### File Description: `optimize_nn_hyperparameters_routine.m`
This MATLAB script is designed to optimize the hyperparameters of a neural network using a genetic algorithm (GA). It performs the following tasks:

1. **Load Data**: The script loads the training and validation data from four `.mat` files: `filtered_QHD_table.mat`, `filtered_QDP_table.mat`, `deleted_QHD_table.mat`, and `deleted_QDP_table.mat`.

2. **Prepare Data**: The data is prepared for training by extracting relevant features and targets:
   - `QH` (Flow rate and Head)
   - `D` (Diameter)
   - `QD` (Flow rate and Diameter)
   - `P` (Power)

3. **Set Hyperparameters**:
   - Number of hidden layers: 5
   - Neuron bounds: Specifies the lower and upper bounds for the number of neurons in each hidden layer.

4. **Genetic Algorithm Options**: The script sets the options for the GA, including the fitness limit.

5. **Call Optimization Routine**: The main function `optimize_nn_hyperparameters` is called to perform the optimization, which returns the best network and its parameters.

6. **Display Results**: The best hyperparameters and training performance are displayed.

#### Key Functions

- **`optimize_nn_hyperparameters`**: This function performs the actual optimization of the neural network hyperparameters.
  - **Inputs**:
    - `input_data`: Input features for training.
    - `output_data`: Target values for training.
    - `num_hidden_layers`: Number of hidden layers in the neural network.
    - `neuron_bounds`: Bounds for the number of neurons in each hidden layer.
    - `ga_opts`: Options for the genetic algorithm.
    - `userSeed`: Seed for random number generation to ensure reproducibility.
  - **Outputs**:
    - `best_net_global`: The best neural network model found.
    - `best_params`: Best hyperparameters (number of neurons in each hidden layer).
    - `nnPerfVect`: Performance vector (training, validation, and test errors).

- **`plot_nn_vs_real_data`**: This function plots the actual vs. predicted output of the neural network for visualization.

- **`nn_fitness`**: This function defines the fitness function for the GA, which trains the neural network and calculates the mean squared error (MSE).

- **`log_nn_details`**: This function logs the details of the neural network training process, including the hyperparameters and performance metrics.

#### Usage

To run the script, ensure the following data files are available in the `./training-data/` directory:
- `filtered_QHD_table.mat`
- `filtered_QDP_table.mat`
- `deleted_QHD_table.mat`
- `deleted_QDP_table.mat`

Execute the script in MATLAB:
```matlab
optimize_nn_hyperparameters_routine
```

The script will load the data, set up the neural network, perform hyperparameter optimization using the genetic algorithm, and display the best parameters and performance metrics.

#### Conclusion
The `optimize_nn_hyperparameters_routine.m` script automates the process of tuning neural network hyperparameters for optimizing pump performance prediction. By leveraging a genetic algorithm, it efficiently searches for the best configuration, ensuring robust and accurate predictive models.





# README for `final_results_plots_tables_comparisons.m`

## Overview

The script `final_results_plots_tables_comparisons.m` is designed to train and evaluate neural networks on various datasets representing hydraulic and power characteristics. The goal is to identify the best-performing neural networks by removing specific diameters from the dataset and analyzing the performance. The script generates plots, logs, and saves results and the best neural networks.

## File Structure

```
final_results_plots_tables_comparisons.m
training-data/
    filtered_QHD_table.mat
    filtered_QDP_table.mat
    deleted_QHD_table.mat
    deleted_QDP_table.mat
final_results_plots_tables_comparisons/
    figures/
    QHD_results.csv
    QDP_results.csv
    QDH_results.csv
    logs.txt
    bestNetQHD.mat
    bestNetQDP.mat
    bestNetQDH.mat
```

## Data Files

- **filtered_QHD_table.mat**: Filtered QHD data containing `FlowRate_m3h`, `Head_m`, and `Diameter_mm`.
- **filtered_QDP_table.mat**: Filtered QDP data containing `FlowRate_m3h`, `Diameter_mm`, and `Power_kW`.
- **deleted_QHD_table.mat**: Deleted QHD data points representing best efficiency points (BEPs).
- **deleted_QDP_table.mat**: Deleted QDP data points representing best efficiency points (BEPs).

## Script Functionality

### 1. Initialization

The script initializes by clearing the workspace and loading the data files. The datasets are extracted into variables for further processing.

### 2. Create Output Directories

Output directories `final_results_plots_tables_comparisons` and `figures` are created if they do not exist.

### 3. Hyperparameters

Hyperparameters for the neural networks are defined, including network size matrices, maximum epochs, training function, and random seed.

### 4. Initialize Results Tables

Results tables `QHD_results`, `QDP_results`, and `QDH_results` are initialized with appropriate headers.

### 5. Weights for Error Calculation

Weights for different errors are defined to compute the weighted score for the neural networks.

### 6. Best Neural Networks Initialization

Structures for storing the best neural networks for QHD, QDP, and QDH are initialized.

### 7. Train on Full Dataset

Neural networks are trained on the full dataset without removing any diameters, and performance metrics are calculated and stored.

### 8. Plot and Save Results

Test data vs trained network predictions are plotted and saved as images in the `figures` directory.

### 9. Train on Different Diameters

A loop iterates through each distinct diameter in the dataset, training neural networks with one diameter removed at a time. Performance metrics are calculated, and results are stored. The best neural network for each type is updated if a better one is found.

### 10. Save Results and Logs

The results tables and logs are saved to CSV files and a text file, respectively. The best neural networks are saved as `.mat` files.

### 11. Display Results

The results and best neural networks are displayed in the MATLAB console.

## How to Use

1. **Prepare Data**: Ensure the required data files (`filtered_QHD_table.mat`, `filtered_QDP_table.mat`, `deleted_QHD_table.mat`, `deleted_QDP_table.mat`) are in the `training-data` directory.
2. **Run the Script**: Execute the script in MATLAB. The script will automatically create output directories, train neural networks, and save results.
3. **View Results**: Check the `final_results_plots_tables_comparisons` directory for the generated plots, results tables, logs, and the best neural networks.

## Functions

### train_nn

```matlab
[trainedNet, avgMSEs, trainPerformance, valPerformance, testPerformance] = train_nn(size_matrix, maxEpochs, trainFcn, input, target, seed)
```

Trains a neural network with the specified parameters and returns the trained network and performance metrics.

### compute_score

```matlab
score = compute_score(trainPerformance, valPerformance, testPerformance, mse_deleted_diameter, mse_beps, weights)
```

Computes the weighted score for the neural network based on the provided performance metrics and weights.

## Plots

- **QHD_Diameter_NaN.png**: Plot of QHD data without any diameter removed.
- **QDH_Diameter_NaN.png**: Plot of QDH data without any diameter removed.
- **QDP_Diameter_NaN.png**: Plot of QDP data without any diameter removed.
- **QHD_Diameter_<removed>.png**: Plot of QHD data with specific diameter removed.
- **QDH_Diameter_<removed>.png**: Plot of QDH data with specific diameter removed.
- **QDP_Diameter_<removed>.png**: Plot of QDP data with specific diameter removed.

## Logs

- **logs.txt**: Contains logs of the training process, including successful training messages and error messages.

## Results

- **QHD_results.csv**: Results of the QHD neural network training.
- **QDP_results.csv**: Results of the QDP neural network training.
- **QDH_results.csv**: Results of the QDH neural network training.
- **bestNetQHD.mat**: Best QHD neural network.
- **bestNetQDP.mat**: Best QDP neural network.
- **bestNetQDH.mat**: Best QDH neural network.

This README provides a comprehensive guide to understanding and using the `final_results_plots_tables_comparisons.m` script. For any issues or questions, please refer to the logs or contact the script author.




## README for `optimize_nn_hyperparameters_nested_loops.m`

### Overview
The `optimize_nn_hyperparameters_nested_loops.m` script is designed to optimize the hyperparameters of a neural network for predicting pump impeller trimming characteristics. This script uses nested loops to iterate over different diameters removed from the training data and optimizes the neural network for each diameter.

### Main Functionality
1. **Load Data**:
   - Loads training and testing data from specified `.mat` files.

2. **Preprocessing**:
   - Constructs matrices `QH`, `D`, `QH_beps`, `D_beps`, `QD`, `P`, `QD_beps`, and `P_beps` from the loaded data.

3. **Hyperparameter Optimization**:
   - Uses nested loops to remove each distinct diameter from the training data.
   - For each removed diameter, optimizes the hyperparameters of a neural network through multiple iterations.
   - Tracks and saves the performance metrics and the optimized neural network.

4. **Performance Evaluation**:
   - Computes Mean Squared Error (MSE) for the removed diameter and another test set (`QH_beps`).
   - Uses a combination of MSE metrics to adjust hyperparameter bounds and guide optimization.

5. **Visualization and Saving**:
   - Generates and saves plots for the predictions of the neural network.
   - Saves the results, including the optimized hyperparameters and performance metrics, to a CSV file and a `.mat` file.

### Key Variables
- **Data Variables**:
  - `QH`, `D`: Training data for flow rate and head, and diameters.
  - `QH_beps`, `D_beps`: Test data for flow rate and head, and diameters.
  - `QD`, `P`: Additional training data for flow rate, diameters, and power.
  - `QD_beps`, `P_beps`: Additional test data for flow rate, diameters, and power.

- **Optimization Parameters**:
  - `userSeed`: Random seed for reproducibility.
  - `mseThreshold`: Threshold for MSE to exit the loop early.
  - `distinctDiameters`: Unique diameters in the training data.
  - `weightDiameter`, `weightBeps`: Weights for combining MSE metrics.

- **Files**:
  - `resultsFilename`: CSV file to store results.
  - `structFilename`: MAT file to store results in a structured format.

### Execution Flow
1. **Initialization**:
   - Clears the workspace and loads data.
   - Initializes bounds for hyperparameters and sets up result storage.

2. **Main Loop**:
   - Iterates over each distinct diameter, removes it from the training data, and optimizes the neural network.
   - In each iteration, performs the following steps:
     - Removes the current diameter from the training data.
     - Optimizes the neural network using `optimizeNNForTrimmingPumpImpeller`.
     - Computes MSE for the removed diameter and the `QH_beps` test set.
     - Saves the performance metrics and the trained network.
     - Generates and saves plots for the neural network predictions.
     - Adjusts hyperparameter bounds based on MSE improvement.
     - Exits the loop early if MSE is below the threshold.

3. **Finalization**:
   - Saves all results to the specified CSV and MAT files.
   - Displays a message indicating the completion of the optimization process.

### Results
- The results are saved in:
  - `results_loop.csv`: Contains hyperparameter values, performance metrics, and random seeds for each iteration.
  - `results_struct.mat`: Structured format of the results, including the trained neural networks and plots.

### Usage
To run the script, ensure that the required data files are in the `training-data` directory. Execute the script in MATLAB, and it will automatically perform the optimization and save the results.

```matlab
optimize_nn_hyperparameters_nested_loops
```

### Dependencies
- MATLAB
- Neural Network Toolbox

### Note
This script is designed for use with specific data formats and structures. Ensure the data files are correctly formatted as expected by the script.

### Contact
For further information or questions, please contact the script author or maintainer.

---



### General Introduction to the Project

This project aims to optimize and validate the performance of neural networks in predicting the characteristics of pump impeller trimming. The primary focus is on leveraging data-driven approaches to predict key metrics like flow rate, head, and power for different pump diameters. The project consists of several MATLAB scripts and data files, which together facilitate the preparation, training, and optimization of neural networks for this purpose.

### Structure of the Project Files

#### 1. Data Preparation and Loading
The first step involves loading and preparing the data. This is handled in multiple files which load and process the necessary datasets for the neural network training and validation.

- **filtered_QHD_table.mat**
- **filtered_QDP_table.mat**
- **deleted_QHD_table.mat**
- **deleted_QDP_table.mat**

These files contain the filtered and deleted data for the pump characteristics. The key variables extracted from these files include flow rate, head, diameter, and power, which are used in training and validating the neural networks.

#### 2. Neural Network Hyperparameter Optimization
The core of the project is the optimization of neural network hyperparameters. This involves iteratively adjusting the network parameters to minimize the error in predictions.

- **optimize_nn_hyperparameters_nested_loops.m**

This script performs the following tasks:
1. **Loading Data**:
   - Loads the filtered and deleted datasets.
   - Constructs matrices for training and test data.
   
2. **Hyperparameter Optimization**:
   - Iterates over each distinct diameter in the dataset.
   - Removes the current diameter from the training data.
   - Optimizes the neural network for the remaining data using nested loops.
   - Computes Mean Squared Error (MSE) for validation.
   
3. **Performance Evaluation**:
   - Calculates MSE for the removed diameter and another test set.
   - Combines MSE metrics to guide optimization.
   
4. **Visualization and Saving**:
   - Generates plots for the predictions of the neural network.
   - Saves results, including optimized hyperparameters and performance metrics, to CSV and MAT files.

#### 3. Hyperparameter Optimization Function
The neural network training and optimization are encapsulated in a separate function which is called by the main optimization script.

- **optimizeNNForTrimmingPumpImpeller.m**

This function:
1. **Receives Input**:
   - Takes in training data, target data, a random seed, and bounds for the hyperparameters.
   
2. **Configures Neural Network**:
   - Sets up a neural network with specified hyperparameters.
   - Trains the network using the training data.
   
3. **Evaluates Performance**:
   - Calculates the MSE for the network's predictions.
   - Returns the optimized hyperparameters, final MSE, random seed used, the trained network, and error metrics.

### Pedagogical Explanation of the Project Structure

1. **Data Loading and Preparation**:
   - Begin by loading the necessary data from the provided `.mat` files.
   - Extract relevant variables (flow rate, head, diameter, power) to be used in neural network training and validation.

2. **Core Optimization Process**:
   - Use the `optimize_nn_hyperparameters_nested_loops.m` script as the main driver of the project.
   - This script orchestrates the removal of specific diameters, optimization of the neural network, and evaluation of its performance.
   - Each iteration involves calling the `optimizeNNForTrimmingPumpImpeller` function, which encapsulates the neural network setup, training, and evaluation.

3. **Visualization and Result Saving**:
   - After each optimization iteration, generate plots to visualize the network's predictions.
   - Save the results, including the optimized parameters and performance metrics, for further analysis.

4. **Reviewing Results**:
   - The results are stored in `results_loop.csv` and `results_struct.mat` for easy access and analysis.
   - Use these results to identify the best-performing neural network configurations and understand the impact of different diameters on the predictions.

By following this structured approach, the project ensures a systematic and efficient optimization process, leading to robust and reliable neural network models for predicting pump impeller characteristics.