import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from bayes_opt import BayesianOptimization
from scipy.stats import chi2
import matplotlib.pyplot as plt 

# Load data from Excel sheet
data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# Total number of rows in the dataset
total_rows = len(data)

# Define the number of rows in each batch
batch_size = 50

# List of columns for ground truth and estimated states
gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
              'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
              'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
              'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
              'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# Extract process noise covariance matrices for each batch
process_noise_covariance_matrices = [
    [
        np.array(data.iloc[start_idx + i, 40:40 + 15*15].values).reshape((15, 15))
        for i in range(min(batch_size, total_rows - start_idx))
    ]
    for start_idx in range(0, total_rows, batch_size)
]

# Define weights for each metric (adjust based on importance)
weight_NEES = 0.5
weight_RPE = 0.2
weight_RRE = 0.2
weight_trace = 0.1

# Define the target NEES value
target_NEES = 24.996

# Function to calculate the objective score for a given set of diagonal elements
def calculate_objective(**kwargs):
    diagonal_elements = np.array([kwargs[f'd{i}'] for i in range(15)])
    
    average_NEES_values = []
    log_NEES_values = []
    average_position_errors = []
    average_orientation_errors = []
    average_trace_covariances = []
    
    for start_idx in range(0, total_rows, batch_size):
        actual_batch_size = min(batch_size, total_rows - start_idx)
        x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        e_x_batch = x_gt_batch.values - x_est_batch.values

        # Adjust the diagonal elements for the process noise covariance matrices
        adjusted_cov_matrices = [
            process_noise_covariance_matrices[start_idx // batch_size][i] + np.diag(diagonal_elements)
            for i in range(actual_batch_size)
        ]
        
        NEES_values = []
        position_errors = []
        orientation_errors = []
        trace_covariances = []

        for i in range(actual_batch_size):
            try:
                reg_matrix = adjusted_cov_matrices[i] + np.eye(15) * 1e-6
                inv_cov_matrix = np.linalg.inv(reg_matrix)
                e_x_i = e_x_batch[i][:15]  # Only use the first 15 elements
                NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
            except np.linalg.LinAlgError:
                NEES = np.inf
            NEES_values.append(NEES)
            
            g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
            s_i = x_est_batch.iloc[i][['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']].values
            position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
            position_errors.append(position_error)
            
            g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
            s_quat = x_est_batch.iloc[i][['ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W']].values
            
            g_euler = R.from_quat(g_quat).as_euler('xyz')
            s_euler = R.from_quat(s_quat).as_euler('xyz')
            
            orientation_error = np.sum(np.abs(np.arccos(np.clip(np.cos(g_euler - s_euler), -1.0, 1.0))))
            orientation_errors.append(orientation_error)
            
            trace_covariance = np.trace(adjusted_cov_matrices[i])
            trace_covariances.append(trace_covariance)

        avg_NEES = np.nanmean(NEES_values)
        avg_position_error = np.sqrt(np.nanmean(position_errors))
        avg_orientation_error = np.nanmean(orientation_errors)
        avg_trace_covariance = np.nanmean(trace_covariances)
        
        # Calculate log_NEES
        log_NEES = abs(np.log(avg_NEES / target_NEES))
        
        average_NEES_values.append(avg_NEES)
        log_NEES_values.append(log_NEES)
        average_position_errors.append(avg_position_error)
        average_orientation_errors.append(avg_orientation_error)
        average_trace_covariances.append(avg_trace_covariance)
    
    overall_log_NEES = np.nanmean(log_NEES_values)
    overall_position_error = np.nanmean(average_position_errors)
    overall_orientation_error = np.nanmean(average_orientation_errors)
    overall_trace_covariance = np.nanmean(average_trace_covariances)
    
    objective_score = (
        weight_NEES * overall_log_NEES +
        weight_RPE * overall_position_error +
        weight_RRE * overall_orientation_error +
        weight_trace * overall_trace_covariance
    )
    
    return -objective_score  # We minimize the negative score because BayesianOptimization maximizes the target function

# Define the bounds for each diagonal element
pbounds = {f'd{i}': (0.00000001, 0.01) for i in range(15)}

# Initialize Bayesian optimizer
optimizer = BayesianOptimization(
    f=calculate_objective,
    pbounds=pbounds,
    random_state=1,
)

# Perform optimization with more iterations
optimizer.maximize(
    init_points=10,
    n_iter=10,
)

# Extract the optimal diagonal elements
optimal_diagonal_elements = [optimizer.max['params'][f'd{i}'] for i in range(15)]
print(f"Optimal diagonal elements: {optimal_diagonal_elements}")

# Function to perform chi-square test for NEES
def chi_square_test(nees_values, dof, significance_level):
    # Calculate critical value from chi-square distribution
    critical_value = chi2.ppf(1 - significance_level, dof)

    # Check if NEES values are consistent
    consistent = all(nees <= critical_value for nees in nees_values)

    return consistent, critical_value

# Calculate NEES with optimized diagonal elements
adjusted_cov_matrices_optimized = [
    process_noise_covariance_matrices[start_idx // batch_size][i] + np.diag(optimal_diagonal_elements)
    for start_idx in range(0, total_rows, batch_size)
    for i in range(min(batch_size, total_rows - start_idx))
]

nees_values = []
for i in range(len(data)):
    x_gt = data[gt_columns].iloc[i].values
    x_est = data[et_columns].iloc[i].values
    e_x = x_gt - x_est
    e_x_i = e_x[:15]  # Only use the first 15 elements
    inv_cov_matrix = np.linalg.inv(adjusted_cov_matrices_optimized[i] + np.eye(15) * 1e-6)
    NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
    nees_values.append(NEES)

# Perform chi-square test
dof = 15
significance_level = 0.05
is_consistent, critical_value = chi_square_test(nees_values, dof, significance_level)
print(f"Are optimized process noise covariance matrices consistent with NEES target (chi-square test)? {is_consistent}")
print(f"Critical value (chi-square distribution with {dof} dof and {significance_level} significance): {critical_value}")

# Plot NEES values
plt.figure(figsize=(10, 6))
plt.plot(nees_values, label='NEES Values')
plt.axhline(y=critical_value, color='r', linestyle='--', label='Critical Value')
plt.title('NEES Values vs Critical Value')
plt.xlabel('Sample Index')
plt.ylabel('NEES Value')
plt.legend()
plt.grid(True)
plt.show()

# Save the NEES values and results to an Excel file
results_df = pd.DataFrame({
    'Sample_Index': np.arange(len(nees_values)),
    'NEES_Values': nees_values
})
results_df.to_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/results_with_nees_graph.xlsx", index=False)

optimal_diagonals_df = pd.DataFrame({'Optimal_Diagonal': optimal_diagonal_elements})
optimal_diagonals_df.to_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/results_with_new_objective10.xlsx", index=False)




# import pandas as pd
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from bayes_opt import BayesianOptimization

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Define weights for each metric (you can adjust these weights based on importance)
# weight_NEES = 0.3
# weight_RPE = 0.3
# weight_RRE = 0.3
# weight_trace = 0.1  # Lower weight as trace value can vary significantly

# # Define the target NEES value
# target_NEES = 24.996

# # Function to calculate the objective score for a given set of diagonal elements
# def calculate_objective(**kwargs):
#     diagonal_elements = np.array([kwargs[f'd{i}'] for i in range(15)])
    
#     average_NEES_values = []
#     average_position_errors = []
#     average_orientation_errors = []
#     average_trace_covariances = []
    
#     for start_idx in range(0, total_rows, batch_size):
#         actual_batch_size = min(batch_size, total_rows - start_idx)
#         x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         e_x_batch = x_gt_batch.values - x_est_batch.values

#         process_noise_covariance_matrices = [np.diag(diagonal_elements) for _ in range(actual_batch_size)]
        
#         NEES_values = []
#         position_errors = []
#         orientation_errors = []
#         trace_covariances = []

#         for i in range(actual_batch_size):
#             try:
#                 reg_matrix = process_noise_covariance_matrices[i] + np.eye(15) * 1e-6
#                 inv_cov_matrix = np.linalg.inv(reg_matrix)
#                 e_x_i = e_x_batch[i][:15]
#                 NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
#             except np.linalg.LinAlgError:
#                 NEES = np.inf
#             NEES_values.append(NEES)
            
#             g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#             s_i = x_est_batch.iloc[i][['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']].values
#             position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
#             position_errors.append(position_error)
            
#             g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#             s_quat = x_est_batch.iloc[i][['ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W']].values
            
#             g_euler = R.from_quat(g_quat).as_euler('xyz')
#             s_euler = R.from_quat(s_quat).as_euler('xyz')
            
#             orientation_error = np.sum(np.abs(np.arccos(np.clip(np.cos(g_euler - s_euler), -1.0, 1.0))))
#             orientation_errors.append(orientation_error)
            
#             trace_covariance = np.trace(process_noise_covariance_matrices[i])
#             trace_covariances.append(trace_covariance)

#         avg_NEES = np.nanmean(NEES_values)
#         avg_position_error = np.sqrt(np.nanmean(position_errors))
#         avg_orientation_error = np.nanmean(orientation_errors)
#         avg_trace_covariance = np.nanmean(trace_covariances)
        
#         average_NEES_values.append(avg_NEES)
#         average_position_errors.append(avg_position_error)
#         average_orientation_errors.append(avg_orientation_error)
#         average_trace_covariances.append(avg_trace_covariance)
    
#     overall_NEES = np.nanmean(average_NEES_values)
#     overall_position_error = np.nanmean(average_position_errors)
#     overall_orientation_error = np.nanmean(average_orientation_errors)
#     overall_trace_covariance = np.nanmean(average_trace_covariances)
    
#     objective_score = (
#         weight_NEES * overall_NEES +
#         weight_RPE * overall_position_error +
#         weight_RRE * overall_orientation_error +
#         weight_trace * overall_trace_covariance
#     )
    
#     return -objective_score  # We minimize the negative score because BayesianOptimization maximizes the target function

# # Define the bounds for each diagonal element
# pbounds = {f'd{i}': (0.00000001, 0.09) for i in range(15)}

# # Initialize Bayesian optimizer
# optimizer = BayesianOptimization(
#     f=calculate_objective,
#     pbounds=pbounds,
#     random_state=1,
# )

# # Perform optimization
# optimizer.maximize(
#     init_points=10,
#     n_iter=30,
# )

# # Extract the optimal diagonal elements
# optimal_diagonal_elements = [optimizer.max['params'][f'd{i}'] for i in range(15)]
# print(f"Optimal diagonal elements: {optimal_diagonal_elements}")




# import pandas as pd
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from scipy.optimize import minimize
# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Define weights for each metric (you can adjust these weights based on importance)
# weight_NEES = 1.0
# weight_RPE = 0.5
# weight_RRE = 0.5
# weight_trace = 0.1  # Lower weight as trace value can vary significantly

# # Initial process noise covariance
# initial_covariance = np.eye(15)

# # Function to compute NEES, RPE, RRE, and trace for a given covariance matrix
# def compute_metrics(cov_matrix):
#     average_NEES_values = []
#     average_position_errors = []
#     average_orientation_errors = []
#     average_trace_covariances = []
    
#     # Loop over the data in steps of batch_size
#     for start_idx in range(0, total_rows, batch_size):
#         actual_batch_size = min(batch_size, total_rows - start_idx)
        
#         x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch.columns = gt_columns
#         e_x_batch = x_gt_batch - x_est_batch
        
#         NEES_values = []
#         position_errors = []
#         orientation_errors = []
#         trace_covariances = []
        
#         for i in range(actual_batch_size):
#             try:
#                 inv_cov_matrix = np.linalg.inv(cov_matrix)
#                 e_x_i = e_x_batch.iloc[i].values[:15]
#                 NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
#             except np.linalg.LinAlgError:
#                 NEES = np.inf
#             NEES_values.append(NEES)
            
#             g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#             s_i = x_est_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#             position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
#             position_errors.append(position_error)
            
#             g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#             s_quat = x_est_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#             g_euler = R.from_quat(g_quat).as_euler('xyz')
#             s_euler = R.from_quat(s_quat).as_euler('xyz')
#             orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
#             orientation_errors.append(orientation_error)
            
#             trace_covariance = np.trace(cov_matrix)
#             trace_covariances.append(trace_covariance)
        
#         avg_NEES = np.mean(NEES_values)
#         avg_position_error = np.sqrt(np.mean(position_errors))
#         avg_orientation_error = np.mean(orientation_errors)
#         avg_trace_covariance = np.mean(trace_covariances)
        
#         average_NEES_values.append(avg_NEES)
#         average_position_errors.append(avg_position_error)
#         average_orientation_errors.append(avg_orientation_error)
#         average_trace_covariances.append(avg_trace_covariance)
    
#     return (
#         np.mean(average_NEES_values),
#         np.mean(average_position_errors),
#         np.mean(average_orientation_errors),
#         np.mean(average_trace_covariances)
#     )

# # Objective function
# def objective_function(cov_matrix_flat):
#     cov_matrix = cov_matrix_flat.reshape((15, 15))
    
#     # Compute metrics
#     avg_NEES, avg_position_error, avg_orientation_error, avg_trace_covariance = compute_metrics(cov_matrix)
    
#     # Compute objective score
#     objective_score = (
#         weight_NEES * avg_NEES +
#         weight_RPE * avg_position_error +
#         weight_RRE * avg_orientation_error +
#         weight_trace * avg_trace_covariance
#     )
    
#     return objective_score

# # Initialize starting values for process noise covariance matrices (flattened)
# initial_covariance_flat = initial_covariance.flatten()

# # Optimization
# result = minimize(objective_function, initial_covariance_flat, method='L-BFGS-B')

# # Extract optimized covariance matrix
# optimized_covariance_matrix = result.x.reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_covariance_matrix)

# # Compute final metrics
# final_NEES, final_RPE, final_RRE, final_trace = compute_metrics(optimized_covariance_matrix)

# print(f"Final NEES: {final_NEES}")
# print(f"Final RPE: {final_RPE}")
# print(f"Final RRE: {final_RRE}")
# print(f"Final Trace: {final_trace}")

####WITH OBJECTIVE FUNCTION
# import pandas as pd
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Initialize lists to store average NEES values, relative position errors, relative orientation errors, and traces of process noise covariance matrices for each batch
# average_NEES_values = []
# average_position_errors = []
# average_orientation_errors = []
# average_trace_covariances = []

# # Define the target NEES value
# target_NEES = 24.996

# # Define weights for each metric (you can adjust these weights based on importance)
# weight_NEES = 1.0
# weight_RPE = 0.5
# weight_RRE = 0.5
# weight_trace = 0.1  # Lower weight as trace value can vary significantly

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Determine the actual batch size (may be less than batch_size for the final batch)
#     actual_batch_size = min(batch_size, total_rows - start_idx)
    
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
    
#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [
#         np.array(data.iloc[start_idx + i, 40:40 + 15*15].values).reshape((15, 15))
#         for i in range(actual_batch_size)
#     ]
    
#     # Calculate NEES, position error, orientation error, and trace of covariance matrix for each sample in the batch
#     NEES_values = []
#     position_errors = []
#     orientation_errors = []
#     trace_covariances = []
    
#     for i in range(actual_batch_size):
#         try:
#             inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             e_x_i = e_x_batch.iloc[i].values[:15]  # Ensure the error vector is 15-dimensional
#             NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
#         except np.linalg.LinAlgError:
#             NEES = np.inf  # If the matrix is singular, assign a very high NEES
#         NEES_values.append(NEES)
        
#         # Calculate relative position error (translation)
#         g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#         s_i = x_est_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#         position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
#         position_errors.append(position_error)
        
#         # Calculate relative orientation error (rotation)
#         g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#         s_quat = x_est_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
        
#         # Convert quaternions to Euler angles
#         g_euler = R.from_quat(g_quat).as_euler('xyz')
#         s_euler = R.from_quat(s_quat).as_euler('xyz')
        
#         orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
#         orientation_errors.append(orientation_error)
        
#         # Calculate trace of process noise covariance matrix
#         trace_covariance = np.trace(process_noise_covariance_matrices[i])
#         trace_covariances.append(trace_covariance)
    
#     # Calculate averages for the batch
#     avg_NEES = np.mean(NEES_values)
#     avg_position_error = np.sqrt(np.mean(position_errors))
#     avg_orientation_error = np.mean(orientation_errors)
#     avg_trace_covariance = np.mean(trace_covariances)
    
#     # Weighted objective function
#     objective_score = (
#         weight_NEES * avg_NEES +
#         weight_RPE * avg_position_error +
#         weight_RRE * avg_orientation_error +
#         weight_trace * avg_trace_covariance
#     )
    
#     # Append the averages to the respective lists
#     average_NEES_values.append(avg_NEES)
#     average_position_errors.append(avg_position_error)
#     average_orientation_errors.append(avg_orientation_error)
#     average_trace_covariances.append(avg_trace_covariance)
    
#     print(f"Batch {start_idx // batch_size}: NEES={avg_NEES}, RPE={avg_position_error}, RRE={avg_orientation_error}, Trace={avg_trace_covariance}, Objective={objective_score}")

# # Print final objective function value
# final_objective = np.mean(objective_score)
# print(f"Final objective function value: {final_objective}")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Initialize lists to store average NEES values, relative position errors, relative orientation errors, and traces of process noise covariance matrices for each batch
# average_NEES_values = []
# average_position_errors = []
# average_orientation_errors = []
# average_trace_covariances = []

# # Define the target NEES value
# target_NEES = 24.996

# # Small epsilon value to prevent division by zero
# epsilon = 1e-6

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Determine the actual batch size (may be less than batch_size for the final batch)
#     actual_batch_size = min(batch_size, total_rows - start_idx)
    
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
    
#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [
#         np.array(data.iloc[start_idx + i, 40:40 + 15*15].values).reshape((15, 15))
#         for i in range(actual_batch_size)
#     ]
    
#     # Calculate NEES, position error, orientation error, and trace of covariance matrix for each sample in the batch
#     NEES_values = []
#     position_errors = []
#     orientation_errors = []
#     trace_covariances = []
    
#     for i in range(actual_batch_size):
#         try:
#             inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             e_x_i = e_x_batch.iloc[i].values[:15]  # Ensure the error vector is 15-dimensional
#             NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
#         except np.linalg.LinAlgError:
#             NEES = np.inf  # If the matrix is singular, assign a very high NEES
#         NEES_values.append(NEES)
        
#         # Calculate relative position error (translation)
#         g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#         s_i = x_est_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#         position_error = np.linalg.norm((g_i - s_i) / (g_i + epsilon)) ** 2
#         position_errors.append(position_error)
        
#         # Calculate relative orientation error (rotation)
#         g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#         s_quat = x_est_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
        
#         # Convert quaternions to Euler angles
#         g_euler = R.from_quat(g_quat).as_euler('xyz')
#         s_euler = R.from_quat(s_quat).as_euler('xyz')
        
#         orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
#         orientation_errors.append(orientation_error)
        
#         # Calculate trace of process noise covariance matrix
#         trace_covariance = np.trace(process_noise_covariance_matrices[i])
#         trace_covariances.append(trace_covariance)
    
#     # Calculate averages for the batch
#     avg_NEES = np.mean(NEES_values)
#     avg_position_error = np.sqrt(np.mean(position_errors))
#     avg_orientation_error = np.mean(orientation_errors)
#     avg_trace_covariance = np.mean(trace_covariances)
    
#     # Append the averages to the respective lists
#     average_NEES_values.append(avg_NEES)
#     average_position_errors.append(avg_position_error)
#     average_orientation_errors.append(avg_orientation_error)
#     average_trace_covariances.append(avg_trace_covariance)

# # Plot average NEES for each batch
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch')
# plt.legend()
# plt.grid(True)

# # Plot average relative position error for each batch
# plt.subplot(3, 1, 2)
# plt.plot(range(len(average_position_errors)), average_position_errors, marker='s', linestyle='-', color='g', label='Average Position Error')
# plt.xlabel('Batch Number')
# plt.ylabel('Position Error')
# plt.title('Average Position Error for Each Batch')
# plt.legend()
# plt.grid(True)

# # Plot average relative orientation error for each batch
# plt.subplot(3, 1, 3)
# plt.plot(range(len(average_orientation_errors)), average_orientation_errors, marker='^', linestyle='-', color='m', label='Average Orientation Error')
# plt.xlabel('Batch Number')
# plt.ylabel('Orientation Error')
# plt.title('Average Orientation Error for Each Batch')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Plot average trace of process noise covariance for each batch
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(average_trace_covariances)), average_trace_covariances, marker='d', linestyle='-', color='c', label='Average Trace of Covariance')
# plt.xlabel('Batch Number')
# plt.ylabel('Trace of Covariance')
# plt.title('Average Trace of Process Noise Covariance for Each Batch')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
