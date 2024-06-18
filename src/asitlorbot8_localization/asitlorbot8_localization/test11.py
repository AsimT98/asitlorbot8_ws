import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

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

# Define weights for each metric (you can adjust these weights based on importance)
weight_NEES = 0.3
weight_RPE = 0.3
weight_RRE = 0.3
weight_trace = 0.1  # Lower weight as trace value can vary significantly

# Define the target NEES value
target_NEES = 24.996

# Objective function to minimize
def objective(diagonal_elements):
    average_NEES_values = []
    average_position_errors = []
    average_orientation_errors = []
    average_trace_covariances = []
    
    for start_idx in range(0, total_rows, batch_size):
        actual_batch_size = min(batch_size, total_rows - start_idx)
        x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        e_x_batch = x_gt_batch.values - x_est_batch.values

        process_noise_covariance_matrices = [np.diag(diagonal_elements) for _ in range(actual_batch_size)]
        
        NEES_values = []
        position_errors = []
        orientation_errors = []
        trace_covariances = []

        for i in range(actual_batch_size):
            try:
                reg_matrix = process_noise_covariance_matrices[i] + np.eye(15) * 1e-6
                inv_cov_matrix = np.linalg.inv(reg_matrix)
                e_x_i = e_x_batch[i][:15]
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
            
            trace_covariance = np.trace(process_noise_covariance_matrices[i])
            trace_covariances.append(trace_covariance)
        
        avg_NEES = np.nanmean(NEES_values)
        avg_position_error = np.sqrt(np.nanmean(position_errors))
        avg_orientation_error = np.nanmean(orientation_errors)
        avg_trace_covariance = np.nanmean(trace_covariances)
        
        average_NEES_values.append(avg_NEES)
        average_position_errors.append(avg_position_error)
        average_orientation_errors.append(avg_orientation_error)
        average_trace_covariances.append(avg_trace_covariance)
    
    overall_NEES = np.nanmean(average_NEES_values)
    overall_position_error = np.nanmean(average_position_errors)
    overall_orientation_error = np.nanmean(average_orientation_errors)
    overall_trace_covariance = np.nanmean(average_trace_covariances)
    
    objective_score = (
        weight_NEES * overall_NEES +
        weight_RPE * overall_position_error +
        weight_RRE * overall_orientation_error +
        weight_trace * overall_trace_covariance
    )
    
    return objective_score

# Initial guess for diagonal elements (15-dimensional)
initial_guess = np.ones(15)

# Perform optimization
result = minimize(objective, initial_guess, method='BFGS', options={'disp': True})

# Optimal diagonal elements
optimal_diagonal_elements = result.x
print(f"Optimal diagonal elements: {optimal_diagonal_elements}")


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
    
#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch.values - x_est_batch.values

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
#             # Add small value to diagonal for regularization to avoid singular matrices
#             reg_matrix = process_noise_covariance_matrices[i] + np.eye(15) * 1e-6
#             inv_cov_matrix = np.linalg.inv(reg_matrix)
#             e_x_i = e_x_batch[i][:15]  # Ensure the error vector is 15-dimensional
#             NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
#         except np.linalg.LinAlgError:
#             NEES = np.inf  # If the matrix is singular, assign a very high NEES
#         NEES_values.append(NEES)
        
#         # Calculate relative position error (translation)
#         g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
#         s_i = x_est_batch.iloc[i][['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']].values
#         position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
#         position_errors.append(position_error)
        
#         # Calculate relative orientation error (rotation)
#         g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
#         s_quat = x_est_batch.iloc[i][['ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W']].values
        
#         # Convert quaternions to Euler angles
#         g_euler = R.from_quat(g_quat).as_euler('xyz')
#         s_euler = R.from_quat(s_quat).as_euler('xyz')
        
#         orientation_error = np.sum(np.abs(np.arccos(np.clip(np.cos(g_euler - s_euler), -1.0, 1.0))))
#         orientation_errors.append(orientation_error)
        
#         # Calculate trace of process noise covariance matrix
#         trace_covariance = np.trace(process_noise_covariance_matrices[i])
#         trace_covariances.append(trace_covariance)
    
#     # Calculate averages for the batch
#     avg_NEES = np.nanmean(NEES_values)
#     avg_position_error = np.sqrt(np.nanmean(position_errors))
#     avg_orientation_error = np.nanmean(orientation_errors)
#     avg_trace_covariance = np.nanmean(trace_covariances)
    
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
# final_objective = np.nanmean(objective_score)
# print(f"Final objective function value: {final_objective}")
