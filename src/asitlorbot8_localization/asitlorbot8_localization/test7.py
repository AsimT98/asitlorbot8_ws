# !/usr/bin/env python3
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from tf_transformations import euler_from_quaternion
import matplotlib.pyplot as plt

# Load data from Excel sheet
data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

def objective_function(process_noise_covariance_1d):
    # Reshape process_noise_covariance_1d to a 15x15 matrix
    process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))

    objective_value_nees = 0
    objective_value_rpe_translation = 0
    objective_value_rpe_rotation = 0

    # Loop over data in batches of 50
    num_batches = len(data) // 50
    num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

    for i in range(num_batches):
        start_idx = i * 50
        end_idx = (i + 1) * 50

        # Extract ground truth state variables for the current batch
        x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
                           'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
                           'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

        # Extract estimated state variables for the current batch
        x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
                            'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
                            'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

        # Compute NEES for all variables
        e_x_batch = x_gt_batch - x_est_batch
        NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
        avg_NEES_batch = np.mean(NEES_batch)
        objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
        # Compute RPE for translation
        translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
        avg_rpe_translation = np.mean(translation_errors)
        objective_value_rpe_translation += avg_rpe_translation

        # Compute RPE for rotation using euler angles
        for j in range(50):
            # Extract quaternion components for ground truth and estimated
            gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
            est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]
            
            # Convert quaternions to euler angles
            gt_euler = euler_from_quaternion(gt_quat)
            est_euler = euler_from_quaternion(est_quat)
            
            # Compute angle difference between the rotations
            angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
            objective_value_rpe_rotation += angle_diff
            num_elements_rpe_rotation += 1
    
    avg_rpe_translation /= num_batches
    avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
    objective_value_nees /= num_batches
    
    # Weightage for trace term and NEES
    weight_trace = 0.3
    weight_nees = 0.1
    
    # Calculate trace of the process noise covariance matrix
    trace = np.trace(process_noise_covariance_matrix)
    
    # Calculate objective value for trace term
    objective_value_trace = weight_trace * trace
    
    # Combine NEES, trace, and RPE with weights
    objective_value = (0.3 * avg_rpe_translation + 
                       0.3 * avg_rpe_rotation + 
                       0.4 * objective_value_nees + 
                       objective_value_trace)
    
    print("Objective Value RPE Translation: ", avg_rpe_translation)
    print("Objective Value RPE Rotation: ", avg_rpe_rotation)
    print("Objective Value NEES: ", objective_value_nees)
    print("Objective Value Trace: ", objective_value_trace)
    print("Combined Objective Value: ", objective_value)
    
    return objective_value

# Define the bounds for the process noise covariance matrix
bounds = [Real(0.001, 1)] * 225  # Assuming a 15x15 matrix
bounds[80] = (0.00002, 0.9)
bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
bounds[192] = (0.00001, 0.1)
# Store NEES values for plotting
nees_values = []

# Modified objective function to capture NEES values
def modified_objective_function(x):
    obj_value = objective_function(x)
    nees_values.append(obj_value)
    return obj_value

# Perform optimization using Bayesian Optimization
res = gp_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

# Extract the optimized process noise covariance matrix
optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

print("Optimized Process Noise Covariance Matrix:")
print(optimized_process_noise_covariance_matrix)

# Extract the values of the objective function at each iteration
objective_values = res.func_vals

# Plot the convergence of the objective function
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, marker='o', linestyle='-')
plt.title('Convergence of Objective Function')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.grid(True)
plt.show()

# Plot the NEES values separately
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(nees_values) + 1), nees_values, marker='o', linestyle='-')
plt.axhline(y=9.488, color='r', linestyle='--', label='Target NEES (9.488)')
plt.title('Convergence of NEES')
plt.xlabel('Iteration')
plt.ylabel('NEES Value')
plt.grid(True)
plt.show()



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
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Define the target NEES value
# target_NEES = 24.996

# # Function to prepare batches from the dataset
# def prepare_batches(data, batch_size):
#     all_e_x_batches = []
#     total_rows = len(data)
#     for start_idx in range(0, total_rows, batch_size):
#         actual_batch_size = min(batch_size, total_rows - start_idx)
#         x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch.columns = gt_columns
#         e_x_batch = x_gt_batch - x_est_batch
#         e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]
#         all_e_x_batches.append(e_x_batch)
#     all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)
#     return all_e_x_data

# # Split the data into training and testing sets (80% training, 20% testing)
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Prepare training and testing batches
# train_e_x_data = prepare_batches(train_data, batch_size)
# test_e_x_data = prepare_batches(test_data, batch_size)

# # Define dimension names for the diagonal elements
# dimension_names = [f'covariance_{i}' for i in range(15)]

# # Define the bounds for the diagonal elements
# bounds = [Real(0.00000001, 0.05, name=name) for name in dimension_names]

# # Define the objective function for Bayesian optimization
# @use_named_args(bounds)
# def objective_function(**params):
#     diagonal_values = [params[name] for name in dimension_names]
#     process_noise_covariance_matrix = np.diag(diagonal_values)
#     try:
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#     except np.linalg.LinAlgError:
#         return np.inf  # Return a high value if the matrix is singular

#     NEES_values = []
#     for i in range(len(train_e_x_data)):
#         NEES = train_e_x_data.iloc[i].values @ inv_process_noise_covariance_matrix @ train_e_x_data.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
    
#     avg_translation_rmse = np.sqrt(np.mean(train_e_x_data[['error_Pos_X', 'error_Pos_Y', 'error_Pos_Z']]**2))
#     avg_rotation_rmse = np.sqrt(np.mean(train_e_x_data[['error_Roll', 'error_Pitch', 'error_Yaw']]**2))
    
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.3 * avg_NEES + 0.1 * trace + 0.3 * avg_translation_rmse + 0.3 * avg_rotation_rmse + penalty

# # Perform Bayesian optimization on training data
# res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=20, n_initial_points=20, random_state=42)

# # Get the optimized diagonal elements from the best solution found by Bayesian optimization
# optimized_diagonal_values = res_gp.x
# optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)

# # Print the optimized process noise covariance matrix
# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Function to evaluate NEES and RMSE
# def evaluate_nees_and_rmse(e_x_data, inv_cov_matrix):
#     NEES_values = []
#     for i in range(len(e_x_data)):
#         NEES = e_x_data.iloc[i].values @ inv_cov_matrix @ e_x_data.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     avg_translation_rmse = np.sqrt(np.mean(e_x_data[['error_Pos_X', 'error_Pos_Y', 'error_Pos_Z']]**2))
#     avg_rotation_rmse = np.sqrt(np.mean(e_x_data[['error_Roll', 'error_Pitch', 'error_Yaw']]**2))
#     return avg_NEES, avg_translation_rmse, avg_rotation_rmse

# # Evaluate on test data
# inv_optimized_covariance_matrix = np.linalg.inv(optimized_process_noise_covariance_matrix)
# test_avg_NEES, test_avg_translation_rmse, test_avg_rotation_rmse = evaluate_nees_and_rmse(test_e_x_data, inv_optimized_covariance_matrix)

# print(f"Test Average NEES: {test_avg_NEES}")
# print(f"Test Average Translation RMSE: {test_avg_translation_rmse}")
# print(f"Test Average Rotation RMSE: {test_avg_rotation_rmse}")

# # Cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# fold_results = []

# for train_index, test_index in kf.split(data):
#     train_data = data.iloc[train_index]
#     test_data = data.iloc[test_index]
    
#     train_e_x_data = prepare_batches(train_data, batch_size)
#     test_e_x_data = prepare_batches(test_data, batch_size)
    
#     # Perform Bayesian optimization on training data
#     res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=20, n_initial_points=20, random_state=42)
#     optimized_diagonal_values = res_gp.x
#     optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)
#     inv_optimized_covariance_matrix = np.linalg.inv(optimized_process_noise_covariance_matrix)
    
#     # Evaluate on test data
#     test_avg_NEES, test_avg_translation_rmse, test_avg_rotation_rmse = evaluate_nees_and_rmse(test_e_x_data, inv_optimized_covariance_matrix)
    
#     fold_results.append((test_avg_NEES, test_avg_translation_rmse, test_avg_rotation_rmse))

# # Calculate the average results across all folds
# avg_results = np.mean(fold_results, axis=0)
# print(f"Cross-Validation Average NEES: {avg_results[0]}")
# print(f"Cross-Validation Average Translation RMSE: {avg_results[1]}")
# print(f"Cross-Validation Average Rotation RMSE: {avg_results[2]}")

# # Plot average NEES for each batch in the test data
# inv_optimized_covariance_matrix = np.linalg.inv(optimized_process_noise_covariance_matrix)
# average_NEES_values = []
# for start_idx in range(0, len(test_e_x_data), batch_size):
#     actual_batch_size = min(batch_size, len(test_e_x_data) - start_idx)
#     e_x_batch = test_e_x_data[start_idx:start_idx + actual_batch_size]

#     NEES_values = []
#     for i in range(actual_batch_size):
#         NEES = e_x_batch.iloc[i].values @ inv_optimized_covariance_matrix @ e_x_batch.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     average_NEES_values.append(avg_NEES)

# plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch in Test Data')
# plt.legend()
# plt.grid()
# plt.show()
