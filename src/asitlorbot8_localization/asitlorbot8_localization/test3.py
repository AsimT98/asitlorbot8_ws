#!/usr/bin/env python3
import pandas as pd
import numpy as np
from skopt import dummy_minimize
from tf_transformations import euler_from_quaternion
import matplotlib.pyplot as plt

# Load data from Excel sheet
data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")


def objective_function(process_noise_covariance_matrix):
    objective_value_nees = 0
    objective_value_rpe_translation = 0
    objective_value_rpe_rotation = 0
    objective_value_trace = 0  # Initialize trace term

    # Loop over data in batches of 50
    num_batches = len(data) // 50

    for i in range(num_batches):
        # Extract process noise covariance values from DataFrame
        process_noise_covariance_1d = data.iloc[0, 40:].values
        
        # Reshape process_noise_covariance_1d to a 15x15 matrix
        process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
        # Calculate trace of the process noise covariance matrix
        trace = np.trace(process_noise_covariance_matrix)
        # print(process_noise_covariance_matrix)
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

        # Compute RPE for rotation using quaternions
        for j in range(50):
            gt_quat = np.array([x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 5], x_gt_batch[j, 6]])  # [qw, qx, qy, qz]
            est_quat = np.array([x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 5], x_est_batch[j, 6]])
            
            # Convert quaternions to euler angles
            gt_euler = euler_from_quaternion(gt_quat)
            est_euler = euler_from_quaternion(est_quat)
            
            # Compute angle difference between the rotations
            angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
            objective_value_rpe_rotation += angle_diff
        
    avg_rpe_translation = objective_value_rpe_translation / num_batches
    avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
    objective_value_nees /= num_batches
    
    # Weightage for trace term and NEES
    weight_trace = 0.3
    weight_nees = 0.1
    
    # Calculate objective value for trace term
    objective_value_trace = weight_trace * trace
    
    # Combine NEES, trace, and RPE with weights
    objective_value = (0 * avg_rpe_translation + 
                       0 * avg_rpe_rotation + 
                       0.7 * objective_value_nees +
                        objective_value_trace)  # Add trace term
    
    print("Objective Value RPE Translation: ", avg_rpe_translation)
    print("Objective Value RPE Rotation: ", avg_rpe_rotation)
    print("Objective Value NEES: ", objective_value_nees)
    print("Objective Value Trace: ", objective_value_trace)  # Print trace term
    print("Combined Objective Value: ", objective_value)
    
    return objective_value, objective_value_nees

# Define the bounds for the process noise covariance matrix
bounds = [(0.001, 1)] * 225  # Assuming a 15x15 matrix
bounds[80] = (0.00002, 0.9)
bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
bounds[192] = (0.00001, 0.1)

# Store NEES values for plotting
nees_values = []

# Modified objective function to capture NEES values
def modified_objective_function(x):
    obj_value, nees_value = objective_function(x)
    nees_values.append(nees_value)
    return obj_value

# Perform optimization
res = dummy_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

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

