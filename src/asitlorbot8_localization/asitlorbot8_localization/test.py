#!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import dummy_minimize
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# # Objective function considering only RPE for translation and rotation
# def objective_function(process_noise_covariance_matrix):
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
    
#     process_noise_covariance_matrix = np.array(process_noise_covariance_matrix).reshape((15, 15))
    
#     # Log the process noise covariance matrix for debugging
#     print("Current Process Noise Covariance Matrix:")
#     print(process_noise_covariance_matrix)
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             gt_euler = x_gt_batch[j, 3:6]
#             est_euler = x_est_batch[j, 3:6]
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
        
#     avg_rpe_translation = objective_value_rpe_translation / num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
    
#     # Combine RPE for translation and rotation with equal weights
#     objective_value = 0.5 * avg_rpe_translation + 0.5 * avg_rpe_rotation
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.9)
# bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.00001, 0.1)

# # Perform Random Search optimization
# res = dummy_minimize(objective_function, bounds, n_calls=30, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Extract the values of the objective function at each iteration
# objective_values = res.func_vals

# # Plot the convergence of the objective function
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(objective_values) + 1), objective_values, marker='o', linestyle='-')
# plt.title('Convergence of Objective Function')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.grid(True)
# plt.show()

# # Plot the NEES values separately
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(objective_values) + 1), [val for val in objective_values], marker='o', linestyle='-')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()




# import pandas as pd
# import numpy as np
# from skopt import dummy_minimize
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     objective_value_trace = 0  # Initialize trace term
    
#     process_noise_covariance_matrix = np.array(process_noise_covariance_matrix).reshape((15, 15))
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using quaternions
#         for j in range(50):
#             gt_quat = np.array([x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 5], x_gt_batch[j, 6]])  # [qw, qx, qy, qz]
#             est_quat = np.array([x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 5], x_est_batch[j, 6]])
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
        
#     avg_rpe_translation = objective_value_rpe_translation / num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
#     objective_value_nees /= num_batches
    
#     # Normalize components
#     normalized_rpe_translation = avg_rpe_translation
#     normalized_rpe_rotation = avg_rpe_rotation
#     normalized_nees = objective_value_nees / 9.488  # Target NEES value for normalization
#     normalized_trace = trace / np.sum(process_noise_covariance_matrix)

#     # Weightage for trace term and NEES
#     weight_trace = 0.5
#     weight_nees = 0.1
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * normalized_trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0 * normalized_rpe_translation + 
#                        0 * normalized_rpe_rotation + 
#                        0.5 * normalized_nees +
#                         objective_value_trace)  # Add trace term
    
#     print("Objective Value RPE Translation: ", normalized_rpe_translation)
#     print("Objective Value RPE Rotation: ", normalized_rpe_rotation)
#     print("Objective Value NEES: ", normalized_nees)
#     print("Objective Value Trace: ", objective_value_trace)  # Print trace term
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value, normalized_nees

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.9)
# bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.00001, 0.1)

# # Store NEES values for plotting
# nees_values = []

# # Modified objective function to capture NEES values
# def modified_objective_function(x):
#     obj_value, nees_value = objective_function(x)
#     nees_values.append(nees_value)
#     return obj_value

# # Perform Random Search optimization
# res = dummy_minimize(modified_objective_function, bounds, n_calls=30, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Extract the values of the objective function at each iteration
# objective_values = res.func_vals

# # Plot the convergence of the objective function
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(objective_values) + 1), objective_values, marker='o', linestyle='-')
# plt.title('Convergence of Objective Function')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.grid(True)
# plt.show()

# # Plot the NEES values separately
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(nees_values) + 1), nees_values, marker='o', linestyle='-')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()

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
    
    process_noise_covariance_matrix = np.array(process_noise_covariance_matrix).reshape((15, 15))
    
    # Calculate trace of the process noise covariance matrix
    trace = np.trace(process_noise_covariance_matrix)
    
    # Loop over data in batches of 50
    num_batches = len(data) // 50

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

        # Debugging: Check if data batches are correct
        print(f"Batch {i+1}/{num_batches}")
        print("Ground Truth Positions (first 5):", x_gt_batch[:5, :3])
        print("Estimated Positions (first 5):", x_est_batch[:5, :3])
        print("Ground Truth Orientations (first 5):", x_gt_batch[:5, 3:6])
        print("Estimated Orientations (first 5):", x_est_batch[:5, 3:6])

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

# Perform Random Search optimization
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

# import pandas as pd
# import numpy as np
# from skopt import dummy_minimize
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     objective_value_trace = 0  # Initialize trace term
    
#     process_noise_covariance_matrix = np.array(process_noise_covariance_matrix).reshape((15, 15))
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using quaternions
#         for j in range(50):
#             gt_quat = np.array([x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 5], x_gt_batch[j, 6]])  # [qw, qx, qy, qz]
#             est_quat = np.array([x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 5], x_est_batch[j, 6]])
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
        
#     avg_rpe_translation = objective_value_rpe_translation / num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
#     objective_value_nees /= num_batches
    
#     # Weightage for trace term and NEES
#     weight_trace = 0.1
#     weight_nees = 0.1
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.4 * avg_rpe_translation + 
#                        0.4 * avg_rpe_rotation + 
#                        0.1 * objective_value_nees +
#                        0.1 * objective_value_trace)  # Add trace term
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)  # Print trace term
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value, objective_value_nees

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.9)
# bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.00001, 0.1)

# # Store NEES values for plotting
# nees_values = []

# # Modified objective function to capture NEES values
# def modified_objective_function(x):
#     obj_value, nees_value = objective_function(x)
#     nees_values.append(nees_value)
#     return obj_value

# # Perform Random Search optimization
# res = dummy_minimize(modified_objective_function, bounds, n_calls=30, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Extract the values of the objective function at each iteration
# objective_values = res.func_vals

# # Plot the convergence of the objective function
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(objective_values) + 1), objective_values, marker='o', linestyle='-')
# plt.title('Convergence of Objective Function')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.grid(True)
# # plt.ylim(20, 20.25)
# plt.show()

# # Plot the NEES values separately
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(nees_values) + 1), nees_values, marker='o', linestyle='-')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()

# import pandas as pd
# import numpy as np
# from skopt import dummy_minimize
# from tf2_geometry_msgs import PointStamped
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     objective_value_trace = 0  # Initialize trace term
    
#     process_noise_covariance_1d = data.iloc[0, 40:].values

#     # Reshape process_noise_covariance_matrix to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using quaternions
#         for j in range(50):
#             gt_quat = np.array([x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 5], x_gt_batch[j, 6]])  # [qw, qx, qy, qz]
#             est_quat = np.array([x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 5], x_est_batch[j, 6]])
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
        
#     avg_rpe_translation = objective_value_rpe_translation / num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
#     objective_value_nees /= num_batches
    
#     # Weightage for trace term and NEES
#     weight_trace = 0.1
#     weight_nees = 0.1
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.4 * avg_rpe_translation + 
#                        0.4 * avg_rpe_rotation + 
#                        0.1 * objective_value_nees +
#                        0.1 * objective_value_trace)  # Add trace term
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)  # Print trace term
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.9)
# bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.9)  # Adjusting forzero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.00001, 0.1)

# # Perform Random Search optimization
# res = dummy_minimize(objective_function, bounds, n_calls=30, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Extract the values of the objective function at each iteration
# objective_values = res.func_vals

# # Plot the convergence of the objective function
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(objective_values) + 1), objective_values, marker='o', linestyle='-')
# plt.title('Convergence of Objective Function (Random Search)')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.grid(True)
# plt.show()

# import yaml
# import pandas as pd

# def read_process_noise_covariance(file_path):
#     with open(file_path, 'r') as file:
#         config = yaml.safe_load(file)
    
#     try:
#         covariance = config['ekf_filter_node']['ros__parameters']['process_noise_covariance']
#         return covariance
#     except KeyError as e:
#         print(f"KeyError: {e} not found in the configuration file.")
#         return None

# def store_covariance_in_excel(covariance, excel_file):
#     # Generate column names from 0 to 224
#     column_names = list(range(225))
    
#     # Convert the covariance list to a DataFrame with one row and specified column names
#     df = pd.DataFrame([covariance], columns=column_names)
    
#     # Create an Excel writer object and save the DataFrame to Excel
#     with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
#         df.to_excel(writer, index=False)

# if __name__ == "__main__":
#     yaml_file = '/home/asimkumar/asitlorbot8_ws/src/asitlorbot8_localization/config/ekf.yaml'
#     excel_file = 'process_noise_covariance.xlsx'
    
#     covariance = read_process_noise_covariance(yaml_file)
    
#     if covariance is not None:
#         print("Process Noise Covariance:")
#         print(covariance)
#         store_covariance_in_excel(covariance, excel_file)
#         print(f"Covariance stored in {excel_file}")
#     else:
#         print("Process noise covariance could not be read.")


# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from tf2_geometry_msgs import PointStamped
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     process_noise_covariance_1d = data.iloc[0, 40:].values

#     # Reshape process_noise_covariance_matrix to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:end_idx].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using quaternions
#         for j in range(50):
#             gt_quat = np.array([x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 5], x_gt_batch[j, 6]])  # [qw, qx, qy, qz]
#             est_quat = np.array([x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 5], x_est_batch[j, 6]])
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
        
#     avg_rpe_translation = objective_value_rpe_translation / num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / (num_batches * 50)
#     objective_value_nees /= num_batches
    
#     # Combine NEES and RPE with weights
#     objective_value = (0.4 * avg_rpe_translation + 
#                        0.4 * avg_rpe_rotation + 
#                        0.2 * objective_value_nees)
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.2)
# bounds[96] = (0.00002, 0.1)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.15)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.0001, 0.1)

# # Perform Bayesian optimization
# res = gp_minimize(objective_function, bounds, acq_func="EI", n_calls=20, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)



# import pandas as pd
# import numpy as np
# from skopt import gp_minimize

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value_nees = 0
#     objective_value_rmse = 0
#     process_noise_covariance_1d = data.iloc[0, 40:].values

#     # Reshape process_noise_covariance_matrix to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
    
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z','GT_Roll','GT_Pitch','GT_Yaw','GT_Vel_X','GT_Vel_Y','GT_Vel_Z','GT_Vel_Roll','GT_Vel_Pitch','GT_Angular_Vel_Yaw','GT_Accel_X','GT_Accel_Y','GT_Accel_Z']][start_idx:end_idx].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z','ET_Roll','ET_Pitch','ET_Yaw','ET_Vel_X','ET_Vel_Y','ET_Vel_Z','ET_Vel_Roll','ET_Vel_Pitch','ET_Angular_Vel_Yaw','ET_Accel_X','ET_Accel_Y','ET_Accel_Z']][start_idx:end_idx].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value
        
#         # Compute RMSE for GT_Angular_Vel_Z and GT_Yaw
#         rmse_batch = np.sqrt(np.mean((x_gt_batch[:, [3, 0]] - x_est_batch[:, [3, 0]]) ** 2))  # GT_Angular_Vel_Z and GT_Yaw
#         objective_value_rmse += rmse_batch

#     # Average objective value over all batches
#     objective_value_nees /= num_batches
#     objective_value_rmse /= num_batches
    
#     # Combine NEES and RMSE with weights
#     objective_value = 0 * objective_value_rmse + 1 * objective_value_nees
    
#     print("Objective Value RMSE: ", objective_value_rmse)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.2)
# bounds[96] = (0.00002, 0.1)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.15)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.0001, 0.1)
# # Perform Bayesian optimization
# res = gp_minimize(objective_function, bounds, acq_func="EI", n_calls=20, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)









# import pandas as pd
# import numpy as np
# from numpy.linalg import det,inv
# from skopt import gp_minimize

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_matrix):
#     objective_value = 0
#     process_noise_covariance_1d = data.iloc[0, 40:].values

#     # Reshape process_noise_covariance_matrix to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#     # print(process_noise_covariance_matrix)
#     # print(det(process_noise_covariance_matrix))
#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z','GT_Roll','GT_Pitch','GT_Yaw','GT_Vel_X','GT_Vel_Y','GT_Vel_Z','GT_Vel_Roll','GT_Vel_Pitch','GT_Angular_Vel_Yaw','GT_Accel_X','GT_Accel_Y','GT_Accel_Z']][start_idx:end_idx].values
#         print(x_gt_batch)
#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z','ET_Roll','ET_Pitch','ET_Yaw','ET_Vel_X','ET_Vel_Y','ET_Vel_Z','ET_Vel_Roll','ET_Vel_Pitch','ET_Angular_Vel_Yaw','ET_Accel_X','ET_Accel_Y','ET_Accel_Z']][start_idx:end_idx].values

#         # Compute estimation error for the current batch
#         e_x_batch = x_gt_batch - x_est_batch
        
#         NEES_batch = np.zeros(50)
        
#         for j in range(50):
#             e_x_reshaped = e_x_batch[j].reshape(-1, 1)
#             NEES_row = np.transpose(e_x_reshaped) @ np.linalg.inv(process_noise_covariance_matrix) @ e_x_reshaped
#             NEES_batch[j] = NEES_row
#             # print(NEES_batch)
        
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value += np.abs(avg_NEES_batch - 9.488)  # Target NEES value

#     # Average objective value over all batches
#     objective_value /= num_batches
#     print("Objective Value: ",objective_value)
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(-0.01, 1)] * 225  # Assuming a 15x15 matrix

# # Perform Bayesian optimization
# res = gp_minimize(objective_function, bounds, acq_func="EI", n_calls=20, random_state=42)

# # Extract the optimized process noise covariance matrix
# optimized_process_noise_covariance_matrix = np.array(res.x).reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# GT_Roll	GT_Pitch	@GT_Yaw	@GT_Vel_X	@GT_Vel_Y	GT_Vel_Z	
# GT_Vel_Roll	GT_Vel_Pitch	@GT_Angular_Vel_Yaw	@GT_Accel_X	GT_Accel_Y	GT_Accel_Z	
# GT_Pos_X	GT_Pos_YGT_Pos_Z	@GT_Orient_X	GT_Orient_Y	GT_Orient_Z	GT_Orient_W	

# ET_Roll	ET_Pitch	ET_Yaw	ET_Vel_X	ET_Vel_Y	ET_Vel_Z	
# ET_Vel_Roll	ET_Vel_Pitch	ET_Angular_Vel_Yaw	ET_Accel_X	ET_Accel_Y	ET_Accel_Z	
# ET_Pos_X	ET_Pos_Y	ET_Pos_Z	Est_Orient_X	Est_Orient_Y	Est_Orient_Z	Est_Orient_W

# GT_Yaw	GT_Vel_X	GT_Vel_Y	GT_Angular_Vel_Z	GT_Accel_X	GT_Pos_X	GT_Pos_Y	GT_Pos_Z	
# GT_Orient_X	GT_Orient_Y	GT_Orient_Z	GT_Orient_W	

# Est_Yaw	Est_Vel_X	Est_Vel_Y	Est_Angular_Vel_Z	Est_Accel_X	Est_Pos_X	Est_Pos_Y	Est_Pos_Z	
# Est_Orient_X	Est_Orient_Y	Est_Orient_Z	Est_Orient_W








# import rclpy
# import yaml
# import pandas as pd
# from rclpy.node import Node
# from geometry_msgs.msg import Pose, Point, PoseStamped, Twist
# from nav_msgs.msg import Odometry  # Import Odometry message type
# from visualization_msgs.msg import Marker, MarkerArray
# import tf2_ros
# from tf2_geometry_msgs import PointStamped
# import numpy as np
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion

# class Subscriber(Node):

#     def __init__(self):
#         super().__init__('subscriber')
#         self.get_logger().info('Subscriber node initialized')

#         self.buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.buffer, self)

#         self.timer = self.create_timer(2, self.timer_callback)

#         self.ground_truth_data = []  # Store ground truth data
#         self.estimated_data = []  # Store estimated data

#         # Publishers for markers
#         self.publisher_gt_marker = self.create_publisher(MarkerArray, 'ground_truth_trajectory', 10)
#         self.publisher_est_marker = self.create_publisher(MarkerArray, 'estimated_trajectory', 10)

#         # Subscribers for Odometry messages
#         self.get_logger().info('Subscribing to ground truth Odometry topic: /diff_cont/odom')
#         self.gt_odom_subscriber = self.create_subscription(Odometry, '/diff_cont/odom', self.gt_odom_callback, 10)

#         self.get_logger().info('Subscribing to estimated Odometry topic: /odometry/filtered')
#         self.est_odom_subscriber = self.create_subscription(Odometry, '/odometry/filtered', self.est_odom_callback, 10)

#         # Initialize variables to store Odometry data
#         self.gt_odom_data = None
#         self.est_odom_data = None

#     def timer_callback(self):
#         try:
#             # Get transforms directly
#             trans_base_footprint = self.buffer.lookup_transform(
#                 'odom', 'base_footprint', Time(seconds=0)
#             )
#             trans_base_footprint_ekf = self.buffer.lookup_transform(
#                 'odom', 'base_footprint_ekf', Time(seconds=0)
#             )

#             # Extract ground truth pose data
#             gt_pose_data = self.extract_data_from_transform(trans_base_footprint, self.gt_odom_data)

#             # Extract estimated pose data
#             est_pose_data = self.extract_data_from_transform(trans_base_footprint_ekf, self.est_odom_data)

#             # Append ground truth and estimated data
#             self.ground_truth_data.append(gt_pose_data)
#             self.estimated_data.append(est_pose_data)

#             # Publish trajectory markers
#             self.publish_trajectory_markers()

#             # Check if enough data points collected
#             if len(self.ground_truth_data) == 25:  # Assuming you want 50 data points
#                 # Save data to Excel
#                 self.save_to_excel()
#                 # Shut down the node after saving data to Excel
#                 rclpy.shutdown()

#         except Exception as e:
#             self.get_logger().error('Error getting transforms: {}'.format(str(e)))

#     def gt_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received ground truth Odometry message: {}'.format(msg))
#         self.gt_odom_data = msg
#         # Print out linear and angular velocities
#         self.get_logger().info('Linear Velocity X: {}'.format(msg.twist.twist.linear.x))
#         self.get_logger().info('Linear Velocity Y: {}'.format(msg.twist.twist.linear.y))
#         self.get_logger().info('Angular Velocity Z: {}'.format(msg.twist.twist.angular.z))

#     def est_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received estimated Odometry message: {}'.format(msg))
#         self.est_odom_data = msg
#         # Print out linear and angular velocities
#         self.get_logger().info('Linear Velocity X: {}'.format(msg.twist.twist.linear.x))
#         self.get_logger().info('Linear Velocity Y: {}'.format(msg.twist.twist.linear.y))
#         self.get_logger().info('Angular Velocity Z: {}'.format(msg.twist.twist.angular.z))

#     def extract_data_from_transform(self, transform, odom_data=None):
#         position = transform.transform.translation
#         orientation = transform.transform.rotation

#         # Convert orientation to yaw
#         quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
#         _, _, yaw = euler_from_quaternion(quaternion)

#         # Default values for velocity and acceleration
#         velocity_x = 0.0
#         velocity_y = 0.0
#         angular_velocity_z = 0.0
#         acceleration_x = 0.0

#         if odom_data is not None:
#             linear_velocity = odom_data.twist.twist.linear
#             angular_velocity = odom_data.twist.twist.angular

#             velocity_x = linear_velocity.x
#             velocity_y = linear_velocity.y
#             angular_velocity_z = angular_velocity.z

#         return [yaw, velocity_x, velocity_y, angular_velocity_z, acceleration_x,
#                 position.x, position.y, position.z,
#                 orientation.x, orientation.y, orientation.z, orientation.w]

#     def publish_trajectory_markers(self):
#         # Create MarkerArray for ground truth trajectory
#         gt_marker_array = MarkerArray()
#         gt_marker = Marker()
#         gt_marker.header.frame_id = 'odom'
#         gt_marker.type = Marker.LINE_STRIP
#         gt_marker.action = Marker.ADD
#         gt_marker.pose.orientation.w = 1.0
#         gt_marker.scale.x = 0.01  # Line width
#         gt_marker.color.r = 0.0  # Red color
#         gt_marker.color.g = 1.0
#         gt_marker.color.b = 0.0
#         gt_marker.color.a = 1.0  # Full opacity

#         # Add points to ground truth marker
#         for data in self.ground_truth_data:
#             point = Point()
#             point.x = data[5]  # GT_Pos_X
#             point.y = data[6]  # GT_Pos_Y
#             point.z = data[7]  # GT_Pos_Z
#             gt_marker.points.append(point)

#         gt_marker_array.markers.append(gt_marker)
#         self.publisher_gt_marker.publish(gt_marker_array)

#         # Create MarkerArray for estimated trajectory
#         est_marker_array = MarkerArray()
#         est_marker = Marker()
#         est_marker.header.frame_id = 'odom'
#         est_marker.type = Marker.LINE_STRIP
#         est_marker.action = Marker.ADD
#         est_marker.pose.orientation.w = 1.0
#         est_marker.scale.x = 0.01  # Line width
#         est_marker.color.r = 1.0  # Red color
#         est_marker.color.g = 0.0
#         est_marker.color.b = 0.0
#         est_marker.color.a = 1.0  # Full opacity

#         # Add points to estimated marker
#         for data in self.estimated_data:
#             point = Point()
#             point.x = data[5]  # Est_Pos_X
#             point.y = data[6]  # Est_Pos_Y
#             point.z = data[7]  # Est_Pos_Z
#             est_marker.points.append(point)

#         est_marker_array.markers.append(est_marker)
#         self.publisher_est_marker.publish(est_marker_array)

#     def save_to_excel(self):
#         excel_filename = 'pose_and_data.xlsx'

#         # Create DataFrame for ground truth data
#         ground_truth_df = pd.DataFrame(self.ground_truth_data, columns=['GT_Yaw', 'GT_Vel_X', 'GT_Vel_Y', 'GT_Angular_Vel_Z', 'GT_Accel_X',
#                                                                         'GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z',
#                                                                         'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W'])

#         # Create DataFrame for estimated data
#         estimated_df = pd.DataFrame(self.estimated_data, columns=['Est_Yaw', 'Est_Vel_X', 'Est_Vel_Y', 'Est_Angular_Vel_Z', 'Est_Accel_X',
#                                                                   'Est_Pos_X', 'Est_Pos_Y', 'Est_Pos_Z',
#                                                                   'Est_Orient_X', 'Est_Orient_Y', 'Est_Orient_Z', 'Est_Orient_W'])

#         # Combine ground truth and estimated DataFrames
#         combined_df = pd.concat([ground_truth_df, estimated_df], axis=1)

#         # Save DataFrame to Excel
#         combined_df.to_excel(excel_filename, index=False)

#         self.get_logger().info('Data saved to {}'.format(excel_filename))

# def main(args=None):
#     rclpy.init(args=args)

#     subscriber = Subscriber()

#     rclpy.spin(subscriber)

# if __name__ == "__main__":
#     main()

# import numpy as np
# process_noise_covariance_matrix = [[1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.000000e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.000000e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.500000e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.428886e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00],
#                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01]]

# e_X_shape =  [[-3.93811211],[-7.74779448],[0.0],[0.0],[0.0],[-0.22775951],[-2.98],[0.0],[0.0],[0.0],[0.0],[-1.19],[-0.049],[0.0],[0.0]]

# NEES_row = np.transpose(e_X_shape) @ (np.linalg.inv(process_noise_covariance_matrix) @ e_X_shape)
# print(np.linalg.inv(process_noise_covariance_matrix))
# print(np.linalg.inv(process_noise_covariance_matrix) @ e_X_shape)
# print(np.transpose(e_X_shape))
# print(NEES_row)

# import rclpy
# import yaml
# import pandas as pd
# from rclpy.node import Node
# from geometry_msgs.msg import Pose, Point, PoseStamped, Twist
# from nav_msgs.msg import Odometry  # Import Odometry message type
# from visualization_msgs.msg import Marker, MarkerArray
# import tf2_ros
# from tf2_geometry_msgs import PointStamped
# import numpy as np
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion

# class Subscriber(Node):

#     def __init__(self):
#         super().__init__('subscriber')
#         self.get_logger().info('Subscriber node initialized')

#         self.buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.buffer, self)

#         self.timer = self.create_timer(2, self.timer_callback)

#         self.ground_truth_data = []  # Store ground truth data
#         self.estimated_data = []  # Store estimated data

#         # Publishers for markers
#         self.publisher_gt_marker = self.create_publisher(MarkerArray, 'ground_truth_trajectory', 10)
#         self.publisher_est_marker = self.create_publisher(MarkerArray, 'estimated_trajectory', 10)

#         # Subscribers for Odometry messages
#         self.get_logger().info('Subscribing to ground truth Odometry topic: /diff_cont/odom')
#         self.gt_odom_subscriber = self.create_subscription(Odometry, '/diff_cont/odom', self.gt_odom_callback, 10)

#         self.get_logger().info('Subscribing to estimated Odometry topic: /odometry/filtered')
#         self.est_odom_subscriber = self.create_subscription(Odometry, '/odometry/filtered', self.est_odom_callback, 10)

#         # Initialize variables to store Odometry data
#         self.gt_odom_data = None
#         self.est_odom_data = None

#     def timer_callback(self):
#         try:
#             # Get transforms directly
#             trans_base_footprint = self.buffer.lookup_transform(
#                 'odom', 'base_footprint', Time(seconds=0)
#             )
#             trans_base_footprint_ekf = self.buffer.lookup_transform(
#                 'odom', 'base_footprint_ekf', Time(seconds=0)
#             )

#             # Extract ground truth pose data
#             gt_pose_data = self.extract_data_from_transform(trans_base_footprint, self.gt_odom_data)

#             # Extract estimated pose data
#             est_pose_data = self.extract_data_from_transform(trans_base_footprint_ekf, self.est_odom_data)

#             # Append ground truth and estimated data
#             self.ground_truth_data.append(gt_pose_data)
#             self.estimated_data.append(est_pose_data)

#             # Publish trajectory markers
#             self.publish_trajectory_markers()

#             # Check if enough data points collected
#             if len(self.ground_truth_data) == 25:  # Assuming you want 50 data points
#                 # Save data to Excel
#                 self.save_to_excel()
#                 # Shut down the node after saving data to Excel
#                 rclpy.shutdown()

#         except Exception as e:
#             self.get_logger().error('Error getting transforms: {}'.format(str(e)))

#     def gt_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received ground truth Odometry message: {}'.format(msg))
#         self.gt_odom_data = msg
#         # Print out linear and angular velocities
#         self.get_logger().info('Linear Velocity X: {}'.format(msg.twist.twist.linear.x))
#         self.get_logger().info('Linear Velocity Y: {}'.format(msg.twist.twist.linear.y))
#         self.get_logger().info('Angular Velocity Z: {}'.format(msg.twist.twist.angular.z))

#     def est_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received estimated Odometry message: {}'.format(msg))
#         self.est_odom_data = msg
#         # Print out linear and angular velocities
#         self.get_logger().info('Linear Velocity X: {}'.format(msg.twist.twist.linear.x))
#         self.get_logger().info('Linear Velocity Y: {}'.format(msg.twist.twist.linear.y))
#         self.get_logger().info('Angular Velocity Z: {}'.format(msg.twist.twist.angular.z))

#     def extract_data_from_transform(self, transform, odom_data=None):
#         position = transform.transform.translation
#         orientation = transform.transform.rotation

#         # Convert orientation to yaw
#         quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
#         _, _, yaw = euler_from_quaternion(quaternion)

#         # Default values for velocity and acceleration
#         velocity_x = 0.0
#         velocity_y = 0.0
#         angular_velocity_z = 0.0
#         acceleration_x = 0.0

#         if odom_data is not None:
#             linear_velocity = odom_data.twist.twist.linear
#             angular_velocity = odom_data.twist.twist.angular

#             velocity_x = linear_velocity.x
#             velocity_y = linear_velocity.y
#             angular_velocity_z = angular_velocity.z

#         return [yaw, velocity_x, velocity_y, angular_velocity_z, acceleration_x,
#                 position.x, position.y, position.z,
#                 orientation.x, orientation.y, orientation.z, orientation.w]

#     def publish_trajectory_markers(self):
#         # Create MarkerArray for ground truth trajectory
#         gt_marker_array = MarkerArray()
#         gt_marker = Marker()
#         gt_marker.header.frame_id = 'odom'
#         gt_marker.type = Marker.LINE_STRIP
#         gt_marker.action = Marker.ADD
#         gt_marker.pose.orientation.w = 1.0
#         gt_marker.scale.x = 0.01  # Line width
#         gt_marker.color.r = 0.0  # Red color
#         gt_marker.color.g = 1.0
#         gt_marker.color.b = 0.0
#         gt_marker.color.a = 1.0  # Full opacity

#         # Add points to ground truth marker
#         for data in self.ground_truth_data:
#             point = Point()
#             point.x = data[5]  # GT_Pos_X
#             point.y = data[6]  # GT_Pos_Y
#             point.z = data[7]  # GT_Pos_Z
#             gt_marker.points.append(point)

#         gt_marker_array.markers.append(gt_marker)
#         self.publisher_gt_marker.publish(gt_marker_array)

#         # Create MarkerArray for estimated trajectory
#         est_marker_array = MarkerArray()
#         est_marker = Marker()
#         est_marker.header.frame_id = 'odom'
#         est_marker.type = Marker.LINE_STRIP
#         est_marker.action = Marker.ADD
#         est_marker.pose.orientation.w = 1.0
#         est_marker.scale.x = 0.01  # Line width
#         est_marker.color.r = 1.0  # Red color
#         est_marker.color.g = 0.0
#         est_marker.color.b = 0.0
#         est_marker.color.a = 1.0  # Full opacity

#         # Add points to estimated marker
#         for data in self.estimated_data:
#             point = Point()
#             point.x = data[5]  # Est_Pos_X
#             point.y = data[6]  # Est_Pos_Y
#             point.z = data[7]  # Est_Pos_Z
#             est_marker.points.append(point)

#         est_marker_array.markers.append(est_marker)
#         self.publisher_est_marker.publish(est_marker_array)

#     def save_to_excel(self):
#         excel_filename = 'pose_and_data.xlsx'

#         # Create DataFrame for ground truth data
#         ground_truth_df = pd.DataFrame(self.ground_truth_data, columns=['GT_Yaw', 'GT_Vel_X', 'GT_Vel_Y', 'GT_Angular_Vel_Z', 'GT_Accel_X',
#                                                                         'GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z',
#                                                                         'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W'])

#         # Create DataFrame for estimated data
#         estimated_df = pd.DataFrame(self.estimated_data, columns=['Est_Yaw', 'Est_Vel_X', 'Est_Vel_Y', 'Est_Angular_Vel_Z', 'Est_Accel_X',
#                                                                   'Est_Pos_X', 'Est_Pos_Y', 'Est_Pos_Z',
#                                                                   'Est_Orient_X', 'Est_Orient_Y', 'Est_Orient_Z', 'Est_Orient_W'])

#         # Combine ground truth and estimated DataFrames
#         combined_df = pd.concat([ground_truth_df, estimated_df], axis=1)

#         # Save DataFrame to Excel
#         combined_df.to_excel(excel_filename, index=False)

#         self.get_logger().info('Data saved to {}'.format(excel_filename))

# def main(args=None):
#     rclpy.init(args=args)

#     subscriber = Subscriber()

#     rclpy.spin(subscriber)

# if __name__ == "__main__":
#     main()

# import rclpy
# import yaml
# import pandas as pd
# from rclpy.node import Node
# from geometry_msgs.msg import Pose, Point, PoseStamped, Twist
# from nav_msgs.msg import Odometry  # Import Odometry message type
# from visualization_msgs.msg import Marker, MarkerArray
# import tf2_ros
# from tf2_geometry_msgs import PointStamped
# import numpy as np
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion

# class Subscriber(Node):

#     def __init__(self):
#         super().__init__('subscriber')
#         self.get_logger().info('Subscriber node initialized')

#         self.buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.buffer, self)

#         self.timer = self.create_timer(2, self.timer_callback)

#         self.ground_truth_data = []  # Store ground truth data
#         self.estimated_data = []  # Store estimated data

#         # Publishers for markers
#         self.publisher_gt_marker = self.create_publisher(MarkerArray, 'ground_truth_trajectory', 10)
#         self.publisher_est_marker = self.create_publisher(MarkerArray, 'estimated_trajectory', 10)

#         # Subscribers for Odometry messages
#         self.get_logger().info('Subscribing to ground truth Odometry topic: /diff_cont/odom')
#         self.gt_odom_subscriber = self.create_subscription(Odometry, '/diff_cont/odom', self.gt_odom_callback, 10)

#         self.get_logger().info('Subscribing to estimated Odometry topic: /odometry/filtered')
#         self.est_odom_subscriber = self.create_subscription(Odometry, '/odometry/filtered', self.est_odom_callback, 10)

#         # Initialize variables to store Odometry data
#         self.gt_odom_data = None
#         self.est_odom_data = None
#         self.prev_gt_velocity_x = 0.0
#         self.prev_est_velocity_x = 0.0
#         self.prev_timestamp = None

#     def timer_callback(self):
#         try:
#             # Get transforms directly
#             trans_base_footprint = self.buffer.lookup_transform(
#                 'odom', 'base_footprint', Time(seconds=0)
#             )
#             trans_base_footprint_ekf = self.buffer.lookup_transform(
#                 'odom', 'base_footprint_ekf', Time(seconds=0)
#             )

#             # Extract ground truth pose data
#             gt_pose_data = self.extract_data_from_transform(trans_base_footprint, self.gt_odom_data)

#             # Extract estimated pose data
#             est_pose_data = self.extract_data_from_transform(trans_base_footprint_ekf, self.est_odom_data)

#             # Append ground truth and estimated data
#             self.ground_truth_data.append(gt_pose_data)
#             self.estimated_data.append(est_pose_data)

#             # Publish trajectory markers
#             self.publish_trajectory_markers()

#             # Check if enough data points collected
#             if len(self.ground_truth_data) == 25:  # Assuming you want 50 data points
#                 # Save data to Excel
#                 self.save_to_excel()
#                 # Shut down the node after saving data to Excel
#                 rclpy.shutdown()

#         except Exception as e:
#             self.get_logger().error('Error getting transforms: {}'.format(str(e)))

#     def gt_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received ground truth Odometry message: {}'.format(msg))
#         self.gt_odom_data = msg

#         if self.prev_timestamp is not None:
#             delta_t = (msg.header.stamp.sec - self.prev_timestamp.sec) + \
#                       (msg.header.stamp.nanosec - self.prev_timestamp.nanosec) / 1e9
#             acceleration_x = (msg.twist.twist.linear.x - self.prev_gt_velocity_x) / delta_t
#             acceleration_x = 0.0 if abs(acceleration_x) < 0.0001 else acceleration_x
#         else:
#             acceleration_x = 0.0

#         self.prev_gt_velocity_x = msg.twist.twist.linear.x
#         self.prev_timestamp = msg.header.stamp

#         self.get_logger().info('Ground Truth Acceleration X: {}'.format(acceleration_x))

#     def est_odom_callback(self, msg: Odometry):
#         self.get_logger().info('Received estimated Odometry message: {}'.format(msg))
#         self.est_odom_data = msg

#         if self.prev_timestamp is not None:
#             delta_t = (msg.header.stamp.sec - self.prev_timestamp.sec) + \
#                       (msg.header.stamp.nanosec - self.prev_timestamp.nanosec) / 1e9
#             acceleration_x = (msg.twist.twist.linear.x - self.prev_est_velocity_x) / delta_t
#             acceleration_x = 0.0 if abs(acceleration_x) < 0.0001 else acceleration_x
#         else:
#             acceleration_x = 0.0

#         self.prev_est_velocity_x = msg.twist.twist.linear.x
#         self.prev_timestamp = msg.header.stamp

#         self.get_logger().info('Estimated Acceleration X: {}'.format(acceleration_x))

#     def extract_data_from_transform(self, transform, odom_data=None):
#         position = transform.transform.translation
#         orientation = transform.transform.rotation

#         # Convert orientation to yaw
#         quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
#         _, _, yaw = euler_from_quaternion(quaternion)

#         # Default values for velocity and acceleration
#         velocity_x = 0.0
#         velocity_y = 0.0
#         angular_velocity_z = 0.0
#         acceleration_x = 0.0

#         if odom_data is not None:
#             linear_velocity = odom_data.twist.twist.linear
#             angular_velocity = odom_data.twist.twist.angular

#             velocity_x = linear_velocity.x
#             velocity_y = linear_velocity.y
#             angular_velocity_z = angular_velocity.z

#         return [yaw, velocity_x, velocity_y, angular_velocity_z, acceleration_x,
#                 position.x, position.y, position.z,
#                 orientation.x, orientation.y, orientation.z, orientation.w]

#     def publish_trajectory_markers(self):
#         # Create MarkerArray for ground truth trajectory
#         gt_marker_array = MarkerArray()
#         gt_marker = Marker()
#         gt_marker.header.frame_id = 'odom'
#         gt_marker.type = Marker.LINE_STRIP
#         gt_marker.action = Marker.ADD
#         gt_marker.pose.orientation.w = 1.0
#         gt_marker.scale.x = 0.01  # Line width
#         gt_marker.color.r = 0.0  # Red color
#         gt_marker.color.g = 1.0
#         gt_marker.color.b = 0.0
#         gt_marker.color.a = 1.0  # Full opacity

#         # Add points to ground truth marker
#         for data in self.ground_truth_data:
#             point = Point()
#             point.x = data[5]  # GT_Pos_X
#             point.y = data[6]  # GT_Pos_Y
#             point.z = data[7]  # GT_Pos_Z
#             gt_marker.points.append(point)

#         gt_marker_array.markers.append(gt_marker)
#         self.publisher_gt_marker.publish(gt_marker_array)

#         # Create MarkerArray for estimated trajectory
#         est_marker_array = MarkerArray()
#         est_marker = Marker()
#         est_marker.header.frame_id = 'odom'
#         est_marker.type = Marker.LINE_STRIP
#         est_marker.action = Marker.ADD
#         est_marker.pose.orientation.w = 1.0
#         est_marker.scale.x = 0.01  # Line width
#         est_marker.color.r = 1.0  # Red color
#         est_marker.color.g = 0.0
#         est_marker.color.b = 0.0
#         est_marker.color.a = 1.0  # Full opacity

#         # Add points to estimated marker
#         for data in self.estimated_data:
#             point = Point()
#             point.x = data[5]  # Est_Pos_X
#             point.y = data[6]  # Est_Pos_Y
#             point.z = data[7]  # Est_Pos_Z
#             est_marker.points.append(point)

#         est_marker_array.markers.append(est_marker)
#         self.publisher_est_marker.publish(est_marker_array)

#     def save_to_excel(self):
#         excel_filename = 'pose_and_data.xlsx'

#         # Create DataFrame for ground truth data
#         ground_truth_df = pd.DataFrame(self.ground_truth_data, columns=['GT_Yaw', 'GT_Vel_X', 'GT_Vel_Y', 'GT_Angular_Vel_Z', 'GT_Accel_X',
#                                                                         'GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z',
#                                                                         'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W'])

#         # Create DataFrame for estimated data
#         estimated_df = pd.DataFrame(self.estimated_data, columns=['Est_Yaw', 'Est_Vel_X', 'Est_Vel_Y', 'Est_Angular_Vel_Z', 'Est_Accel_X',
#                                                                   'Est_Pos_X', 'Est_Pos_Y', 'Est_Pos_Z',
#                                                                   'Est_Orient_X', 'Est_Orient_Y', 'Est_Orient_Z', 'Est_Orient_W'])

#         # Combine ground truth and estimated DataFrames
#         combined_df = pd.concat([ground_truth_df, estimated_df], axis=1)

#         # Save DataFrame to Excel
#         combined_df.to_excel(excel_filename, index=False)

#         self.get_logger().info('Data saved to {}'.format(excel_filename))

# def main(args=None):
#     rclpy.init(args=args)

#     subscriber = Subscriber()

#     rclpy.spin(subscriber)

# if __name__ == "__main__":
#     main()

# import rclpy
# import yaml
# import pandas as pd
# from rclpy.node import Node
# from geometry_msgs.msg import Pose, Point, PoseStamped, Twist
# from nav_msgs.msg import Odometry  # Import Odometry message type
# from visualization_msgs.msg import Marker, MarkerArray
# import tf2_ros
# from tf2_geometry_msgs import PointStamped
# import numpy as np
# from rclpy.time import Time
# from tf_transformations import euler_from_quaternion

# class Subscriber(Node):

#     def __init__(self):
#         super().__init__('subscriber')
#         self.get_logger().info('Subscriber node initialized')

#         self.buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.buffer, self)

#         self.timer = self.create_timer(2, self.timer_callback)

#         self.ground_truth_data = []  # Store ground truth data
#         self.estimated_data = []  # Store estimated data

#         # Publishers for markers
#         self.publisher_gt_marker = self.create_publisher(MarkerArray, 'ground_truth_trajectory', 10)
#         self.publisher_est_marker = self.create_publisher(MarkerArray, 'estimated_trajectory', 10)

#         # Subscribers for Odometry messages
#         self.get_logger().info('Subscribing to ground truth Odometry topic: /diff_cont/odom')
#         self.gt_odom_subscriber = self.create_subscription(Odometry, '/diff_cont/odom', self.gt_odom_callback, 10)

#         self.get_logger().info('Subscribing to estimated Odometry topic: /odometry/filtered')
#         self.est_odom_subscriber = self.create_subscription(Odometry, '/odometry/filtered', self.est_odom_callback, 10)

#         # Initialize variables to store Odometry data
#         self.prev_gt_velocity_x = 0.0
#         self.prev_est_velocity_x = 0.0
#         self.prev_timestamp = None

#     def timer_callback(self):
#         try:
#             # Get transforms directly
#             trans_base_footprint = self.buffer.lookup_transform(
#                 'odom', 'base_footprint', Time(seconds=0)
#             )
#             trans_base_footprint_ekf = self.buffer.lookup_transform(
#                 'odom', 'base_footprint_ekf', Time(seconds=0)
#             )

#             # Extract ground truth pose data
#             gt_pose_data = self.extract_data_from_transform(trans_base_footprint, self.prev_gt_velocity_x)

#             # Extract estimated pose data
#             est_pose_data = self.extract_data_from_transform(trans_base_footprint_ekf, self.prev_est_velocity_x)

#             # Append ground truth and estimated data
#             self.ground_truth_data.append(gt_pose_data)
#             self.estimated_data.append(est_pose_data)

#             # Publish trajectory markers
#             self.publish_trajectory_markers()

#             # Check if enough data points collected
#             if len(self.ground_truth_data) == 25:  # Assuming you want 50 data points
#                 # Save data to Excel
#                 self.save_to_excel()
#                 # Shut down the node after saving data to Excel
#                 rclpy.shutdown()

#         except Exception as e:
#             self.get_logger().error('Error getting transforms: {}'.format(str(e)))

#     def gt_odom_callback(self, msg: Odometry):
#         # Calculate acceleration in x direction
#         if self.prev_timestamp is not None:
#             delta_t = (msg.header.stamp.sec - self.prev_timestamp.sec) + \
#                     (msg.header.stamp.nanosec - self.prev_timestamp.nanosec) / 1e9
#             acceleration_x = (msg.twist.twist.linear.x - self.prev_gt_velocity_x) / delta_t
#             # Set acceleration to zero if it's close to zero
#             acceleration_x = 0.0 if abs(acceleration_x) < 0.0001 else acceleration_x
#         else:
#             acceleration_x = 0.0

#         # Update previous velocity and timestamp for next iteration
#         self.prev_gt_velocity_x = msg.twist.twist.linear.x
#         self.prev_timestamp = msg.header.stamp

#         self.get_logger().info('Ground Truth Acceleration X: {}'.format(acceleration_x))

#     def est_odom_callback(self, msg: Odometry):
#         # Calculate acceleration in x direction
#         if self.prev_timestamp is not None:
#             delta_t = (msg.header.stamp.sec - self.prev_timestamp.sec) + \
#                     (msg.header.stamp.nanosec - self.prev_timestamp.nanosec) / 1e9
#             acceleration_x = (msg.twist.twist.linear.x - self.prev_est_velocity_x) / delta_t
#             # Set acceleration to zero if it's close to zero
#             acceleration_x = 0.0 if abs(acceleration_x) < 0.0001 else acceleration_x
#         else:
#             acceleration_x = 0.0

#         # Update previous velocity and timestamp for next iteration
#         self.prev_est_velocity_x = msg.twist.twist.linear.x
#         self.prev_timestamp = msg.header.stamp

#         self.get_logger().info('Estimated Acceleration X: {}'.format(acceleration_x))

#     def extract_data_from_transform(self, transform, prev_velocity_x):
#         position = transform.transform.translation
#         orientation = transform.transform.rotation

#         # Convert orientation to yaw
#         quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
#         _, _, yaw = euler_from_quaternion(quaternion)

#         # Default values for velocity and acceleration
#         velocity_x = 0.0
#         velocity_y = 0.0
#         angular_velocity_z = 0.0
#         acceleration_x = 0.0

#         if prev_velocity_x is not None:
#             velocity_x = prev_velocity_x

#         return [yaw, velocity_x, velocity_y, angular_velocity_z, acceleration_x,
#                 position.x, position.y, position.z,
#                 orientation.x, orientation.y, orientation.z, orientation.w]

#     def publish_trajectory_markers(self):
#         # Create MarkerArray for ground truth trajectory
#         gt_marker_array = MarkerArray()
#         gt_marker = Marker()
#         gt_marker.header.frame_id = 'odom'
#         gt_marker.type = Marker.LINE_STRIP
#         gt_marker.action = Marker.ADD
#         gt_marker.pose.orientation.w = 1.0
#         gt_marker.scale.x = 0.01  # Line width
#         gt_marker.color.r = 0.0  # Red color
#         gt_marker.color.g = 1.0
#         gt_marker.color.b = 0.0
#         gt_marker.color.a = 1.0  # Full opacity

#         # Add points to ground truth marker
#         for data in self.ground_truth_data:
#             point = Point()
#             point.x = data[5]  # GT_Pos_X
#             point.y = data[6]  # GT_Pos_Y
#             point.z = data[7]  # GT_Pos_Z
#             gt_marker.points.append(point)

#         gt_marker_array.markers.append(gt_marker)
#         self.publisher_gt_marker.publish(gt_marker_array)

#         # Create MarkerArray for estimated trajectory
#         est_marker_array = MarkerArray()
#         est_marker = Marker()
#         est_marker.header.frame_id = 'odom'
#         est_marker.type = Marker.LINE_STRIP
#         est_marker.action = Marker.ADD
#         est_marker.pose.orientation.w = 1.0
#         est_marker.scale.x = 0.01  # Line width
#         est_marker.color.r = 1.0  # Red color
#         est_marker.color.g = 0.0
#         est_marker.color.b = 0.0
#         est_marker.color.a = 1.0  # Full opacity

#         # Add points to estimated marker
#         for data in self.estimated_data:
#             point = Point()
#             point.x = data[5]  # Est_Pos_X
#             point.y = data[6]  # Est_Pos_Y
#             point.z = data[7]  # Est_Pos_Z
#             est_marker.points.append(point)

#         est_marker_array.markers.append(est_marker)
#         self.publisher_est_marker.publish(est_marker_array)

#     def save_to_excel(self):
#         excel_filename = 'pose_and_data.xlsx'

#         # Create DataFrame for ground truth data
#         ground_truth_df = pd.DataFrame(self.ground_truth_data, columns=['GT_Yaw', 'GT_Vel_X', 'GT_Vel_Y', 'GT_Angular_Vel_Z', 'GT_Accel_X',
#                                                                         'GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z',
#                                                                         'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W'])

#         # Create DataFrame for estimated data
#         estimated_df = pd.DataFrame(self.estimated_data, columns=['Est_Yaw', 'Est_Vel_X', 'Est_Vel_Y', 'Est_Angular_Vel_Z', 'Est_Accel_X',
#                                                                   'Est_Pos_X', 'Est_Pos_Y', 'Est_Pos_Z',
#                                                                   'Est_Orient_X', 'Est_Orient_Y', 'Est_Orient_Z', 'Est_Orient_W'])

#         # Combine ground truth and estimated DataFrames
#         combined_df = pd.concat([ground_truth_df, estimated_df], axis=1)

#         # Save DataFrame to Excel
#         combined_df.to_excel(excel_filename, index=False)

#         self.get_logger().info('Data saved to {}'.format(excel_filename))

# def main(args=None):
#     rclpy.init(args=args)

#     subscriber = Subscriber()

#     rclpy.spin(subscriber)

# if __name__ == "__main__":
#     main()
