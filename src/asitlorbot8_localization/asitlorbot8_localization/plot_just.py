import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
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

# Initialize lists to store average NEES values, relative position errors, relative orientation errors, and traces of process noise covariance matrices for each batch
average_NEES_values = []
average_position_errors = []
average_orientation_errors = []
average_trace_covariances = []

# Define the target NEES value
target_NEES = 24.996

# Define weights for each metric (you can adjust these weights based on importance)
weight_NEES = 1.0
weight_RPE = 0.5
weight_RRE = 0.5
weight_trace = 0.1  # Lower weight as trace value can vary significantly

# Initialize figure and axes for plots
fig, axs = plt.subplots(4, figsize=(10, 15))
axs[0].set_title('NEES')
axs[1].set_title('Relative Position Error')
axs[2].set_title('Relative Orientation Error')
axs[3].set_title('Trace of Process Noise Covariance')

# Loop over the data in steps of batch_size
for start_idx in range(0, total_rows, batch_size):
    # Determine the actual batch size (may be less than batch_size for the final batch)
    actual_batch_size = min(batch_size, total_rows - start_idx)
    
    # Extract ground truth state variables for the current batch
    x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)

    # Extract estimated state variables for the current batch
    x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
    
    # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
    x_est_batch.columns = gt_columns

    # Check if 'ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z' are in x_est_batch.columns
    if all(col in x_est_batch.columns for col in ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']):
        # Calculate errors and other metrics
        e_x_batch = x_gt_batch - x_est_batch
        
        # Calculate NEES, position error, orientation error, and trace of covariance matrix for each sample in the batch
        NEES_values = []
        position_errors = []
        orientation_errors = []
        trace_covariances = []

        for i in range(actual_batch_size):
            # Calculate position_error
            g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
            s_i = x_est_batch.iloc[i][['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']].values
            position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
            position_errors.append(position_error)

            # Calculate orientation_error
            g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
            s_quat = x_est_batch.iloc[i][['ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W']].values
            
            # Convert quaternions to Euler angles
            g_euler = R.from_quat(g_quat).as_euler('xyz')
            s_euler = R.from_quat(s_quat).as_euler('xyz')
            
            orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
            orientation_errors.append(orientation_error)

            # Calculate trace of process noise covariance matrix
            trace_covariance = np.trace(process_noise_covariance_matrices[i])
            trace_covariances.append(trace_covariance)

            # Calculate NEES
            try:
                inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
                e_x_i = e_x_batch.iloc[i].values[:15]
                NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
            except np.linalg.LinAlgError:
                NEES = np.inf

            NEES_values.append(NEES)

        # Calculate averages for the batch
        avg_NEES = np.mean(NEES_values)
        avg_position_error = np.sqrt(np.mean(position_errors))
        avg_orientation_error = np.mean(orientation_errors)
        avg_trace_covariance = np.mean(trace_covariances)

        # Weighted objective function
        objective_score = (
            weight_NEES * avg_NEES +
            weight_RPE * avg_position_error +
            weight_RRE * avg_orientation_error +
            weight_trace * avg_trace_covariance
        )

        # Append the averages to the respective lists
        average_NEES_values.append(avg_NEES)
        average_position_errors.append(avg_position_error)
        average_orientation_errors.append(avg_orientation_error)
        average_trace_covariances.append(avg_trace_covariance)

        # Plotting
        batch_num = start_idx // batch_size
        axs[0].plot(batch_num, avg_NEES, marker='o', color='b')
        axs[1].plot(batch_num, avg_position_error, marker='o', color='g')
        axs[2].plot(batch_num, avg_orientation_error, marker='o', color='r')
        axs[3].plot(batch_num, avg_trace_covariance, marker='o', color='c')

        print(f"Batch {batch_num}: NEES={avg_NEES}, RPE={avg_position_error}, RRE={avg_orientation_error}, Trace={avg_trace_covariance}, Objective={objective_score}")

    else:
        print(f"Columns 'ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z' not found in x_est_batch. Skipping batch {start_idx}.")
        continue  # Skip this iteration or handle the missing columns appropriately

# Set labels and legends
axs[0].set_xlabel('Batch Number')
axs[0].set_ylabel('NEES')
axs[1].set_xlabel('Batch Number')
axs[1].set_ylabel('Relative Position Error')
axs[2].set_xlabel('Batch Number')
axs[2].set_ylabel('Relative Orientation Error')
axs[3].set_xlabel('Batch Number')
axs[3].set_ylabel('Trace of Process Noise Covariance')

plt.tight_layout()
plt.show()

# Print final objective function value
final_objective = np.mean(objective_score)
print(f"Final objective function value: {final_objective}")
