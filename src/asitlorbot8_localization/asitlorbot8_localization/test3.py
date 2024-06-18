import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# Load data from Excel sheet
data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")

# Total number of rows in the dataset
total_rows = len(data)

# Define the number of rows in each batch
batch_size = 50

# List of columns for ground truth and estimated states
gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
              'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
              'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
              'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
              'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# Initialize lists to store NEES, trace, and objective values for each batch
average_NEES_values = []
trace_values = []
objective_values = []
batch_colors = []

# Define the target NEES value
target_NEES = 24.996

# Collect error data for all batches
all_e_x_batches = []
all_process_noise_covariance_matrices = []

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

    # Calculate error between estimated state and ground truth state for the current batch
    e_x_batch = x_gt_batch - x_est_batch

    # Rename columns in e_x_batch with "error_" prefix
    e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

    # Extract the process noise covariance matrices for the current batch
    process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(actual_batch_size)]

    # Append error data and process noise covariance matrices to the combined lists
    all_e_x_batches.append(e_x_batch)
    all_process_noise_covariance_matrices.extend(process_noise_covariance_matrices)

# Concatenate all error batches into a single DataFrame
all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)

# Define dimension names
dimension_names = [f'covariance_{i}' for i in range(225)]

# Create a Space object with dimension names
space = []
for i, name in enumerate(dimension_names):
    if i == 176:
        # Set different bounds for the 176th element
        space.append(Real(0.000000001, 0.01, name=name))
    else:
        # Default bounds for all other elements
        space.append(Real(0.00000001, 0.01, name=name))

# Define the objective function for Bayesian optimization
@use_named_args(space)
def objective_function(**params):
    covariance_values = [params[name] for name in dimension_names]
    process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
    try:
        inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
    except np.linalg.LinAlgError:
        return np.inf  # Return a high value if the matrix is singular

    NEES_values = []
    for i in range(len(all_e_x_data)):
        NEES = all_e_x_data.iloc[i].values @ inv_process_noise_covariance_matrix @ all_e_x_data.iloc[i].values.T
        NEES_values.append(NEES)
    avg_NEES = np.mean(NEES_values)
    trace = np.trace(process_noise_covariance_matrix)
    # Add penalty term to encourage NEES to be close to the target value
    penalty = abs(avg_NEES - target_NEES)
    return 0.7 * avg_NEES + 0.3 * trace + penalty

# Perform Bayesian optimization to find the optimized process noise covariance matrix using data from all batches
res_gp = gp_minimize(objective_function, dimensions=space, n_calls=50, n_random_starts=10)

# Get the optimized process noise covariance matrix
optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# Print the final optimized process noise covariance matrix
print(f"Final Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")

# Calculate and print average NEES for each batch
for start_idx in range(0, total_rows, batch_size):
    actual_batch_size = min(batch_size, total_rows - start_idx)
    e_x_batch = all_e_x_data[start_idx:start_idx + actual_batch_size]

    NEES_batch = []
    for i in range(actual_batch_size):
        try:
            inv_process_noise_covariance_matrix = np.linalg.inv(optimized_covariance_matrix)
            NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
            NEES_batch.append(NEES)
        except np.linalg.LinAlgError:
            NEES_batch.append(np.nan)

    avg_NEES_batch = np.nanmean(NEES_batch)
    average_NEES_values.append(avg_NEES_batch)
    print(f"Average NEES for Batch {start_idx // batch_size}: {avg_NEES_batch}")

# Plot average NEES values for all batches
plt.figure(figsize=(12, 6))
for i in range(len(average_NEES_values)):
    start_idx = i * batch_size
    if average_NEES_values[i] <= 100:
        plt.plot(start_idx, average_NEES_values[i], marker='o', linestyle='-', label=f'Batch {i}', color=np.random.rand(3,))
plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
plt.xlabel('Batch')
plt.ylabel('Average NEES')
plt.title('Average NEES Values for All Batches')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt

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

# # Initialize lists to store NEES, trace, and objective values for each batch
# average_NEES_values = []
# trace_values = []
# objective_values = []
# batch_colors = []

# # Define the target NEES value
# target_NEES = 24.996

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

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(actual_batch_size)]

#     # Initialize lists to store NEES and trace values for each data point in the batch
#     NEES_batch = []
#     trace_batch = []

#     # Calculate NEES and trace for each data point in the batch
#     for i in range(actual_batch_size):
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             trace = np.trace(process_noise_covariance_matrices[i])
#             NEES_batch.append(NEES)
#             trace_batch.append(trace)
#         except np.linalg.LinAlgError:
#             print(f"Skipping singular matrix at index {start_idx + i}")
#             NEES_batch.append(np.nan)
#             trace_batch.append(np.nan)

#     # Calculate the average NEES and trace for the batch
#     avg_NEES_batch = np.nanmean(NEES_batch)
#     avg_trace_batch = np.nanmean(trace_batch)

#     # Skip batches where avg_NEES_batch or avg_trace_batch is NaN
#     if np.isnan(avg_NEES_batch) or np.isnan(avg_trace_batch):
#         print(f"Skipping batch {start_idx} due to NaN values in NEES or trace")
#         continue

#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * avg_trace_batch

#     # Append NEES, trace, and objective values to the respective lists
#     average_NEES_values.append(avg_NEES_batch)
#     trace_values.append(avg_trace_batch)
#     objective_values.append(objective_value)
#     batch_colors.append(np.random.rand(3,))  # Store a random color for each batch

#     # Print the average NEES value for the current batch
#     print(f"Average NEES for Batch {start_idx // batch_size}: {avg_NEES_batch}")

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     space = []
#     for i, name in enumerate(dimension_names):
#         if i == 176:
#             # Set different bounds for the 176th element
#             space.append(Real(0.000000001, 0.01, name=name))
#         else:
#             # Default bounds for all other elements
#             space.append(Real(0.00000001, 0.01, name=name))

#     # Define the objective function for Bayesian optimization
#     @use_named_args(space)
#     def objective_function(**params):
#         covariance_values = [params[name] for name in dimension_names]
#         process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#         except np.linalg.LinAlgError:
#             return np.inf  # Return a high value if the matrix is singular

#         NEES_values = []
#         for i in range(actual_batch_size):
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             NEES_values.append(NEES)
#         avg_NEES = np.mean(NEES_values)
#         trace = np.trace(process_noise_covariance_matrix)
#         # Add penalty term to encourage NEES to be close to the target value
#         penalty = abs(avg_NEES - target_NEES)
#         return 0.7 * avg_NEES + 0.3 * trace + penalty

#     # Perform Bayesian optimization to find the optimized process noise covariance matrix
#     res_gp = gp_minimize(objective_function, dimensions=space, n_calls=10, n_random_starts=5)

#     # Get the optimized process noise covariance matrix
#     optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

#     # Print the optimized process noise covariance matrix for the current batch
#     print(f"Optimized Process Noise Covariance Matrix for Batch {start_idx // batch_size}:\n{optimized_covariance_matrix}")

# # Filter out NEES values greater than 100 for plotting
# filtered_NEES_values = []
# filtered_data_points = []
# for i, batch_NEES in enumerate(average_NEES_values):
#     if batch_NEES <= 100:
#         filtered_NEES_values.append(batch_NEES)
#         filtered_data_points.append(i * batch_size)

# # Plot average NEES values for all batches on a single figure with different colors for each batch
# plt.figure(figsize=(12, 6))
# for i in range(len(average_NEES_values)):
#     start_idx = i * batch_size
#     if average_NEES_values[i] <= 100:
#         plt.plot(start_idx, average_NEES_values[i], marker='o', linestyle='-', label=f'Batch {i}', color=batch_colors[i])
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch')
# plt.ylabel('Average NEES')
# plt.title('Average NEES Values for All Batches')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(f"Final Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")

# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt

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

# # Initialize lists to store errors and process noise covariance matrices for all data points
# all_errors = []
# all_process_noise_covariances = []
# all_batch_indices = []

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

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(actual_batch_size)]

#     # Store errors and process noise covariance matrices
#     all_errors.extend(e_x_batch.values)
#     all_process_noise_covariances.extend(process_noise_covariance_matrices)
#     all_batch_indices.extend([start_idx + i for i in range(actual_batch_size)])

# # Define dimension names
# dimension_names = [f'covariance_{i}' for i in range(225)]

# # Create a Space object with dimension names
# space = []
# for i, name in enumerate(dimension_names):
#     if i == 176:
#         # Set different bounds for the 176th element
#         space.append(Real(0.000000001, 0.00000001, name=name))
#     else:
#         # Default bounds for all other elements
#         space.append(Real(0.00000001, 0.01, name=name))

# # Define the objective function for Bayesian optimization
# @use_named_args(space)
# def objective_function(**params):
#     covariance_values = [params[name] for name in dimension_names]
#     process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#     try:
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#     except np.linalg.LinAlgError:
#         return np.inf  # Return a high value if the matrix is singular

#     NEES_values = []
#     for error in all_errors:
#         NEES = error @ inv_process_noise_covariance_matrix @ error.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
#     # Add penalty term to encourage NEES to be close to the target value
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.7 * avg_NEES + 0.3 * trace + penalty

# # Perform Bayesian optimization to find the optimized process noise covariance matrix
# res_gp = gp_minimize(objective_function, dimensions=space, n_calls=10, n_random_starts=5)

# # Get the optimized process noise covariance matrix
# optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# # Calculate NEES values using the optimized process noise covariance matrix
# inv_optimized_covariance_matrix = np.linalg.inv(optimized_covariance_matrix)
# NEES_values = [error @ inv_optimized_covariance_matrix @ error.T for error in all_errors]

# # Plot NEES values for all data points with different colors for each batch
# plt.figure(figsize=(12, 6))
# unique_batches = np.unique([i // batch_size for i in all_batch_indices])
# batch_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batches)))

# for batch_idx in unique_batches:
#     batch_data_points = [i for i in range(len(all_batch_indices)) if all_batch_indices[i] // batch_size == batch_idx]
#     batch_NEES_values = [NEES_values[i] for i in batch_data_points]
#     plt.plot(batch_data_points, batch_NEES_values, marker='o', linestyle='-', label=f'Batch {batch_idx}', color=batch_colors[batch_idx])

# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Data Point')
# plt.ylabel('NEES')
# plt.title('NEES Values for All Data Points')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(f"Final Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")




# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# from skopt.space import Space
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

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

# # Initialize lists to store NEES, trace, and objective values for each batch
# NEES_values = []
# trace_values = []
# objective_values = []
# avg_NEES_values = []

# # Define the target NEES value
# target_NEES = 9.488

# # Initialize lists to store all NEES values and their corresponding data points
# all_data_points = []

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

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(actual_batch_size)]

#     # Initialize lists to store NEES and trace values for each data point in the batch
#     NEES_batch = []
#     trace_batch = []

#     # Calculate NEES and trace for each data point in the batch
#     for i in range(actual_batch_size):
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             trace = np.trace(process_noise_covariance_matrices[i])
#             NEES_batch.append(NEES)
#             trace_batch.append(trace)
#         except np.linalg.LinAlgError:
#             print(f"Skipping singular matrix at index {start_idx + i}")
#             NEES_batch.append(np.nan)
#             trace_batch.append(np.nan)
    
#     # Filter out NEES values greater than 100
#     filtered_NEES_batch = [ne for ne in NEES_batch if ne <= 100]
    
#     # Collect all NEES values and their corresponding data points
#     all_data_points.extend(range(start_idx, start_idx + actual_batch_size))

#     # Calculate the average NEES and trace for the batch
#     avg_NEES_batch = np.nanmean(filtered_NEES_batch)
#     avg_trace_batch = np.nanmean(trace_batch)

#     # Skip batches where avg_NEES_batch or avg_trace_batch is NaN
#     if np.isnan(avg_NEES_batch) or np.isnan(avg_trace_batch):
#         print(f"Skipping batch {start_idx} due to NaN values in NEES or trace")
#         continue

#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * avg_trace_batch

#     # Append NEES, trace, and objective values to the respective lists
#     NEES_values.append(filtered_NEES_batch)
#     trace_values.append(trace_batch)
#     objective_values.append(objective_value)
#     avg_NEES_values.append(avg_NEES_batch)

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     space = [Real(0.000000001, 1.0, name=name) for name in dimension_names]

#     # Define the objective function for Bayesian optimization
#     @use_named_args(space)
#     def objective_function(**params):
#         covariance_values = [params[name] for name in dimension_names]
#         process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#         except np.linalg.LinAlgError:
#             return np.inf  # Return a high value if the matrix is singular

#         NEES_values = []
#         for i in range(actual_batch_size):
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             NEES_values.append(NEES)
#         avg_NEES = np.mean(NEES_values)
#         trace = np.trace(process_noise_covariance_matrix)
#         # Add penalty term to encourage NEES to be close to the target value
#         penalty = abs(avg_NEES - target_NEES)
#         return 0.7 * avg_NEES + 0.3 * trace + penalty

#     # Perform Bayesian optimization to find the optimized process noise covariance matrix
#     res_gp = gp_minimize(objective_function, dimensions=space, n_calls=10, n_random_starts=5)

#     # Get the optimized process noise covariance matrix
#     optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# # Plot the average NEES for each batch
# plt.figure(figsize=(12, 6))
# plt.plot(range(len(avg_NEES_values)), avg_NEES_values, marker='o', linestyle='-', label='Average NEES per Batch')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(f"Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")



# #!/usr/bin/env python3
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

#     # Loop over data in batches of 50
#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         # Extract process noise covariance values from DataFrame
#         process_noise_covariance_1d = data.iloc[0, 40:].values
        
#         # Reshape process_noise_covariance_1d to a 15x15 matrix
#         process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#         # Calculate trace of the process noise covariance matrix
#         trace = np.trace(process_noise_covariance_matrix)
#         # print(process_noise_covariance_matrix)
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
#     weight_trace = 0.3
#     weight_nees = 0.1
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0 * avg_rpe_translation + 
#                        0 * avg_rpe_rotation + 
#                        0.7 * objective_value_nees +
#                         objective_value_trace)  # Add trace term
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)  # Print trace term
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value, objective_value_nees

# # Define the bounds for the process noise covariance matrix
# bounds = [(0.001, 1)] * 225  # Assuming a 15x15 matrix
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

# # Perform optimization
# res = dummy_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

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
# plt.axhline(y=9.488, color='r', linestyle='--', label='Target NEES (9.488)')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()
#############################################################################################################
#!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_1d):
#     # Reshape process_noise_covariance_1d to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))

#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0

#     # Loop over data in batches of 50
#     num_batches = len(data) // 50
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

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

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1
    
#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches
    
#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [Real(0.001, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.00002, 0.9)
# bounds[96] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[112] = (0.00002, 0.9)  # Adjusting for zero-based index
# bounds[176] = (0.0000000002, 0.02)  # Adjusting for zero-based index
# bounds[192] = (0.00001, 0.1)
# # Store NEES values for plotting
# nees_values = []

# # Modified objective function to capture NEES values
# def modified_objective_function(x):
#     obj_value = objective_function(x)
#     nees_values.append(obj_value)
#     return obj_value

# # Perform optimization using Bayesian Optimization
# res = gp_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

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
# plt.axhline(y=9.488, color='r', linestyle='--', label='Target NEES (9.488)')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()
#######################################################################################################

#!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlor_ws/data4.xlsx")

# def objective_function(process_noise_covariance_1d):
#     # Reshape process_noise_covariance_1d to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))

#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0

#     # Loop over data in batches of 50
#     num_batches = len(data) // 50
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

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

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]
            
#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1
    
#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches
    
#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [Real(0.001, 1)] * 225  # Assuming a 15x15 matrix

# # Store NEES values for plotting
# nees_values = []

# # Modified objective function to capture NEES values
# def modified_objective_function(x):
#     obj_value = objective_function(x)
#     nees_values.append(obj_value)
#     return obj_value

# # Perform optimization using Bayesian Optimization
# res = gp_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

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
# plt.axhline(y=9.488, color='r', linestyle='--', label='Target NEES (9.488)')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()

# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# def objective_function(process_noise_covariance_1d):
#     # Reshape process_noise_covariance_1d to a 15x15 matrix
#     # process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#     # print(process_noise_covariance_matrix)
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0

#     # Loop over data in batches of 50
#     num_batches = len(data) // 50
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50
#         process_noise_covariance_1d = data.iloc[i, 40:].values
#         process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#         # print(process_noise_covariance_matrix)
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

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]

#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)

#             # print(f"Batch {i}, Sample {j} - GT Euler: {gt_euler}, EST Euler: {est_euler}")  # Debugging line
            
#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1
    
#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches
    
#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1
    
#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)
    
#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace
    
#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)
    
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)
#     print("Combined Objective Value: ", objective_value)
    
#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [Real(0.001, 1)] * 225  # Assuming a 15x15 matrix

# # Store NEES values for plotting
# nees_values = []

# # Modified objective function to capture NEES values
# def modified_objective_function(x):
#     obj_value = objective_function(x)
#     nees_values.append(obj_value)
#     return obj_value

# # Perform optimization using Bayesian Optimization
# res = gp_minimize(modified_objective_function, bounds, n_calls=80, random_state=42)

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
# plt.axhline(y=9.488, color='r', linestyle='--', label='Target NEES (9.488)')
# plt.title('Convergence of NEES')
# plt.xlabel('Iteration')
# plt.ylabel('NEES Value')
# plt.grid(True)
# plt.show()
#####################################################################################
# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# def objective_function(process_noise_covariance_1d):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

#     num_batches = len(data) // 50

#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50
#         process_noise_covariance_1d = data.iloc[i, 40:].values
#         process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#         print(process_noise_covariance_matrix)
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

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]

#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)

#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1

#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches

#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1

#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)

#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace

#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)
#     print("Combined Objective Value: ", objective_value)

#     return objective_value

# # Define the bounds for the process noise covariance matrix
# # bounds = [Real(0.001, 1)] * 225  # Assuming a 15x15 matrix
# bounds = [(0.001, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.000000001, 0.1)
# bounds[96] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[112] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[176] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[192] = (0.000000001, 0.1)
# # Perform optimization using Bayesian Optimization
# res = gp_minimize(objective_function, bounds, n_calls=80, random_state=42)

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
################################################################################
# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# def objective_function(process_noise_covariance_1d):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

#     num_batches = len(data) // 50
#     process_noise_covariance_matrix = None  # Initialize outside the loop
#     # print(process_noise_covariance_matrix)
#     for i in range(num_batches):
#         start_idx = i * 50
#         end_idx = (i + 1) * 50

#         process_noise_covariance_1d = data.iloc[start_idx, 40:].values
#         process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#         print(process_noise_covariance_matrix)
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

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]

#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)

#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1

#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches

#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1

#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)

#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace

#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)
#     print("Objective Value RPE Translation: ", avg_rpe_translation)
#     print("Objective Value RPE Rotation: ", avg_rpe_rotation)
#     print("Objective Value NEES: ", objective_value_nees)
#     print("Objective Value Trace: ", objective_value_trace)
#     print("Combined Objective Value: ", objective_value)

#     return objective_value

# # Define the bounds for the process noise covariance matrix
# bounds = [(0.001, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.000000001, 0.1)
# bounds[96] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[112] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[176] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[192] = (0.000000001, 0.1)

# # Perform optimization using Bayesian Optimization
# res = gp_minimize(objective_function, bounds, n_calls=80, random_state=42)

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


# import pandas as pd
# import numpy as np

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# # Total number of rows in the dataset
# total_rows = len(data)

# # Define the number of rows in each batch
# batch_size = 50

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Extract the first row of each batch as process noise covariance
#     process_noise_covariance_1d = data.iloc[start_idx, 40:].values
#     # Reshape to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))

#     # Print process noise covariance matrix for the current batch
#     print(f"Process Noise Covariance Matrix for batch {start_idx // batch_size + 1}:")
#     print(process_noise_covariance_matrix)
#     print()  # Print a blank line for better readability

#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                        'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                        'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][start_idx:start_idx+batch_size]

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                         'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                         'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][start_idx:start_idx+batch_size]

#     # Print ground truth and estimated state variables for the current batch
#     print("Ground Truth State Variables:")
#     print(x_gt_batch)
#     print("Estimated State Variables:")
#     print(x_est_batch)
#     print()  # Print a blank line for better readability



# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from tf_transformations import euler_from_quaternion
# import matplotlib.pyplot as plt

# def objective_function(process_noise_covariance_1d):
#     objective_value_nees = 0
#     objective_value_rpe_translation = 0
#     objective_value_rpe_rotation = 0
#     num_elements_rpe_rotation = 0  # Initialize counter for RPE rotation elements

#     num_batches = len(process_noise_covariance_1d)

#     for i in range(num_batches):
#         process_noise_covariance_matrix = np.array(process_noise_covariance_1d[i]).reshape((15, 15))

#         # Extract ground truth state variables for the current batch
#         x_gt_batch = data[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#                            'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#                            'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']][i*50:(i+1)*50].values

#         # Extract estimated state variables for the current batch
#         x_est_batch = data[['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#                             'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#                             'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']][i*50:(i+1)*50].values

#         # Compute NEES for all variables
#         e_x_batch = x_gt_batch - x_est_batch
#         NEES_batch = np.sum(e_x_batch @ np.linalg.inv(process_noise_covariance_matrix) * e_x_batch, axis=1)
#         avg_NEES_batch = np.mean(NEES_batch)
#         objective_value_nees += np.abs(avg_NEES_batch - 9.488)  # Target NEES value

#         # Compute RPE for translation
#         translation_errors = np.linalg.norm(x_gt_batch[:, :3] - x_est_batch[:, :3], axis=1) / np.linalg.norm(x_gt_batch[:, :3], axis=1)
#         avg_rpe_translation = np.mean(translation_errors)
#         objective_value_rpe_translation += avg_rpe_translation

#         # Compute RPE for rotation using euler angles
#         for j in range(50):
#             # Extract quaternion components for ground truth and estimated
#             gt_quat = [x_gt_batch[j, 5], x_gt_batch[j, 3], x_gt_batch[j, 4], x_gt_batch[j, 6]]  # [w, x, y, z]
#             est_quat = [x_est_batch[j, 5], x_est_batch[j, 3], x_est_batch[j, 4], x_est_batch[j, 6]]  # [w, x, y, z]

#             # Convert quaternions to euler angles
#             gt_euler = euler_from_quaternion(gt_quat)
#             est_euler = euler_from_quaternion(est_quat)

#             # Compute angle difference between the rotations
#             angle_diff = np.linalg.norm(np.array(gt_euler) - np.array(est_euler))
#             objective_value_rpe_rotation += angle_diff
#             num_elements_rpe_rotation += 1

#     avg_rpe_translation /= num_batches
#     avg_rpe_rotation = objective_value_rpe_rotation / num_elements_rpe_rotation
#     objective_value_nees /= num_batches

#     # Weightage for trace term and NEES
#     weight_trace = 0.3
#     weight_nees = 0.1

#     # Calculate trace of the process noise covariance matrix
#     trace = np.trace(process_noise_covariance_matrix)

#     # Calculate objective value for trace term
#     objective_value_trace = weight_trace * trace

#     # Combine NEES, trace, and RPE with weights
#     objective_value = (0.3 * avg_rpe_translation + 
#                        0.3 * avg_rpe_rotation + 
#                        0.4 * objective_value_nees + 
#                        objective_value_trace)

#     return objective_value

# # Load data from Excel sheet
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# # Initialize a list to store the process noise covariance matrices for each batch
# process_noise_covariance_matrices = []

# # Loop over the data in steps of 50 to extract covariance matrices for each batch
# for start_idx in range(0, len(data), 50):
#     process_noise_covariance_1d = data.iloc[start_idx, 40:].values
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))
#     process_noise_covariance_matrices.append(process_noise_covariance_matrix)

# # Define the bounds for the process noise covariance matrix
# bounds = [(0.001, 1)] * 225  # Assuming a 15x15 matrix
# bounds[80] = (0.000000001, 0.1)
# bounds[96] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[112] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[176] = (0.000000001, 0.1)  # Adjusting for zero-based index
# bounds[192] = (0.000000001, 0.1)

# # Perform optimization using Bayesian Optimization
# res = gp_minimize(objective_function, bounds, n_calls=80, random_state=42)

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

# Final Optimized Process Noise Covariance Matrix:
# [[9.76561675e-03 3.59449594e-03 4.83086807e-03 6.31445667e-03
#   2.35471363e-05 4.50881698e-03 3.55131169e-03 9.33999253e-03
#   6.67386404e-03 3.24668556e-03 9.32719431e-03 4.93411225e-03
#   1.62949782e-03 3.84899470e-03 3.45863198e-03]
#  [6.85334554e-03 7.01274304e-03 5.49565338e-03 1.60810293e-03
#   4.54175287e-03 8.07050090e-03 3.66258687e-03 6.93446698e-03
#   9.51827002e-03 7.95465424e-03 7.28749955e-03 2.60391466e-03
#   3.17894681e-03 1.16564792e-03 5.80834932e-03]
#  [7.19547471e-03 2.47906469e-03 8.32962675e-03 6.37911194e-03
#   5.65054097e-03 3.67223665e-03 2.55790680e-03 7.06390271e-03
#   2.53248115e-04 7.12901761e-03 1.39298635e-03 8.80018176e-03
#   8.15731786e-03 5.06541802e-04 2.07249997e-03]
#  [1.70125833e-03 4.87082455e-03 1.37805155e-03 9.09497929e-03
#   6.05331199e-03 7.82227337e-03 4.84222164e-03 7.31650477e-03
#   3.72952169e-03 5.74823942e-03 7.56214159e-03 5.04831245e-04
#   5.65293745e-03 7.51186699e-03 9.34646839e-03]
#  [4.76170759e-03 2.46487012e-03 7.41189725e-03 4.09596576e-03
#   7.10264011e-04 3.76192285e-03 6.46508614e-03 7.08126145e-04
#   4.05471394e-03 2.00270612e-04 6.29026854e-03 6.95234793e-03
#   3.53026019e-03 8.16717096e-03 2.91664108e-03]
#  [3.01548410e-03 9.72768042e-04 1.50707532e-03 1.09246191e-03
#   5.60506705e-03 9.98598975e-03 8.22410386e-03 4.48062157e-03
#   9.89546050e-03 1.89656709e-03 4.63858509e-03 6.23997050e-03
#   7.75164005e-03 6.23426489e-03 8.72127397e-03]
#  [5.49470883e-03 6.53094396e-03 3.52580521e-03 4.45611289e-03
#   6.21804939e-03 9.49712320e-03 9.54016843e-03 1.46597944e-03
#   2.26729603e-03 1.84947836e-04 1.51099450e-04 7.91616433e-03
#   4.12040050e-03 7.55646158e-03 9.00150196e-03]
#  [5.11736186e-03 6.78103495e-03 3.33532806e-04 6.37173239e-03
#   2.85469752e-03 2.15268914e-03 9.52226697e-03 2.37383312e-03
#   4.99419282e-03 4.67629877e-03 6.38680410e-03 6.82262780e-03
#   4.81114424e-03 2.20520668e-03 5.32789152e-03]
#  [5.83071015e-03 4.13515610e-03 6.64286112e-04 8.95144506e-03
#   7.91939853e-04 3.94916818e-04 4.97647786e-03 4.51442548e-03
#   2.47061938e-04 2.10071095e-03 4.79892247e-03 8.89233012e-03
#   9.11833176e-03 1.80227540e-03 4.07335604e-03]
#  [8.08935769e-03 2.75114731e-03 3.37319329e-03 4.35214981e-03
#   9.49073799e-03 2.35178094e-03 4.13649681e-03 8.09161389e-04
#   6.16177600e-03 3.68228748e-03 7.90820623e-03 2.44217496e-03
#   1.60338762e-04 5.31317655e-03 1.95446623e-03]
#  [2.80612200e-03 3.16043185e-03 4.30036419e-03 7.05775496e-03
#   5.56505936e-03 3.31833407e-03 5.49526400e-03 2.79775296e-03
#   5.02214413e-03 9.10344109e-03 5.07857668e-03 5.93395399e-03
#   2.60949636e-03 5.10857827e-03 2.93642035e-03]
#  [3.99439359e-03 3.86488012e-03 8.68109794e-04 3.33628672e-03
#   4.76967829e-03 5.47357916e-03 2.09060281e-03 2.03648795e-03
#   7.63217271e-03 7.41213760e-03 1.42146723e-03 7.01593688e-09
#   7.98973399e-03 2.56582861e-03 4.20360902e-03]
#  [9.86965827e-03 6.10932393e-03 4.51563437e-03 6.24734518e-03
#   6.59886833e-03 4.89247369e-03 8.26990112e-03 4.35391076e-04
#   1.16659880e-05 3.63478694e-03 9.60506731e-03 3.50551430e-03
#   6.19493034e-04 9.27032398e-03 4.73072445e-03]
#  [6.41006287e-03 2.65600909e-03 2.17572094e-03 7.28317829e-03
#   4.25602244e-03 5.46554945e-03 9.70592070e-03 5.44311450e-03
#   2.87781553e-03 9.98516504e-03 5.64735885e-03 4.85025209e-03
#   4.89313401e-03 1.90013436e-03 7.78826015e-03]
#  [1.08593355e-03 5.72448696e-03 4.88447564e-03 4.45605362e-03
#   8.88092866e-03 7.44312276e-03 6.63142650e-03 8.51277538e-03
#   9.95719772e-04 1.36562740e-03 3.24563642e-03 9.32791247e-04
#   4.65311959e-03 1.23809321e-03 7.05910277e-03]]


