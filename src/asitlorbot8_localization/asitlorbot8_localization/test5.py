import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

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

    # Append error data to the combined lists
    all_e_x_batches.append(e_x_batch)

# Concatenate all error batches into a single DataFrame
all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)

# Define dimension names for the diagonal elements
dimension_names = [f'covariance_{i}' for i in range(15)]

# Define the bounds for the diagonal elements
bounds = [Real(0.00000001, 0.09, name=name) for name in dimension_names]

# Define a function to calculate RMSE for translation and rotation
def calculate_rmse(x_gt_batch, x_est_batch):
    translation_rmse = np.sqrt(np.mean((x_gt_batch[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values - 
                                        x_est_batch[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values)**2))
    
    rotation_rmse = np.sqrt(np.mean((x_gt_batch[['GT_Roll', 'GT_Pitch', 'GT_Yaw']].values - 
                                     x_est_batch[['GT_Roll', 'GT_Pitch', 'GT_Yaw']].values)**2))
    
    return translation_rmse, rotation_rmse

# Define the objective function for Bayesian optimization
@use_named_args(bounds)
def objective_function(**params):
    diagonal_values = [params[name] for name in dimension_names]
    process_noise_covariance_matrix = np.diag(diagonal_values)
    try:
        inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
    except np.linalg.LinAlgError:
        return np.inf  # Return a high value if the matrix is singular

    NEES_values = []
    translation_rmse_values = []
    rotation_rmse_values = []
    
    for start_idx in range(0, total_rows, batch_size):
        actual_batch_size = min(batch_size, total_rows - start_idx)
        x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        x_est_batch.columns = gt_columns
        e_x_batch = x_gt_batch - x_est_batch
        e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]

        # Calculate NEES
        for i in range(actual_batch_size):
            NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
            NEES_values.append(NEES)
        
        # Calculate RMSE
        translation_rmse, rotation_rmse = calculate_rmse(x_gt_batch, x_est_batch)
        translation_rmse_values.append(translation_rmse)
        rotation_rmse_values.append(rotation_rmse)

    avg_NEES = np.mean(NEES_values)
    avg_translation_rmse = np.mean(translation_rmse_values)
    avg_rotation_rmse = np.mean(rotation_rmse_values)
    trace = np.trace(process_noise_covariance_matrix)

    # Combine NEES, trace, and RMSE into the objective function
    penalty = abs(avg_NEES - target_NEES)
    return 0.5 * avg_NEES + 0.2 * trace + 0.2 * avg_translation_rmse + 0.1 * avg_rotation_rmse + penalty

# Perform Bayesian optimization to find the optimized diagonal elements
res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=50, n_random_starts=10, random_state=42)

# Get the optimized diagonal elements from the best solution found by Bayesian optimization
optimized_diagonal_values = res_gp.x
optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)

# Print the optimized process noise covariance matrix
print("Optimized Process Noise Covariance Matrix:")
print(optimized_process_noise_covariance_matrix)

# Calculate the average NEES for each batch using the optimized process noise covariance matrix
inv_optimized_covariance_matrix = np.linalg.inv(optimized_process_noise_covariance_matrix)
average_NEES_values = []
for start_idx in range(0, total_rows, batch_size):
    actual_batch_size = min(batch_size, total_rows - start_idx)
    x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
    x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
    x_est_batch.columns = gt_columns
    e_x_batch = x_gt_batch - x_est_batch
    e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]

    NEES_values = []
    for i in range(actual_batch_size):
        NEES = e_x_batch.iloc[i].values @ inv_optimized_covariance_matrix @ e_x_batch.iloc[i].values.T
        NEES_values.append(NEES)
    avg_NEES = np.mean(NEES_values)
    average_NEES_values.append(avg_NEES)

# Plot average NEES for each batch
plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
plt.xlabel('Batch Number')
plt.ylabel('Average NEES')
plt.title('Average NEES for Each Batch')
plt.legend()
plt.grid()
plt.show()

# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt

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

# # Initialize lists to store NEES, trace, and objective values for each batch
# average_NEES_values = []
# trace_values = []
# objective_values = []
# batch_colors = []

# # Define the target NEES value
# target_NEES = 24.996

# # Collect error data for all batches
# all_e_x_batches = []

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

#     # Append error data to the combined lists
#     all_e_x_batches.append(e_x_batch)

# # Concatenate all error batches into a single DataFrame
# all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)

# # Define dimension names for the diagonal elements
# dimension_names = [f'covariance_{i}' for i in range(15)]

# # Define the bounds for the diagonal elements
# bounds = [Real(0.00000001, 0.09, name=name) for name in dimension_names]

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
#     for i in range(len(all_e_x_data)):
#         NEES = all_e_x_data.iloc[i].values @ inv_process_noise_covariance_matrix @ all_e_x_data.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
#     # Add penalty term to encourage NEES to be close to the target value
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.7 * avg_NEES + 0.3 * trace + penalty

# # Perform Bayesian optimization to find the optimized diagonal elements
# res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=20, n_initial_points=20, random_state=42)

# # Get the optimized diagonal elements from the best solution found by Bayesian optimization
# optimized_diagonal_values = res_gp.x
# optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)

# # Print the optimized process noise covariance matrix
# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Calculate the average NEES for each batch using the optimized process noise covariance matrix
# inv_optimized_covariance_matrix = np.linalg.inv(optimized_process_noise_covariance_matrix)
# average_NEES_values = []
# for start_idx in range(0, total_rows, batch_size):
#     actual_batch_size = min(batch_size, total_rows - start_idx)
#     x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#     x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#     x_est_batch.columns = gt_columns
#     e_x_batch = x_gt_batch - x_est_batch
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]

#     NEES_values = []
#     for i in range(actual_batch_size):
#         NEES = e_x_batch.iloc[i].values @ inv_optimized_covariance_matrix @ e_x_batch.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     average_NEES_values.append(avg_NEES)

# # Plot average NEES for each batch
# plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch')
# plt.legend()
# plt.grid()
# plt.show()




# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt

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

# # Initialize lists to store NEES, trace, and objective values for each batch
# average_NEES_values = []
# trace_values = []
# objective_values = []
# batch_colors = []

# # Define the target NEES value
# target_NEES = 24.996

# # Collect error data for all batches
# all_e_x_batches = []
# all_process_noise_covariance_matrices = []

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

#     # Append error data and process noise covariance matrices to the combined lists
#     all_e_x_batches.append(e_x_batch)
#     all_process_noise_covariance_matrices.extend(process_noise_covariance_matrices)

# # Concatenate all error batches into a single DataFrame
# all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)

# # Define dimension names
# dimension_names = [f'covariance_{i}' for i in range(225)]

# # Create a Space object with dimension names
# space = []
# for i, name in enumerate(dimension_names):
#     if i == 176:
#         # Set different bounds for the 176th element
#         space.append(Real(0.000000001, 0.01, name=name))
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
#     for i in range(len(all_e_x_data)):
#         NEES = all_e_x_data.iloc[i].values @ inv_process_noise_covariance_matrix @ all_e_x_data.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
#     # Add penalty term to encourage NEES to be close to the target value
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.7 * avg_NEES + 0.3 * trace + penalty

# # Perform Bayesian optimization to find the optimized process noise covariance matrix using data from all batches
# res_gp = gp_minimize(objective_function, dimensions=space, n_calls=50, n_random_starts=10)

# # Get the optimized process noise covariance matrix
# optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# # Print the final optimized process noise covariance matrix
# print(f"Final Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")

# # Calculate and print average NEES for each batch
# for start_idx in range(0, total_rows, batch_size):
#     actual_batch_size = min(batch_size, total_rows - start_idx)
#     e_x_batch = all_e_x_data[start_idx:start_idx + actual_batch_size]

#     NEES_batch = []
#     for i in range(actual_batch_size):
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(optimized_covariance_matrix)
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             NEES_batch.append(NEES)
#         except np.linalg.LinAlgError:
#             NEES_batch.append(np.nan)

#     avg_NEES_batch = np.nanmean(NEES_batch)
#     average_NEES_values.append(avg_NEES_batch)
#     print(f"Average NEES for Batch {start_idx // batch_size}: {avg_NEES_batch}")

# # Plot average NEES values for all batches
# plt.figure(figsize=(12, 6))
# for i in range(len(average_NEES_values)):
#     start_idx = i * batch_size
#     plt.plot(start_idx, average_NEES_values[i], marker='o', linestyle='-', label=f'Batch {i}', color=np.random.rand(3,))
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch')
# plt.ylabel('Average NEES')
# plt.title('Average NEES Values for All Batches')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# import matplotlib.pyplot as plt

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

# # Initialize lists to store NEES, trace, and objective values for each batch
# average_NEES_values = []
# trace_values = []
# objective_values = []
# batch_colors = []

# # Define the target NEES value
# target_NEES = 24.996

# # Collect error data for all batches
# all_e_x_batches = []
# all_process_noise_covariance_matrices = []

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

#     # Append error data and process noise covariance matrices to the combined lists
#     all_e_x_batches.append(e_x_batch)
#     all_process_noise_covariance_matrices.extend(process_noise_covariance_matrices)

# # Concatenate all error batches into a single DataFrame
# all_e_x_data = pd.concat(all_e_x_batches, ignore_index=True)

# # Define dimension names
# dimension_names = [f'covariance_{i}' for i in range(225)]

# # Create a Space object with dimension names
# space = []
# for i, name in enumerate(dimension_names):
#     if i == 176:
#         # Set different bounds for the 176th element
#         space.append(Real(0.000000001, 0.01, name=name))
#     else:
#         # Default bounds for all other elements
#         space.append(Real(0.00000001, 0.1, name=name))

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
#     for i in range(len(all_e_x_data)):
#         NEES = all_e_x_data.iloc[i].values @ inv_process_noise_covariance_matrix @ all_e_x_data.iloc[i].values.T
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
#     # Add penalty term to encourage NEES to be close to the target value
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.7 * avg_NEES + 0.3 * trace + penalty

# # Perform Bayesian optimization to find the optimized process noise covariance matrix using data from all batches
# res_gp = gp_minimize(objective_function, dimensions=space, n_calls=50, n_random_starts=10)

# # Get the optimized process noise covariance matrix
# optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# # Print the final optimized process noise covariance matrix
# print(f"Final Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")

# # Calculate and print average NEES for each batch
# for start_idx in range(0, total_rows, batch_size):
#     actual_batch_size = min(batch_size, total_rows - start_idx)
#     e_x_batch = all_e_x_data[start_idx:start_idx + actual_batch_size]

#     NEES_batch = []
#     for i in range(actual_batch_size):
#         try:
#             inv_process_noise_covariance_matrix = np.linalg.inv(optimized_covariance_matrix)
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             if 0 <= NEES <= 100:  # Exclude NEES values below 0 and above 100
#                 NEES_batch.append(NEES)
#         except np.linalg.LinAlgError:
#             NEES_batch.append(np.nan)

#     avg_NEES_batch = np.nanmean(NEES_batch)
#     average_NEES_values.append(avg_NEES_batch)
#     print(f"Average NEES for Batch {start_idx // batch_size}: {avg_NEES_batch}")

# # Plot average NEES values for all batches
# plt.figure(figsize=(12, 6))
# for i in range(len(average_NEES_values)):
#     start_idx = i * batch_size
#     if average_NEES_values[i] <= 100:
#         plt.plot(start_idx, average_NEES_values[i], marker='o', linestyle='-', label=f'Batch {i}', color=np.random.rand(3,))
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch')
# plt.ylabel('Average NEES')
# plt.title('Average NEES Values for All Batches')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
