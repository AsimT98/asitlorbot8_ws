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
        space.append(Real(0.0000000001, 0.053, name=name))
    elif i == 32:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.1, name=name))
    elif i == 48:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.3, name=name))
    elif i == 64:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.03, name=name))
    elif i == 80:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.06, name=name))
    elif i == 96 or 112:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.025, name=name))
    elif i == 128:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.04, name=name))
    elif i == 144:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.06, name=name))
    elif i == 160:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.015, name=name))
    elif i == 192 or 208 or 224:
        # Set different bounds for the 176th element
        space.append(Real(0.00000001, 0.015, name=name))
    elif i == 1 or 16:
        # Default bounds for all other elements
        space.append(Real(0.00000001, 0.5, name=name))

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
    plt.plot(start_idx, average_NEES_values[i], marker='o', linestyle='-', label=f'Batch {i}', color=np.random.rand(3,))
plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
plt.xlabel('Batch')
plt.ylabel('Average NEES')
plt.title('Average NEES Values for All Batches')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 0.5	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.5	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.03	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.06	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.025	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.025	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.04	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.06	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.015	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.0534946414	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.015	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.015	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.015
# 0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1E-10	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.00000001

# import pandas as pd
# import numpy as np

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

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the first row of each batch as process noise covariance
#     process_noise_covariance_1d = data.iloc[start_idx, 40:].values
#     # Reshape to a 15x15 matrix
#     process_noise_covariance_matrix = np.array(process_noise_covariance_1d).reshape((15, 15))

#     # Calculate NEES for each sample in the batch
#     inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#     NEES_batch = [e_x @ inv_process_noise_covariance_matrix @ e_x.T for e_x in e_x_batch.values]
#     avg_NEES_batch = np.mean(NEES_batch)
    
#     # Calculate the trace of the process noise covariance matrix
#     trace_process_noise_covariance = np.trace(process_noise_covariance_matrix)
    
#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * trace_process_noise_covariance

#     # Print process noise covariance matrix, trace, ground truth, estimated state variables, error, and NEES for the current batch
#     print(f"Process Noise Covariance Matrix for batch {start_idx // batch_size + 1}:")
#     print(process_noise_covariance_matrix)
#     print(f"Trace of Process Noise Covariance Matrix for batch {start_idx // batch_size + 1}: {trace_process_noise_covariance}")
#     print()  # Print a blank line for better readability

#     # print("Ground Truth State Variables:")
#     # print(x_gt_batch)
#     # print("Estimated State Variables:")
#     # print(x_est_batch)
#     # print("Error between Ground Truth and Estimated State Variables:")
#     # print(e_x_batch)
#     print()  # Print a blank line for better readability

#     print(f"NEES for batch {start_idx // batch_size + 1}:")
#     print(NEES_batch)
#     print(f"Average NEES for batch {start_idx // batch_size + 1}: {avg_NEES_batch}")
#     print(f"Objective value combining NEES and trace for batch {start_idx // batch_size + 1}: {objective_value}")
#     print()  # Print a blank line for better readability

# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# from skopt.space import Space

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

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(batch_size)]

#     # Initialize lists to store NEES and trace values for each data point in the batch
#     NEES_batch = []
#     trace_batch = []

#     # Calculate NEES and trace for each data point in the batch
#     for i in range(batch_size):
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#         NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#         trace = np.trace(process_noise_covariance_matrices[i])
#         NEES_batch.append(NEES)
#         trace_batch.append(trace)

#     # Calculate the average NEES and trace for the batch
#     avg_NEES_batch = np.mean(NEES_batch)
#     avg_trace_batch = np.mean(trace_batch)

#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * avg_trace_batch

#     # Append NEES, trace, and objective values to the respective lists
#     NEES_values.append(NEES_batch)
#     trace_values.append(trace_batch)
#     objective_values.append(objective_value)

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     space = [Real(0.000000001, 1.0, name=name) for name in dimension_names]

#     # Define the objective function for Bayesian optimization
#     @use_named_args(space)
#     def objective_function(**params):
#         covariance_values = [params[name] for name in dimension_names]
#         process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#         NEES_values = []
#         for i in range(batch_size):
#             NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#             NEES_values.append(NEES)
#         avg_NEES = np.mean(NEES_values)
#         trace = np.trace(process_noise_covariance_matrix)
#         return 0.7 * avg_NEES + 0.3 * trace

#     # Perform Bayesian optimization to find the optimized process noise covariance matrix
#     res_gp = gp_minimize(objective_function, dimensions=space, n_calls=10, n_random_starts=5)

#     # Get the optimized process noise covariance matrix
#     optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

#     # Print process noise covariance matrices, trace, and NEES for the current batch
#     print(f"Process Noise Covariance Matrices for batch {start_idx // batch_size + 1}:")
#     for i in range(batch_size):
#         print(f"Matrix {i + 1}:")
#         print(process_noise_covariance_matrices[i])
#         print()
#     print(f"Average NEES for batch {start_idx // batch_size + 1}: {avg_NEES_batch}")
#     print(f"Average Trace for batch {start_idx // batch_size + 1}: {avg_trace_batch}")
#     print(f"Objective value for batch {start_idx // batch_size + 1}: {objective_value}")
#     print(f"Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")
#     print()  # Print a blank line for better readability


########
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

# # Define the target NEES value
# target_NEES = 9.488

# # Initialize a list to store NEES plots
# NEES_plots = []

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(batch_size)]

#     # Initialize lists to store NEES and trace values for each data point in the batch
#     NEES_batch = []
#     trace_batch = []

#     # Calculate NEES and trace for each data point in the batch
#     for i in range(batch_size):
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#         NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#         trace = np.trace(process_noise_covariance_matrices[i])
#         NEES_batch.append(NEES)
#         trace_batch.append(trace)

#     # Calculate the average NEES and trace for the batch
#     avg_NEES_batch = np.mean(NEES_batch)
#     avg_trace_batch = np.mean(trace_batch)

#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * avg_trace_batch

#     # Append NEES, trace, and objective values to the respective lists
#     NEES_values.append(NEES_batch)
#     trace_values.append(trace_batch)
#     objective_values.append(objective_value)

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     space = [Real(0.000000001, 1.0, name=name) for name in dimension_names]

#     # Define the objective function for Bayesian optimization
#     @use_named_args(space)
#     def objective_function(**params):
#         covariance_values = [params[name] for name in dimension_names]
#         process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#         NEES_values = []
#         for i in range(batch_size):
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

#     # Plot NEES values for the current batch
#     # Plot NEES values for the current batch
# plt.figure(figsize=(10, 5))
# plt.plot(range(start_idx, start_idx + batch_size), NEES_batch, marker='o', linestyle='-')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Data Point')
# plt.ylabel('NEES')
# plt.title(f'NEES Values for Batch {start_idx // batch_size + 1}')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# NEES_plots.append(plt)  # Append the plot to the list of NEES plots
# plt.show()
# plt.close()  # Close the current figure after displaying it


#     # Print process noise covariance matrices, trace, and NEES for the current batch
#     # print(f"Process Noise Covariance Matrices for batch {start_idx // batch_size + 1}:")
#     # for i in range(batch_size):
#     #     print(f"Matrix {i + 1}:")
#     #     print(process_noise_covariance_matrices[i])
#     #     print()
#     # print(f"Average NEES for batch {start_idx // batch_size + 1}: {avg_NEES_batch}")
#     # print(f"Average Trace for batch {start_idx // batch_size + 1}: {avg_trace_batch}")
#     # print(f"Objective value for batch {start_idx // batch_size + 1}: {objective_value}")
# print(f"Optimized Process Noise Covariance Matrix:\n{optimized_covariance_matrix}")
# plt.figure(figsize=(12, 8))
# for i, nees_plot in enumerate(NEES_plots, start=1):
#     plt.subplot(len(NEES_plots)//2, 2, i)
#     nees_plot.plot()
# plt.tight_layout()
# plt.show()

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

# # Define the target NEES value
# target_NEES = 9.488

# # Initialize a list to store NEES plots
# NEES_plots = []

# # Loop over the data in steps of batch_size
# for start_idx in range(0, total_rows, batch_size):
#     # Extract ground truth state variables for the current batch
#     x_gt_batch = data[gt_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Extract estimated state variables for the current batch
#     x_est_batch = data[et_columns][start_idx:start_idx + batch_size].reset_index(drop=True)

#     # Ensure the columns are correctly aligned by renaming columns of x_est_batch to match x_gt_batch structure
#     x_est_batch.columns = gt_columns

#     # Calculate error between estimated state and ground truth state for the current batch
#     e_x_batch = x_gt_batch - x_est_batch

#     # Rename columns in e_x_batch with "error_" prefix
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15)) for i in range(batch_size)]

#     # Initialize lists to store NEES and trace values for each data point in the batch
#     NEES_batch = []
#     trace_batch = []

#     # Calculate NEES and trace for each data point in the batch
#     for i in range(batch_size):
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#         NEES = e_x_batch.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x_batch.iloc[i].values.T
#         trace = np.trace(process_noise_covariance_matrices[i])
#         NEES_batch.append(NEES)
#         trace_batch.append(trace)

#     # Calculate the average NEES and trace for the batch
#     avg_NEES_batch = np.mean(NEES_batch)
#     avg_trace_batch = np.mean(trace_batch)

#     # Calculate the objective function value combining NEES and trace with weights 0.7 and 0.3 respectively
#     objective_value = 0.7 * avg_NEES_batch + 0.3 * avg_trace_batch

#     # Append NEES, trace, and objective values to the respective lists
#     NEES_values.append(NEES_batch)
#     trace_values.append(trace_batch)
#     objective_values.append(objective_value)

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     space = [Real(0.000000001, 1.0, name=name) for name in dimension_names]

#     # Define the objective function for Bayesian optimization
#     @use_named_args(space)
#     def objective_function(**params):
#         covariance_values = [params[name] for name in dimension_names]
#         process_noise_covariance_matrix = np.array(covariance_values).reshape((15, 15))
#         inv_process_noise_covariance_matrix = np.linalg.inv(process_noise_covariance_matrix)
#         NEES_values = []
#         for i in range(batch_size):
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

#     # Plot NEES values for the current batch
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(start_idx, start_idx + batch_size), NEES_batch, marker='o', linestyle='-')
#     plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
#     plt.xlabel('Data Point')
#     plt.ylabel('NEES')
#     plt.title(f'NEES Values for Batch {start_idx // batch_size + 1}')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     NEES_plots.append(plt)  # Append the plot to the list of NEES plots

#     # Print process noise covariance matrices, trace, and NEES for the current batch
#     print(f"Process Noise Covariance Matrices for batch {start_idx // batch_size + 1}:")
#     for i in range(batch_size):
#         print(f"Matrix {i + 1}:")
#         print(process_noise_covariance_matrices[i])
#         print()
#     print(f"Average NEES for batch {start_idx // batch_size + 1}: {avg_NEES_batch}")
#     print(f"Average Trace for batch {start_idx // batch_size + 1}: {avg_trace_batch}")
#     print(f"Objective value for batch {start_idx // batch_size + 1}: {objective_value}")
#     plt.figure(figsize=(12, 8))
#     for i, nees_plot in enumerate(NEES_plots, start=1):
#         plt.subplot(len(NEES_plots)//2, 2, i)
#         nees_plot
#     plt.tight_layout()
#     plt.show()

###################################################################
# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
# from skopt.space import Space
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
# NEES_values = []
# trace_values = []
# objective_values = []
# batch_colors = []

# # Define the target NEES value
# target_NEES = 24.996

# # Initialize lists to store all NEES values and their corresponding data points
# all_NEES_values = []
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
    
#     # Collect all NEES values and their corresponding data points
#     all_NEES_values.extend(NEES_batch)
#     all_data_points.extend(range(start_idx, start_idx + actual_batch_size))

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
#     NEES_values.append(NEES_batch)
#     trace_values.append(trace_batch)
#     objective_values.append(objective_value)
#     batch_colors.append(np.random.rand(3,))  # Store a random color for each batch
#     print(f"NEES Values for Batch {start_idx // batch_size}: {NEES_batch}")

#     # Define dimension names
#     dimension_names = [f'covariance_{i}' for i in range(225)]

#     # Create a Space object with dimension names
#     # space = [Real(0.000000001, 0.0009, name=name) for name in dimension_names]
#     space = []
#     for i, name in enumerate(dimension_names):
#         if i == 176:
#             # Set different bounds for the 86th element
#             space.append(Real(0.000000001, 0.01, name=name))
#         # elif i == 96:
#         #     # Set different bounds for the 96th element
#         #     space.append(Real(0.000000001, 0.1, name=name))
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

# # Filter out NEES values greater than 100
# filtered_NEES_values = [ne for ne in all_NEES_values if ne <= 100]
# filtered_data_points = [all_data_points[i] for i in range(len(all_NEES_values)) if all_NEES_values[i] <= 100]

# # Plot all NEES values for all batches on a single figure with different colors for each batch
# plt.figure(figsize=(12, 6))
# for i in range(len(NEES_values)):
#     start_idx = i * batch_size
#     end_idx = start_idx + len(NEES_values[i])
#     # Filter out NEES values greater than 100 for plotting
#     batch_NEES = [ne for ne in NEES_values[i] if ne <= 100]
#     batch_indices = [start_idx + j for j in range(len(batch_NEES))]
#     plt.plot(batch_indices, batch_NEES, marker='o', linestyle='-', label=f'Batch {i}', color=batch_colors[i])
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Data Point')
# plt.ylabel('NEES')
# plt.title('NEES Values for All Batches')
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
# data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/data_7.xlsx")

# # Check for NaN values in the data
# if data.isnull().values.any():
#     print("Data contains NaN values. Please clean the data before proceeding.")
#     exit()

# # Total number of rows in the dataset
# total_rows = len(data)

# # List of columns for ground truth and estimated states
# gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Roll', 'GT_Pitch', 'GT_Yaw', 
#               'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
#               'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

# et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Roll', 'ET_Pitch', 'ET_Yaw', 
#               'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
#               'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# # Extract ground truth state variables for the entire dataset
# x_gt = data[gt_columns].reset_index(drop=True)

# # Extract estimated state variables for the entire dataset
# x_est = data[et_columns].reset_index(drop=True)

# # Ensure the columns are correctly aligned by renaming columns of x_est to match x_gt structure
# x_est.columns = gt_columns

# # Calculate error between estimated state and ground truth state for the entire dataset
# e_x = x_gt - x_est

# # Rename columns in e_x with "error_" prefix
# e_x.columns = [f'error_{col[3:]}' for col in gt_columns]  # strip 'GT_' and add 'error_'

# # Extract the process noise covariance matrices for the entire dataset
# process_noise_covariance_matrices = [np.array(data.iloc[i, 40:].values).reshape((15, 15)) for i in range(total_rows)]

# # Define dimension names
# dimension_names = [f'covariance_{i}' for i in range(225)]

# # Create a Space object with dimension names
# space = [Real(0.000000001, 0.009, name=name) for name in dimension_names]

# # Define the target NEES value
# target_NEES = 24.996

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
#     for i in range(total_rows):
#         try:
#             NEES = e_x.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x.iloc[i].values.T
#             NEES_values.append(NEES)
#         except Exception as e:
#             print(f"Error at index {i}: {e}")
#             NEES_values.append(np.nan)
    
#     avg_NEES = np.nanmean(NEES_values)
#     trace = np.trace(process_noise_covariance_matrix)
#     if np.isnan(avg_NEES):
#         return np.inf  # Return a high value if the average NEES is NaN

#     # Add penalty term to encourage NEES to be close to the target value
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.7 * avg_NEES + 0.3 * trace + penalty

# # Perform Bayesian optimization to find the optimized process noise covariance matrix
# res_gp = gp_minimize(objective_function, dimensions=space, n_calls=50, n_random_starts=10)

# # Get the optimized process noise covariance matrix
# optimized_covariance_matrix = np.array(res_gp.x).reshape((15, 15))

# # Print the optimized process noise covariance matrix
# print(f"Optimized Process Noise Covariance Matrix for the Entire Dataset:\n{optimized_covariance_matrix}")

# # Calculate NEES and trace values for the entire dataset using the optimized covariance matrix
# NEES_values = []
# trace_values = []
# for i in range(total_rows):
#     try:
#         inv_process_noise_covariance_matrix = np.linalg.inv(optimized_covariance_matrix)
#         NEES = e_x.iloc[i].values @ inv_process_noise_covariance_matrix @ e_x.iloc[i].values.T
#         trace = np.trace(optimized_covariance_matrix)
#         NEES_values.append(NEES)
#         trace_values.append(trace)
#     except np.linalg.LinAlgError:
#         print(f"Skipping singular matrix at index {i}")
#         NEES_values.append(np.nan)
#         trace_values.append(np.nan)

# # Filter out NEES values greater than 100
# filtered_NEES_values = [ne for ne in NEES_values if ne <= 100]
# filtered_data_points = [i for i in range(total_rows) if NEES_values[i] <= 100]

# # Plot all NEES values for the entire dataset
# plt.figure(figsize=(12, 6))
# plt.plot(filtered_data_points, filtered_NEES_values, marker='o', linestyle='-', label='NEES Values')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Data Point')
# plt.ylabel('NEES')
# plt.title('NEES Values for the Entire Dataset')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




# [[1.00000000e-02 3.23949125e-03 4.14831544e-03 1.00000000e-02
#   5.86338894e-03 1.39142880e-03 1.00000000e-02 1.00000000e-02
#   6.87289929e-04 1.00000000e-02 3.17815737e-03 4.61824218e-03
#   2.38370078e-03 1.00000000e-08 1.00000000e-08]
#  [6.97558293e-04 8.34795519e-03 4.35917671e-03 7.58752106e-03
#   5.06660643e-03 5.55029827e-05 3.81919540e-03 4.79847857e-03
#   8.94388750e-03 1.00000000e-02 9.39029322e-03 1.00000000e-08
#   1.00000000e-02 7.07442294e-03 8.16587506e-03]
#  [1.00000000e-02 2.46542239e-03 1.00000000e-02 1.00000000e-02
#   7.96118437e-03 9.14497212e-03 1.00000000e-08 3.33074392e-03
#   5.26283684e-03 8.01502797e-03 8.91456149e-04 1.00000000e-02
#   2.97703577e-03 1.00000000e-02 1.98236680e-03]
#  [7.68605500e-03 1.00000000e-08 1.00000000e-02 1.00000000e-08
#   1.00000000e-08 4.56021156e-03 4.47139886e-03 5.55999696e-03
#   3.11400619e-04 1.01990451e-03 2.70888609e-03 1.00000000e-08
#   1.22407031e-03 1.53205683e-03 3.15945217e-03]
#  [1.00000000e-02 1.00000000e-08 3.53156401e-03 1.00000000e-08
#   3.07082033e-03 1.00000000e-02 2.41692322e-03 1.00000000e-08
#   7.40707049e-03 3.55028764e-03 6.99183049e-04 1.00000000e-02
#   3.90317043e-03 8.52733999e-03 5.39746768e-03]
#  [9.51326422e-03 1.00000000e-08 6.66291606e-03 1.00000000e-08
#   4.23570198e-03 1.00000000e-08 5.10027745e-03 3.37823672e-03
#   1.00000000e-08 3.27846553e-03 6.28736279e-03 6.83965445e-03
#   7.23527728e-03 1.40195411e-03 1.00000000e-08]
#  [2.23095062e-03 1.00000000e-08 3.60463346e-03 1.00000000e-02
#   1.00000000e-02 4.61801446e-03 5.54900412e-03 8.69635912e-03
#   9.94855447e-03 3.73188005e-03 7.69640964e-03 9.30644548e-05
#   1.00000000e-02 1.00000000e-08 6.22618851e-03]
#  [1.00000000e-08 5.04659801e-03 4.23309749e-03 5.35147953e-03
#   1.00000000e-02 1.09452804e-05 9.01828489e-03 1.00000000e-08
#   2.44081427e-04 9.98926123e-06 4.06932434e-03 5.86557304e-03
#   4.71351826e-03 8.16083777e-03 1.00000000e-08]
#  [2.37960481e-03 3.97936713e-03 4.91767452e-03 2.66927484e-03
#   1.00000000e-08 1.00000000e-08 6.90703171e-03 7.74204029e-04
#   3.94472098e-03 2.86322068e-03 1.00000000e-08 1.00000000e-02
#   7.24890064e-03 5.40744216e-03 9.43930005e-03]
#  [1.00000000e-08 1.00000000e-08 4.32873626e-03 1.00000000e-08
#   6.94491649e-03 5.33319491e-03 5.72837149e-03 5.44833347e-03
#   1.00000000e-02 2.90621106e-03 4.09467481e-03 8.32044655e-03
#   8.66872280e-03 5.83375344e-03 8.52861122e-04]
#  [3.87177698e-03 2.00934338e-03 3.94671298e-03 2.22795062e-03
#   7.57138494e-03 1.00000000e-08 7.57602486e-03 9.43209553e-03
#   3.00288756e-03 2.23307610e-03 3.72471404e-03 3.19434279e-03
#   1.67887759e-03 8.90382335e-03 3.79194837e-03]
#  [1.00000000e-02 1.00000000e-02 1.98590359e-03 2.68205492e-03
#   2.45461816e-03 8.49241391e-03 8.53981610e-03 7.23305153e-03
#   5.63210848e-03 1.84842277e-03 7.76630408e-03 1.90101280e-03
#   2.53359514e-03 7.50420134e-03 5.66662130e-03]
#  [6.35055165e-03 1.00000000e-02 9.64818936e-03 1.00000000e-02
#   1.00000000e-08 4.16057206e-04 6.51456388e-03 1.00000000e-08
#   3.74704587e-03 1.00000000e-08 1.00000000e-02 1.00000000e-02
#   5.78662541e-03 2.15521060e-03 9.00083576e-03]
#  [3.64988062e-03 9.10358874e-03 1.13569894e-03 3.24534320e-03
#   5.27500904e-03 5.00939654e-03 5.50065806e-03 2.01022039e-03
#   1.68686950e-03 9.85495099e-03 2.80013457e-03 6.57477968e-03
#   1.25410645e-03 1.00000000e-02 5.06879220e-04]
#  [3.55703927e-03 2.06702380e-03 7.89572622e-03 1.67468893e-03
#   1.17135699e-03 1.98735980e-03 5.91731613e-03 1.00000000e-02
#   5.33768192e-03 2.04326684e-03 2.45918001e-03 4.01504506e-06
#   1.00000000e-02 9.68152923e-03 6.85929539e-03]]


