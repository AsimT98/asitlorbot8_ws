import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from skopt import gp_minimize
from skopt.space import Real

# Load Data
data = pd.read_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/optimization_7.xlsx")
total_rows = len(data)
batch_size = 50

# Define Columns
gt_columns = ['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z', 'GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W', 
              'GT_Vel_X', 'GT_Vel_Y', 'GT_Vel_Z', 'GT_Vel_Roll', 'GT_Vel_Pitch', 
              'GT_Angular_Vel_Yaw', 'GT_Accel_X', 'GT_Accel_Y', 'GT_Accel_Z']

et_columns = ['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z', 'ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W', 
              'ET_Vel_X', 'ET_Vel_Y', 'ET_Vel_Z', 'ET_Vel_Roll', 'ET_Vel_Pitch', 
              'ET_Angular_Vel_Yaw', 'ET_Accel_X', 'ET_Accel_Y', 'ET_Accel_Z']

# Define Objective Function
def objective_function(params):
    # Reshape params to form the process noise covariance matrix
    process_noise_covariance_matrix = np.array(params).reshape((15, 15))

    average_NEES_values = []
    average_position_errors = []
    average_orientation_errors = []
    average_trace_covariances = []

    target_NEES = 24.996
    weight_NEES = 1.0
    weight_RPE = 0.5
    weight_RRE = 0.5
    weight_trace = 0.1

    for start_idx in range(0, total_rows, batch_size):
        actual_batch_size = min(batch_size, total_rows - start_idx)
        x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)

        e_x_batch = x_gt_batch - x_est_batch

        NEES_values = []
        position_errors = []
        orientation_errors = []
        trace_covariances = []

        for i in range(actual_batch_size):
            try:
                inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrix)
                e_x_i = e_x_batch.iloc[i].values[:15]
                NEES = e_x_i @ inv_cov_matrix @ e_x_i.T
                if np.isnan(NEES):
                    NEES = np.inf
            except np.linalg.LinAlgError:
                NEES = np.inf
            NEES_values.append(NEES)

            g_i = x_gt_batch.iloc[i][['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values
            s_i = x_est_batch.iloc[i][['ET_Pos_X', 'ET_Pos_Y', 'ET_Pos_Z']].values
            position_error = np.linalg.norm((g_i - s_i) / (g_i + 1e-6)) ** 2
            if np.isnan(position_error):
                position_error = np.inf
            position_errors.append(position_error)

            g_quat = x_gt_batch.iloc[i][['GT_Orient_X', 'GT_Orient_Y', 'GT_Orient_Z', 'GT_Orient_W']].values
            s_quat = x_est_batch.iloc[i][['ET_Orient_X', 'ET_Orient_Y', 'ET_Orient_Z', 'ET_Orient_W']].values
            g_euler = R.from_quat(g_quat).as_euler('xyz')
            s_euler = R.from_quat(s_quat).as_euler('xyz')
            orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
            if np.isnan(orientation_error):
                orientation_error = np.inf
            orientation_errors.append(orientation_error)

            trace_covariance = np.trace(process_noise_covariance_matrix)
            if np.isnan(trace_covariance):
                trace_covariance = np.inf
            trace_covariances.append(trace_covariance)

        avg_NEES = np.mean(NEES_values)
        avg_position_error = np.sqrt(np.mean(position_errors))
        avg_orientation_error = np.mean(orientation_errors)
        avg_trace_covariance = np.mean(trace_covariances)

        objective_score = (
            weight_NEES * avg_NEES +
            weight_RPE * avg_position_error +
            weight_RRE * avg_orientation_error +
            weight_trace * avg_trace_covariance
        )

        average_NEES_values.append(avg_NEES)
        average_position_errors.append(avg_position_error)
        average_orientation_errors.append(avg_orientation_error)
        average_trace_covariances.append(avg_trace_covariance)

    final_objective = np.mean(objective_score)
    
    # Return a large value if final_objective is NaN
    if np.isnan(final_objective):
        final_objective = np.inf

    return final_objective

# Define Search Space and Run Optimization
space = [Real(1e-6, 1e-1, name=f'param_{i}') for i in range(15 * 15)]  # Assuming the covariance matrix has small values

res = gp_minimize(objective_function, space, n_calls=50, random_state=0)

# Output the Results
print(f"Optimized process noise covariance parameters: {res.x}")
print(f"Minimum objective function value: {res.fun}")



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

# # Define the target NEES value
# target_NEES = 24.996

# # Define weights for each metric (you can adjust these weights based on importance)
# weight_NEES = 1.0
# weight_RPE = 0.5
# weight_RRE = 0.5
# weight_trace = 0.1  # Lower weight as trace value can vary significantly

# def objective_function(process_noise_covariance_flat):
#     process_noise_covariance_matrices = [
#         np.array(process_noise_covariance_flat).reshape((15, 15))
#         for _ in range(total_rows)
#     ]
    
#     average_NEES_values = []
#     average_position_errors = []
#     average_orientation_errors = []
#     average_trace_covariances = []
    
#     for start_idx in range(0, total_rows, batch_size):
#         actual_batch_size = min(batch_size, total_rows - start_idx)
        
#         x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
        
#         NEES_values = []
#         position_errors = []
#         orientation_errors = []
#         trace_covariances = []
        
#         for i in range(actual_batch_size):
#             try:
#                 inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[start_idx + i])
#                 e_x_i = (x_gt_batch.iloc[i].values - x_est_batch.iloc[i].values)[:15]
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
#             orientation_error = np.sum((np.cos(g_euler - s_euler)) ** -1)
#             orientation_errors.append(orientation_error)
            
#             trace_covariance = np.trace(process_noise_covariance_matrices[start_idx + i])
#             trace_covariances.append(trace_covariance)
        
#         avg_NEES = np.mean(NEES_values)
#         avg_position_error = np.sqrt(np.mean(position_errors))
#         avg_orientation_error = np.mean(orientation_errors)
#         avg_trace_covariance = np.mean(trace_covariances)
        
#         average_NEES_values.append(avg_NEES)
#         average_position_errors.append(avg_position_error)
#         average_orientation_errors.append(avg_orientation_error)
#         average_trace_covariances.append(avg_trace_covariance)
    
#     final_objective = (
#         weight_NEES * np.mean(average_NEES_values) +
#         weight_RPE * np.mean(average_position_errors) +
#         weight_RRE * np.mean(average_orientation_errors) +
#         weight_trace * np.mean(average_trace_covariances)
#     )
    
#     return final_objective

# initial_covariance_matrix = np.eye(15)
# initial_covariance_flat = initial_covariance_matrix.flatten()

# result = minimize(objective_function, initial_covariance_flat, method='Nelder-Mead')

# optimized_covariance_matrix = result.x.reshape((15, 15))

# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_covariance_matrix)



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

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

# # Initialize list to store average NEES values for each batch
# average_NEES_values = []

# # Define the target NEES value
# target_NEES = 24.996

# # Initialize a PDF file to save plots
# pdf_pages = PdfPages('average_NEES_plots.pdf')

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
#         np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15))
#         for i in range(actual_batch_size)
#     ]
    
#     # Print process noise covariance matrices for the current batch
#     print(f"Batch {start_idx // batch_size + 1} - Process Noise Covariance Matrix:")
#     for i, cov_matrix in enumerate(process_noise_covariance_matrices):
#         print(f"Sample {i + 1}:")
#         print(cov_matrix)
    
#     # Calculate NEES for each sample in the batch
#     NEES_values = []
#     for i in range(actual_batch_size):
#         try:
#             inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             NEES = e_x_batch.iloc[i].values @ inv_cov_matrix @ e_x_batch.iloc[i].values.T
#         except np.linalg.LinAlgError:
#             NEES = np.inf  # If the matrix is singular, assign a very high NEES
#         NEES_values.append(NEES)
    
#     # Calculate average NEES for the batch
#     avg_NEES = np.mean(NEES_values)
#     average_NEES_values.append(avg_NEES)
    
#     # Plot NEES for the current batch
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(NEES_values)), NEES_values, marker='o', linestyle='-', color='b', label='NEES')
#     plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
#     plt.xlabel('Sample Number')
#     plt.ylabel('NEES')
#     plt.title(f'NEES for Batch {start_idx // batch_size + 1}')
#     plt.legend()
#     plt.grid(True)
    
#     # Annotate the plot with the process noise covariance matrix used
#     plt.annotate(f'Process Noise Covariance Matrix:\n{process_noise_covariance_matrices}',
#                  xy=(0.5, 0.95), xycoords='axes fraction',
#                  horizontalalignment='center', verticalalignment='top',
#                  fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
#     # Save the current plot to PDF
#     pdf_pages.savefig()
#     plt.close()

# # Close the PDF file
# pdf_pages.close()

# # Plot average NEES for each batch
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch')
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
# from matplotlib.backends.backend_pdf import PdfPages

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

# # Define dimension names for the diagonal elements
# dimension_names = [f'covariance_{i}' for i in range(15)]

# # Define the bounds for the diagonal elements
# bounds = [Real(0.00000001, 0.05, name=name) for name in dimension_names]

# # Define the target NEES value
# target_NEES = 24.996

# # Function to calculate RMSE for translation and rotation
# def calculate_rmse(x_gt_batch, x_est_batch):
#     translation_rmse = np.sqrt(np.mean((x_gt_batch[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values - 
#                                         x_est_batch[['GT_Pos_X', 'GT_Pos_Y', 'GT_Pos_Z']].values)**2))
    
#     rotation_rmse = np.sqrt(np.mean((x_gt_batch[['GT_Roll', 'GT_Pitch', 'GT_Yaw']].values - 
#                                      x_est_batch[['GT_Roll', 'GT_Pitch', 'GT_Yaw']].values)**2))
    
#     return translation_rmse, rotation_rmse

# # Define the objective function for Bayesian optimization
# @use_named_args(bounds)
# def objective_function(**params):
#     diagonal_values = [params[name] for name in dimension_names]
#     process_noise_covariance_matrix = np.diag(diagonal_values)
    
#     NEES_values = []
#     translation_rmse_values = []
#     rotation_rmse_values = []
    
#     for start_idx in range(0, total_rows, batch_size):
#         actual_batch_size = min(batch_size, total_rows - start_idx)
#         x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#         x_est_batch.columns = gt_columns
#         e_x_batch = x_gt_batch - x_est_batch
#         e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]

#         # Extract the process noise covariance matrices for the current batch
#         process_noise_covariance_matrices = [
#             np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15))
#             for i in range(actual_batch_size)
#         ]
        
#         # Print process noise covariance matrices for the current batch
#         # print(f"Batch {start_idx // batch_size + 1} - Process Noise Covariance Matrix:")
#         # for i, cov_matrix in enumerate(process_noise_covariance_matrices):
#         #     print(f"Sample {i + 1}:")
#         #     print(cov_matrix)

#         # Calculate NEES
#         for i in range(actual_batch_size):
#             try:
#                 inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#                 NEES = e_x_batch.iloc[i].values @ inv_cov_matrix @ e_x_batch.iloc[i].values.T
#             except np.linalg.LinAlgError:
#                 NEES = np.inf  # If the matrix is singular, assign a very high NEES
#             NEES_values.append(NEES)
        
#         # Calculate RMSE
#         translation_rmse, rotation_rmse = calculate_rmse(x_gt_batch, x_est_batch)
#         translation_rmse_values.append(translation_rmse)
#         rotation_rmse_values.append(rotation_rmse)

#     avg_NEES = np.mean(NEES_values)
#     avg_translation_rmse = np.mean(translation_rmse_values)
#     avg_rotation_rmse = np.mean(rotation_rmse_values)
#     trace = np.trace(process_noise_covariance_matrix)

#     # Combine NEES, trace, and RMSE into the objective function
#     penalty = abs(avg_NEES - target_NEES)
#     return 0.5 * avg_NEES + 0.2 * trace + 0.2 * avg_translation_rmse + 0.1 * avg_rotation_rmse + penalty

# # Perform Bayesian optimization to find the optimized diagonal elements
# res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=50, n_random_starts=10, random_state=42)

# # Get the optimized diagonal elements from the best solution found by Bayesian optimization
# optimized_diagonal_values = res_gp.x
# optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)

# # Print the optimized process noise covariance matrix
# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Initialize a PDF file to save plots
# # pdf_pages = PdfPages('average_NEES_plots_optimized.pdf')

# # Calculate the average NEES for each batch using the optimized process noise covariance matrix
# average_NEES_values = []
# for start_idx in range(0, total_rows, batch_size):
#     actual_batch_size = min(batch_size, total_rows - start_idx)
#     x_gt_batch = data[gt_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#     x_est_batch = data[et_columns][start_idx:start_idx + actual_batch_size].reset_index(drop=True)
#     x_est_batch.columns = gt_columns
#     e_x_batch = x_gt_batch - x_est_batch
#     e_x_batch.columns = [f'error_{col[3:]}' for col in gt_columns]

#     # Extract the process noise covariance matrices for the current batch
#     process_noise_covariance_matrices = [
#         np.array(data.iloc[start_idx + i, 40:].values).reshape((15, 15))
#         for i in range(actual_batch_size)
#     ]
    
#     NEES_values = []
#     for i in range(actual_batch_size):
#         try:
#             inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrices[i])
#             NEES = e_x_batch.iloc[i].values @ inv_cov_matrix @ e_x_batch.iloc[i].values.T
#         except np.linalg.LinAlgError:
#             NEES = np.inf  # If the matrix is singular, assign a very high NEES
#         NEES_values.append(NEES)
#     avg_NEES = np.mean(NEES_values)
#     average_NEES_values.append(avg_NEES)
    
#     # Plot NEES for the current batch
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(NEES_values)), NEES_values, marker='o', linestyle='-', color='b', label='NEES')
#     plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
#     plt.xlabel('Sample Number')
#     plt.ylabel('NEES')
#     plt.title(f'NEES for Batch {start_idx // batch_size + 1}')
#     plt.legend()
#     plt.grid(True)
    
#     # Annotate the plot with the process noise covariance matrix used
#     plt.annotate(f'Process Noise Covariance Matrices used in the batch:\n'
#                  f'{process_noise_covariance_matrices}',
#                  xy=(0.5, 0.95), xycoords='axes fraction',
#                  horizontalalignment='center', verticalalignment='top',
#                  fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
#     # Save the current plot to PDF
#     # pdf_pages.savefig()
#     plt.close()

# # Close the PDF file
# # pdf_pages.close()

# # Plot average NEES for each batch
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(average_NEES_values)), average_NEES_values, marker='o', linestyle='-', color='b', label='Average NEES')
# plt.axhline(y=target_NEES, color='r', linestyle='--', label='Target NEES')
# plt.xlabel('Batch Number')
# plt.ylabel('Average NEES')
# plt.title('Average NEES for Each Batch')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
