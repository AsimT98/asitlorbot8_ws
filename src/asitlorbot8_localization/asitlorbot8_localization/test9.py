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
# pbounds = {f'd{i}': (0.00000001, 0.9) for i in range(15)}
pbounds = {
    'd0': (0.001, 0.1),    # Position x  
    'd1': (0.001, 0.1),    # Position y
    'd2': (0.001, 0.1),    # Position z
    'd3': (0.01, 0.09),  # Orientation roll
    'd4': (0.01, 0.09),  # Orientation pitch
    'd5': (0.01, 0.09),  # Orientation yaw
    'd6': (0.02, 0.09),      # Linear velocity x
    'd7': (0.02, 0.09),      # Linear velocity y
    'd8': (0.02, 0.09),      # Linear velocity z
    'd9': (0.02, 0.09),     # Angular velocity x
    'd10': (0.02, 0.09),    # Angular velocity y
    'd11': (0.02, 0.09),    # Angular velocity z
    'd12': (0.02, 0.09),       # Linear acceleration x
    'd13': (0.02, 0.09),       # Linear acceleration y
    'd14': (0.02, 0.09),       # Linear acceleration z
}

# Initialize Bayesian optimizer
optimizer = BayesianOptimization(
    f=calculate_objective,
    pbounds=pbounds,
    random_state=1,
)

# Perform optimization with more iterations
optimizer.maximize(
    init_points=5,
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
optimal_diagonals_df.to_excel("/home/asimkumar/asitlorbot8_ws/exceldata/optimization/results_with_new_objective9.xlsx", index=False)

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

# # Extract process noise covariance matrices for each batch
# process_noise_covariance_matrices = [
#     [
#         np.array(data.iloc[start_idx + i, 40:40 + 15*15].values).reshape((15, 15))
#         for i in range(min(batch_size, total_rows - start_idx))
#     ]
#     for start_idx in range(0, total_rows, batch_size)
# ]

# # Define weights for each metric (you can adjust these weights based on importance)
# weight_NEES = 0.4
# weight_RPE = 0.2
# weight_RRE = 0.2
# weight_trace = 0.2  # Lower weight as trace value can vary significantly

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

#         # Adjust the diagonal elements for the process noise covariance matrices
#         adjusted_cov_matrices = [
#             process_noise_covariance_matrices[start_idx // batch_size][i] + np.diag(diagonal_elements)
#             for i in range(actual_batch_size)
#         ]
        
#         NEES_values = []
#         position_errors = []
#         orientation_errors = []
#         trace_covariances = []

#         for i in range(actual_batch_size):
#             try:
#                 reg_matrix = adjusted_cov_matrices[i] + np.eye(15) * 1e-6
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
            
#             trace_covariance = np.trace(adjusted_cov_matrices[i])
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
# pbounds = {f'd{i}': (0.00000001,0.01) for i in range(15)}

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

# [[0.08608513 0.06637575 0.01666761 0.01566985 0.00885562 0.05942725
#   0.06879354 0.02385419 0.00188506 0.00739546 0.03179263 0.02668821
#   0.0314733  0.06971882 0.05952336]
#  [0.00892604 0.03613449 0.01692223 0.03198462 0.07054126 0.04988039
#   0.00047068 0.06848917 0.00317803 0.06711604 0.07200639 0.01836322
#   0.04995765 0.06597642 0.05543869]
#  [0.01822326 0.08622661 0.08967009 0.03314786 0.04037496 0.06498639
#   0.07975762 0.05337399 0.03523732 0.03713597 0.03311467 0.02942385
#   0.01339993 0.02750439 0.07889857]
#  [0.06260564 0.00028965 0.04328757 0.0578093  0.00583683 0.05219855
#   0.05053362 0.05045941 0.05431389 0.06088212 0.05576304 0.03199438
#   0.07147776 0.00836917 0.05293821]
#  [0.07244901 0.02428387 0.07064073 0.06369129 0.07097535 0.04655422
#   0.03961792 0.01327074 0.02953735 0.03906175 0.07425445 0.04484302
#   0.00693525 0.00526959 0.03008146]
#  [0.00797405 0.01985508 0.03791233 0.05725597 0.07070865 0.01065027
#   0.03689145 0.07558221 0.03454497 0.05146851 0.05384028 0.06620968
#   0.08985128 0.0839802  0.05783087]
#  [0.05289925 0.01660287 0.02457638 0.04662709 0.02688531 0.08466113
#   0.02333672 0.03866912 0.07854572 0.07577402 0.0326012  0.03010602
#   0.00235771 0.00217727 0.07485274]
#  [0.01674914 0.0722379  0.04468151 0.03931657 0.06565574 0.06889616
#   0.01430174 0.05492027 0.01218188 0.06762376 0.04123683 0.0434672
#   0.01201321 0.00725415 0.06551454]
#  [0.05912597 0.08609532 0.08156291 0.02245917 0.02447548 0.06834585
#   0.04047659 0.06990395 0.00588296 0.04388141 0.00620623 0.00513493
#   0.02539684 0.02355352 0.0222281 ]
#  [0.00302523 0.0056388  0.080985   0.00196412 0.05974108 0.0867055
#   0.05041514 0.08431402 0.00470322 0.0376914  0.08157937 0.01253209
#   0.04791787 0.03699861 0.03126091]
#  [0.0871074  0.02659004 0.05535117 0.06971704 0.05795139 0.0477272
#   0.00377562 0.08716399 0.07188428 0.02635399 0.06923009 0.05621972
#   0.03437457 0.01851186 0.01092479]
#  [0.08819733 0.05416935 0.03044408 0.08352753 0.02021541 0.03349504
#   0.03888693 0.03954645 0.05516457 0.08487683 0.05241804 0.06732659
#   0.07305928 0.05908308 0.01152863]
#  [0.02166235 0.01093513 0.0775233  0.07618629 0.08270339 0.0227017
#   0.06795378 0.04144856 0.07577987 0.06556416 0.01777235 0.07982324
#   0.05812298 0.02573162 0.07343523]
#  [0.06988027 0.05905457 0.01483334 0.0118556  0.06533819 0.07360068
#   0.01921603 0.04552675 0.07566327 0.06595214 0.0159686  0.04905243
#   0.08862028 0.08436493 0.00388565]
#  [0.04880135 0.0531313  0.05726623 0.06850094 0.01440646 0.04154018
#   0.00083986 0.02220111 0.06538156 0.0892629  0.04575245 0.02677937
#   0.05085198 0.06199968 0.07859906]]


# import pandas as pd
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args

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
# bounds = [Real(0.0001, 1, name=name) for name in dimension_names]

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
# def objective_function(*params):
#     diagonal_values = params
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

#         # Calculate NEES
#         for i in range(actual_batch_size):
#             try:
#                 inv_cov_matrix = np.linalg.inv(process_noise_covariance_matrix)
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
#     objective_value = 0.5 * avg_NEES + 0.2 * trace + 0.2 * avg_translation_rmse + 0.1 * avg_rotation_rmse + penalty

#     # Debugging information
#     print(f"Evaluated objective function with params: {params}")
#     print(f"Objective value: {objective_value}")
#     print(f"NEES: {avg_NEES}, Translation RMSE: {avg_translation_rmse}, Rotation RMSE: {avg_rotation_rmse}, Trace: {trace}")
#     print(f"Process noise covariance matrix: {process_noise_covariance_matrix}")

#     return objective_value

# # Perform Bayesian optimization to find the optimized diagonal elements
# res_gp = gp_minimize(objective_function, dimensions=bounds, n_calls=50, n_random_starts=10, random_state=42)

# # Get the optimized diagonal elements from the best solution found by Bayesian optimization
# optimized_diagonal_values = res_gp.x
# optimized_process_noise_covariance_matrix = np.diag(optimized_diagonal_values)

# # Print the optimized process noise covariance matrix
# print("Optimized Process Noise Covariance Matrix:")
# print(optimized_process_noise_covariance_matrix)

# # Print the final objective value for better understanding of optimization outcome
# final_objective_value = objective_function(*optimized_diagonal_values)
# print("Final Objective Value:", final_objective_value)
