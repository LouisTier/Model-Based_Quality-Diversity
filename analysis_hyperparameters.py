"""
This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import os
import matplotlib.pyplot as plt
from annexed_methods import find_txt, plot_metrics_solo_hyper, plot_hyperparam, retrieve_elements, best_qd_score
import numpy as np

os.makedirs("GridSearch_Comparison/Explicit_Uncorrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/Explicit_Corrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/Implicit_Uncorrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/Implicit_Corrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/UQ_Explicit_Uncorrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/UQ_Explicit_Corrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/UQ_Implicit_Uncorrected", exist_ok=True)
os.makedirs("GridSearch_Comparison/UQ_Implicit_Corrected", exist_ok=True)

log_period = 10
sampling_size = 512
a = "arm"
na = "noisy_arm"
width = 18 # 20 | 18 (Report)
height = 6 # 9 | 6 (Report)
top_k=10

# True = GridSearch analysis of MBMEUQ algorithms, False = GridSearch analysis of MBME algorithms
uq_analysis = True

###########################################
#         GridSearch Explicit Arm         #
###########################################

if uq_analysis == False:
  time_exp_a = "2023-07-30_15_58_34_2008152" 
  method_exp = "MEMB_Explicit_Naive"
  path_exp_a = f"results_from_HPC/{time_exp_a}/{method_exp}"
  task_name = 'Deterministic Arm'
  add_title = ""
  # Saving path
  path_saving_unc = "GridSearch_Comparison/Explicit_Uncorrected/Arm"
  path_saving_cor = "GridSearch_Comparison/Explicit_Corrected/Arm"
  best_path_saving_unc = "GridSearch_Comparison/Explicit_Uncorrected/Best_Arm"
  best_path_saving_cor = "GridSearch_Comparison/Explicit_Corrected/Best_Arm"
else:
  time_exp_a = "2023-08-19_11_13_22_1456520" 
  method_exp = "MEMBUQ_NLL_Explicit_Naive"
  path_exp_a = f"results_from_HPC/{time_exp_a}/{method_exp}"
  task_name = 'Deterministic Arm'
  add_title = "UQ_"
  # Saving path
  path_saving_unc = "GridSearch_Comparison/UQ_Explicit_Uncorrected/Arm"
  path_saving_cor = "GridSearch_Comparison/UQ_Explicit_Corrected/Arm"
  best_path_saving_unc = "GridSearch_Comparison/UQ_Explicit_Uncorrected/Best_Arm"
  best_path_saving_cor = "GridSearch_Comparison/UQ_Explicit_Corrected/Best_Arm"

# Figures
fig_determ_a_unc, axes_determ_a_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
fig_determ_a_cor, axes_determ_a_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_determ_a_unc, top_axes_determ_a_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_determ_a_cor, top_axes_determ_a_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))

# Retrieve all txt files from the GridSearch, there are as many files as the number of simulations
txt_files_list = find_txt(path_exp_a)

# Initialization of the lists
start_training_list, nb_epochs_list, per_data_list, csv_unc_list, csv_cor_list, time_list = [], [], [], [], [], []

# Lists of the hyperparameters and lists of pre-processed csvs to ensure a good plot of the metrics
start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list = retrieve_elements(
  txt_list=txt_files_list, 
  file_path=path_exp_a, 
  method=method_exp, 
  task_name=a, 
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list, 
  per_data_list=per_data_list, 
  time_list=time_list,
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list
)

# unc and cor lists are the same except for the csv when we don't care about the best lists
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=path_saving_unc,
  save_cor=path_saving_cor,
  fing_unc=fig_determ_a_unc, 
  ax_unc=axes_determ_a_unc, 
  fig_cor=fig_determ_a_cor, 
  ax_cor=axes_determ_a_cor,
  start_training_list_unc=start_training_list, 
  nb_epochs_list_unc=nb_epochs_list, 
  per_data_list_unc=per_data_list, 
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list,
  start_training_list_cor=start_training_list, 
  nb_epochs_list_cor=nb_epochs_list, 
  per_data_list_cor=per_data_list,
  show_legend=False,
  algo_name=method_exp
)

print(f"\nAnalysis for the GridSearch of the arm with the {add_title}Explicit algorithm")
# Retrieving best csv and hyperparameters lists based on last qd_score value (uncorrected metrics)
top_unc_indices, top_unc_lists, top_unc_start_training_list, top_unc_nb_epochs_list, top_unc_per_data_list, top_unc_time_list = best_qd_score(
  csv_list=csv_unc_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the uncorrected metrics are the ones with the following time: \nindices: {top_unc_indices} \ntime: {top_unc_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")
# if ['28-07-2023_11-06-32', '28-07-2023_11-23-49', '28-07-2023_11-32-18'], it means 11-06-32 < 11-23-49 < 11-32-18

# Retrieving best csv and hyperparameters lists based on last qd_score value (corrected metrics)
top_cor_indices, top_cor_lists, top_cor_start_training_list, top_cor_nb_epochs_list, top_cor_per_data_list, top_cor_time_list = best_qd_score(
  csv_list=csv_cor_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the corrected metrics are the ones with the following time: \nindices: {top_cor_indices} \ntime: {top_cor_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")

# unc and cor lists are different when we care about plotting the best ones
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=best_path_saving_unc,
  save_cor=best_path_saving_cor,
  fing_unc=top_fig_determ_a_unc, 
  ax_unc=top_axes_determ_a_unc, 
  fig_cor=top_fig_determ_a_cor, 
  ax_cor=top_axes_determ_a_cor,
  start_training_list_unc=top_unc_start_training_list, 
  nb_epochs_list_unc=top_unc_nb_epochs_list, 
  per_data_list_unc=top_unc_per_data_list, 
  csv_unc_list=top_unc_lists, 
  csv_cor_list=top_cor_lists,
  start_training_list_cor=top_cor_start_training_list, 
  nb_epochs_list_cor=top_cor_nb_epochs_list, 
  per_data_list_cor=top_cor_per_data_list,
  show_legend=True,
  algo_name=method_exp
)



###########################################
#     GridSearch Explicit Noisy Arm       #
###########################################

if uq_analysis == False:
  time_exp_na = "2023-07-30_15_58_33_467574" 
  method_exp = "MEMB_Explicit_Naive"
  path_exp_na = f"results_from_HPC/{time_exp_na}/{method_exp}"
  task_name = 'Uncertain Arm'
  add_title = ""
  # Saving path
  path_saving_unc = "GridSearch_Comparison/Explicit_Uncorrected/Noisy_Arm"
  path_saving_cor = "GridSearch_Comparison/Explicit_Corrected/Noisy_Arm"
  best_path_saving_unc = "GridSearch_Comparison/Explicit_Uncorrected/Best_Noisy_Arm"
  best_path_saving_cor = "GridSearch_Comparison/Explicit_Corrected/Best_Noisy_Arm"
else:
  time_exp_na = "2023-08-19_11_15_48_1459306" 
  method_exp = "MEMBUQ_NLL_Explicit_Naive"
  path_exp_na = f"results_from_HPC/{time_exp_na}/{method_exp}"
  task_name = 'Uncertain Arm'
  add_title = "UQ_"
  # Saving path
  path_saving_unc = "GridSearch_Comparison/UQ_Explicit_Uncorrected/Noisy_Arm"
  path_saving_cor = "GridSearch_Comparison/UQ_Explicit_Corrected/Noisy_Arm"
  best_path_saving_unc = "GridSearch_Comparison/UQ_Explicit_Uncorrected/Best_Noisy_Arm"
  best_path_saving_cor = "GridSearch_Comparison/UQ_Explicit_Corrected/Best_Noisy_Arm"

# Figures
fig_uncert_na_unc, axes_uncert_na_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
fig_uncert_na_cor, axes_uncert_na_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_uncert_na_unc, top_axes_uncert_na_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_uncert_na_cor, top_axes_uncert_na_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))

# Retrieve all txt files from the GridSearch, there are as many files as the number of simulations
txt_files_list = find_txt(path_exp_na)

# Initialization of the lists
start_training_list, nb_epochs_list, per_data_list, csv_unc_list, csv_cor_list, time_list = [], [], [], [], [], []

# Lists of the hyperparameters and lists of pre-processed csvs to ensure a good plot of the metrics
start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list = retrieve_elements(
  txt_list=txt_files_list, 
  file_path=path_exp_na, 
  method=method_exp, 
  task_name=na, 
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list, 
  per_data_list=per_data_list, 
  time_list=time_list,
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list
)

# unc and cor lists are the same except for the csv when we don't care about the best lists
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=path_saving_unc,
  save_cor=path_saving_cor,
  fing_unc=fig_uncert_na_unc, 
  ax_unc=axes_uncert_na_unc, 
  fig_cor=fig_uncert_na_cor, 
  ax_cor=axes_uncert_na_cor,
  start_training_list_unc=start_training_list, 
  nb_epochs_list_unc=nb_epochs_list, 
  per_data_list_unc=per_data_list, 
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list,
  start_training_list_cor=start_training_list, 
  nb_epochs_list_cor=nb_epochs_list, 
  per_data_list_cor=per_data_list,
  show_legend=False,
  algo_name=method_exp
)

print(f"\nAnalysis for the GridSearch of the noisy arm with the {add_title}Explicit algorithm")
# Retrieving best csv and hyperparameters lists based on last qd_score value (uncorrected metrics)
top_unc_indices, top_unc_lists, top_unc_start_training_list, top_unc_nb_epochs_list, top_unc_per_data_list, top_unc_time_list = best_qd_score(
  csv_list=csv_unc_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the uncorrected metrics are the ones with the following time: \nindices: {top_unc_indices} \ntime: {top_unc_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")
# if ['28-07-2023_11-06-32', '28-07-2023_11-23-49', '28-07-2023_11-32-18'], it means 11-06-32 < 11-23-49 < 11-32-18

# Retrieving best csv and hyperparameters lists based on last qd_score value (corrected metrics)
top_cor_indices, top_cor_lists, top_cor_start_training_list, top_cor_nb_epochs_list, top_cor_per_data_list, top_cor_time_list = best_qd_score(
  csv_list=csv_cor_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the corrected metrics are the ones with the following time: \nindices: {top_cor_indices} \ntime: {top_cor_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")

# unc and cor lists are different when we care about plotting the best ones
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=best_path_saving_unc,
  save_cor=best_path_saving_cor,
  fing_unc=top_fig_uncert_na_unc, 
  ax_unc=top_axes_uncert_na_unc, 
  fig_cor=top_fig_uncert_na_cor, 
  ax_cor=top_axes_uncert_na_cor,
  start_training_list_unc=top_unc_start_training_list, 
  nb_epochs_list_unc=top_unc_nb_epochs_list, 
  per_data_list_unc=top_unc_per_data_list, 
  csv_unc_list=top_unc_lists, 
  csv_cor_list=top_cor_lists,
  start_training_list_cor=top_cor_start_training_list, 
  nb_epochs_list_cor=top_cor_nb_epochs_list, 
  per_data_list_cor=top_cor_per_data_list,
  show_legend=True,
  algo_name=method_exp
)



###########################################
#         GridSearch Implicit Arm         #
###########################################
if uq_analysis == False:
  time_imp_a = "2023-07-30_15_58_35_4189677" 
  method_imp = "MEMB_Implicit"
  path_imp_a = f"results_from_HPC/{time_imp_a}/{method_imp}"
  task_name = 'Deterministic Arm'
  add_title = ""
  # Saving path
  path_saving_unc = "GridSearch_Comparison/Implicit_Uncorrected/Arm"
  path_saving_cor = "GridSearch_Comparison/Implicit_Corrected/Arm"
  best_path_saving_unc = "GridSearch_Comparison/Implicit_Uncorrected/Best_Arm"
  best_path_saving_cor = "GridSearch_Comparison/Implicit_Corrected/Best_Arm"
else:
  time_imp_a = "2023-08-19_11_35_38_1076879"
  method_imp = "MEMBUQ_Implicit"
  path_imp_a = f"results_from_HPC/{time_imp_a}/{method_imp}"
  task_name = 'Deterministic Arm'
  add_title = "UQ_"
  # Saving path
  path_saving_unc = "GridSearch_Comparison/UQ_Implicit_Uncorrected/Arm"
  path_saving_cor = "GridSearch_Comparison/UQ_Implicit_Corrected/Arm"
  best_path_saving_unc = "GridSearch_Comparison/UQ_Implicit_Uncorrected/Best_Arm"
  best_path_saving_cor = "GridSearch_Comparison/UQ_Implicit_Corrected/Best_Arm"

# Figures
fig_determ_a_unc, axes_determ_a_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
fig_determ_a_cor, axes_determ_a_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_determ_a_unc, top_axes_determ_a_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_determ_a_cor, top_axes_determ_a_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))

# Retrieve all txt files from the GridSearch, there are as many files as the number of simulations
txt_files_list = find_txt(path_imp_a)

# Initialization of the lists
start_training_list, nb_epochs_list, per_data_list, csv_unc_list, csv_cor_list, time_list = [], [], [], [], [], []

# Lists of the hyperparameters and lists of pre-processed csvs to ensure a good plot of the metrics
start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list = retrieve_elements(
  txt_list=txt_files_list, 
  file_path=path_imp_a, 
  method=method_imp, 
  task_name=a, 
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list, 
  per_data_list=per_data_list, 
  time_list=time_list,
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list
)

# unc and cor lists are the same except for the csv when we don't care about the best lists
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=path_saving_unc,
  save_cor=path_saving_cor,
  fing_unc=fig_determ_a_unc, 
  ax_unc=axes_determ_a_unc, 
  fig_cor=fig_determ_a_cor, 
  ax_cor=axes_determ_a_cor,
  start_training_list_unc=start_training_list, 
  nb_epochs_list_unc=nb_epochs_list, 
  per_data_list_unc=per_data_list, 
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list,
  start_training_list_cor=start_training_list, 
  nb_epochs_list_cor=nb_epochs_list, 
  per_data_list_cor=per_data_list,
  show_legend=False,
  algo_name=method_imp
)

print(f"\nAnalysis for the GridSearch of the arm with the {add_title}Implicit algorithm")
# Retrieving best csv and hyperparameters lists based on last qd_score value (uncorrected metrics)
top_unc_indices, top_unc_lists, top_unc_start_training_list, top_unc_nb_epochs_list, top_unc_per_data_list, top_unc_time_list = best_qd_score(
  csv_list=csv_unc_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the uncorrected metrics are the ones with the following time: \nindices: {top_unc_indices} \ntime: {top_unc_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")
# if ['28-07-2023_11-06-32', '28-07-2023_11-23-49', '28-07-2023_11-32-18'], it means 11-06-32 < 11-23-49 < 11-32-18

# Retrieving best csv and hyperparameters lists based on last qd_score value (corrected metrics)
top_cor_indices, top_cor_lists, top_cor_start_training_list, top_cor_nb_epochs_list, top_cor_per_data_list, top_cor_time_list = best_qd_score(
  csv_list=csv_cor_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the corrected metrics are the ones with the following time: \nindices: {top_cor_indices} \ntime: {top_cor_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")

# unc and cor lists are different when we care about plotting the best ones
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=best_path_saving_unc,
  save_cor=best_path_saving_cor,
  fing_unc=top_fig_determ_a_unc, 
  ax_unc=top_axes_determ_a_unc, 
  fig_cor=top_fig_determ_a_cor, 
  ax_cor=top_axes_determ_a_cor,
  start_training_list_unc=top_unc_start_training_list, 
  nb_epochs_list_unc=top_unc_nb_epochs_list, 
  per_data_list_unc=top_unc_per_data_list, 
  csv_unc_list=top_unc_lists, 
  csv_cor_list=top_cor_lists,
  start_training_list_cor=top_cor_start_training_list, 
  nb_epochs_list_cor=top_cor_nb_epochs_list, 
  per_data_list_cor=top_cor_per_data_list,
  show_legend=True,
  algo_name=method_imp
)



###########################################
#      GridSearch Implicit Noisy Arm      #
###########################################
if uq_analysis == False:
  time_imp_na = "2023-07-30_15_58_35_2620392" 
  method_imp = "MEMB_Implicit"
  path_imp_na = f"results_from_HPC/{time_imp_na}/{method_imp}"
  task_name = 'Uncertain Arm'
  add_title= ""
  # Saving path
  path_saving_unc = "GridSearch_Comparison/Implicit_Uncorrected/Noisy_Arm"
  path_saving_cor = "GridSearch_Comparison/Implicit_Corrected/Noisy_Arm"
  best_path_saving_unc = "GridSearch_Comparison/Implicit_Uncorrected/Best_Noisy_Arm"
  best_path_saving_cor = "GridSearch_Comparison/Implicit_Corrected/Best_Noisy_Arm"
else:
  time_imp_na = "2023-08-19_11_35_38_4114503" 
  method_imp = "MEMBUQ_Implicit"
  path_imp_na = f"results_from_HPC/{time_imp_na}/{method_imp}"
  task_name = 'Uncertain Arm'
  add_title= "UQ_"
  # Saving path
  path_saving_unc = "GridSearch_Comparison/UQ_Implicit_Uncorrected/Noisy_Arm"
  path_saving_cor = "GridSearch_Comparison/UQ_Implicit_Corrected/Noisy_Arm"
  best_path_saving_unc = "GridSearch_Comparison/UQ_Implicit_Uncorrected/Best_Noisy_Arm"
  best_path_saving_cor = "GridSearch_Comparison/UQ_Implicit_Corrected/Best_Noisy_Arm"


# Figures
fig_uncert_na_unc, axes_uncert_na_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
fig_uncert_na_cor, axes_uncert_na_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_uncert_na_unc, top_axes_uncert_na_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
top_fig_uncert_na_cor, top_axes_uncert_na_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))

# Retrieve all txt files from the GridSearch, there are as many files as the number of simulations
txt_files_list = find_txt(path_imp_na)

# Initialization of the lists
start_training_list, nb_epochs_list, per_data_list, csv_unc_list, csv_cor_list, time_list = [], [], [], [], [], []

# Lists of the hyperparameters and lists of pre-processed csvs to ensure a good plot of the metrics
start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list = retrieve_elements(
  txt_list=txt_files_list, 
  file_path=path_imp_na, 
  method=method_imp, 
  task_name=na, 
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list, 
  per_data_list=per_data_list, 
  time_list=time_list,
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list,
)

# unc and cor lists are the same except for the csv when we don't care about the best lists
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=path_saving_unc,
  save_cor=path_saving_cor,
  fing_unc=fig_uncert_na_unc, 
  ax_unc=axes_uncert_na_unc, 
  fig_cor=fig_uncert_na_cor, 
  ax_cor=axes_uncert_na_cor,
  start_training_list_unc=start_training_list, 
  nb_epochs_list_unc=nb_epochs_list, 
  per_data_list_unc=per_data_list, 
  csv_unc_list=csv_unc_list, 
  csv_cor_list=csv_cor_list,
  start_training_list_cor=start_training_list, 
  nb_epochs_list_cor=nb_epochs_list, 
  per_data_list_cor=per_data_list,
  show_legend=False,
  algo_name=method_imp
)

print(f"\nAnalysis for the GridSearch of the noisy arm with the {add_title}Implicit algorithm")
# Retrieving best csv and hyperparameters lists based on last qd_score value (uncorrected metrics)
top_unc_indices, top_unc_lists, top_unc_start_training_list, top_unc_nb_epochs_list, top_unc_per_data_list, top_unc_time_list = best_qd_score(
  csv_list=csv_unc_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the uncorrected metrics are the ones with the following time: \nindices: {top_unc_indices} \ntime: {top_unc_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")
# if ['28-07-2023_11-06-32', '28-07-2023_11-23-49', '28-07-2023_11-32-18'], it means 11-06-32 < 11-23-49 < 11-32-18

# Retrieving best csv and hyperparameters lists based on last qd_score value (corrected metrics)
top_cor_indices, top_cor_lists, top_cor_start_training_list, top_cor_nb_epochs_list, top_cor_per_data_list, top_cor_time_list = best_qd_score(
  csv_list=csv_cor_list, 
  top_k=top_k,
  start_training_list=start_training_list, 
  nb_epochs_list=nb_epochs_list,
  per_data_list=per_data_list,
  time_list=time_list
)
print(f"\nThe {top_k} best simulations for the corrected metrics are the ones with the following time: \nindices: {top_cor_indices} \ntime: {top_cor_time_list} \npresented as [lowest_qd_score < ... < highest_qd_score]")

# unc and cor lists are different when we care about plotting the best ones
plot_hyperparam(
  log_period=log_period, 
  sampling_size=sampling_size, 
  full_name=task_name,
  save_unc=best_path_saving_unc,
  save_cor=best_path_saving_cor,
  fing_unc=top_fig_uncert_na_unc, 
  ax_unc=top_axes_uncert_na_unc, 
  fig_cor=top_fig_uncert_na_cor, 
  ax_cor=top_axes_uncert_na_cor,
  start_training_list_unc=top_unc_start_training_list, 
  nb_epochs_list_unc=top_unc_nb_epochs_list, 
  per_data_list_unc=top_unc_per_data_list, 
  csv_unc_list=top_unc_lists, 
  csv_cor_list=top_cor_lists,
  start_training_list_cor=top_cor_start_training_list, 
  nb_epochs_list_cor=top_cor_nb_epochs_list, 
  per_data_list_cor=top_cor_per_data_list,
  show_legend=True,
  algo_name=method_imp
)
