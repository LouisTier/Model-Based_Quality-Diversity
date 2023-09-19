"""  
This code comes from the work provided by Louis BERTHIER as part of his Independent Study Option at Imperial College London.
@article{berthier2023iso,
  title={Model-Based Uncertainty Quantification in the context of Reinforcement Learning and Quality-Diversity},
  author={Berthier, Louis and Lim, Bryan and Flageat, Manon and Cully, Antoine},
  year={2023}
}

This work has been adapted and modified by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from annexed_methods import plot_metrics, plot_metrics_solo

# os.makedirs("results_sim_arg/Metrics_Comparison", exist_ok=True)
# # Size of scans for each generation
log_period = 10
# # It stands for the number of environment steps at each generation: S_size = Batch_size * N_reevaluations
sampling_size = 512

a = "arm"
na = "noisy_arm"
unc = ""
cor = "reeval"

width = 20
height = 7.5


###############################
#             ME              #
###############################

method = "ME"
path_ME = f"results_sim_arg/{method}"
time_ME_a = "01-09-2023_10-43-52"
time_ME_na = "01-09-2023_11-00-54"

# Loading all results from CSV
ME_NonNoisy_Uncorrected = pd.read_csv(f'{path_ME}/{method}_{a}_{time_ME_a}.csv')
ME_Noisy_Uncorrected = pd.read_csv(f'{path_ME}/{method}_{na}_{time_ME_na}.csv')
ME_NonNoisy_Corrected = pd.read_csv(f'{path_ME}/{cor}_{method}_{a}_{time_ME_a}.csv')
ME_Noisy_Corrected = pd.read_csv(f'{path_ME}/{cor}_{method}_{na}_{time_ME_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
ME_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
ME_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
ME_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
ME_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
ME_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
ME_Noisy_Uncorrected.iloc[0, 3] = -0.12
ME_NonNoisy_Corrected.iloc[0, 3] = -0.12
ME_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#             MES             #
###############################

method = "MES"
path_MES = f"results_sim_arg/{method}"
time_MES_a = "30-08-2023_20-10-19"
time_MES_na = "30-08-2023_20-10-27"

MES_NonNoisy_Uncorrected = pd.read_csv(f'{path_MES}/{method}_{a}_{time_MES_a}.csv')
MES_Noisy_Uncorrected = pd.read_csv(f'{path_MES}/{method}_{na}_{time_MES_na}.csv')
MES_NonNoisy_Corrected = pd.read_csv(f'{path_MES}/{cor}_{method}_{a}_{time_MES_a}.csv')
MES_Noisy_Corrected = pd.read_csv(f'{path_MES}/{cor}_{method}_{na}_{time_MES_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MES_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MES_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MES_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MES_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MES_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MES_Noisy_Uncorrected.iloc[0, 3] = -0.12
MES_NonNoisy_Corrected.iloc[0, 3] = -0.12
MES_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#       MEMB (Implicit)       #
###############################

method = "MEMB_Implicit"
path_Imp = f"results_sim_arg/{method}"
time_Imp_a = "30-08-2023_18-16-40"
time_Imp_na = "30-08-2023_18-16-42"

MEMB_Implicit_NonNoisy_Uncorrected = pd.read_csv(f'{path_Imp}/{method}_{a}_{time_Imp_a}.csv')
MEMB_Implicit_Noisy_Uncorrected = pd.read_csv(f'{path_Imp}/{method}_{na}_{time_Imp_na}.csv')
MEMB_Implicit_NonNoisy_Corrected = pd.read_csv(f'{path_Imp}/{cor}_{method}_{a}_{time_Imp_a}.csv')
MEMB_Implicit_Noisy_Corrected = pd.read_csv(f'{path_Imp}/{cor}_{method}_{na}_{time_Imp_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMB_Implicit_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMB_Implicit_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Implicit_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Implicit_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMB_Implicit_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#     MEMB Wipe (Implicit)    #
###############################

method = "MEMB_Implicit_Wipe"
path_ImpW = f"results_sim_arg/{method}"
time_ImpW_a = "30-08-2023_15-55-47"
time_ImpW_na = "30-08-2023_15-55-46"

MEMB_Implicit_Wipe_NonNoisy_Uncorrected = pd.read_csv(f'{path_ImpW}/{method}_{a}_{time_ImpW_a}.csv')
MEMB_Implicit_Wipe_Noisy_Uncorrected = pd.read_csv(f'{path_ImpW}/{method}_{na}_{time_ImpW_na}.csv')
MEMB_Implicit_Wipe_NonNoisy_Corrected = pd.read_csv(f'{path_ImpW}/{cor}_{method}_{a}_{time_ImpW_a}.csv')
MEMB_Implicit_Wipe_Noisy_Corrected = pd.read_csv(f'{path_ImpW}/{cor}_{method}_{na}_{time_ImpW_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMB_Implicit_Wipe_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_Wipe_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_Wipe_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMB_Implicit_Wipe_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMB_Implicit_Wipe_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Implicit_Wipe_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Implicit_Wipe_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMB_Implicit_Wipe_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#    MEMB (Explicit Naive)    #
###############################

method = "MEMB_Explicit_Naive"
path_Exp = f"results_sim_arg/{method}"
time_Exp_a = "30-08-2023_14-39-43"
time_Exp_na = "30-08-2023_14-39-43" 

MEMB_Explicit_Naive_NonNoisy_Uncorrected = pd.read_csv(f'{path_Exp}/{method}_{a}_{time_Exp_a}.csv')
MEMB_Explicit_Naive_Noisy_Uncorrected = pd.read_csv(f'{path_Exp}/{method}_{na}_{time_Exp_na}.csv')
MEMB_Explicit_Naive_NonNoisy_Corrected = pd.read_csv(f'{path_Exp}/{cor}_{method}_{a}_{time_Exp_a}.csv')
MEMB_Explicit_Naive_Noisy_Corrected = pd.read_csv(f'{path_Exp}/{cor}_{method}_{na}_{time_Exp_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMB_Explicit_Naive_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Naive_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Naive_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Naive_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMB_Explicit_Naive_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Explicit_Naive_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Explicit_Naive_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMB_Explicit_Naive_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#  MEMB Wipe (Explicit Naive) #
###############################

method = "MEMB_Explicit_Naive_Wipe"
path_ExpW = f"results_sim_arg/{method}"
time_ExpW_a = "30-08-2023_12-29-43"
time_ExpW_na = "30-08-2023_12-29-48" 

MEMB_Explicit_Wipe_Naive_NonNoisy_Uncorrected = pd.read_csv(f'{path_ExpW}/{method}_{a}_{time_ExpW_a}.csv')
MEMB_Explicit_Wipe_Naive_Noisy_Uncorrected = pd.read_csv(f'{path_ExpW}/{method}_{na}_{time_ExpW_na}.csv')
MEMB_Explicit_Wipe_Naive_NonNoisy_Corrected = pd.read_csv(f'{path_ExpW}/{cor}_{method}_{a}_{time_ExpW_a}.csv')
MEMB_Explicit_Wipe_Naive_Noisy_Corrected = pd.read_csv(f'{path_ExpW}/{cor}_{method}_{na}_{time_ExpW_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMB_Explicit_Wipe_Naive_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Wipe_Naive_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Wipe_Naive_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMB_Explicit_Wipe_Naive_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMB_Explicit_Wipe_Naive_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Explicit_Wipe_Naive_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMB_Explicit_Wipe_Naive_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMB_Explicit_Wipe_Naive_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#       MEMBUQ (Implicit)     # 
###############################

method = "MEMBUQ_Implicit"
path_ImpUQ = f"results_sim_arg/{method}" 
time_ImpUQ_a = "30-08-2023_16-08-30" 
time_ImpUQ_na = "30-08-2023_16-08-40" 

MEMBUQ_Implicit_NonNoisy_Uncorrected = pd.read_csv(f'{path_ImpUQ}/{method}_{a}_{time_ImpUQ_a}.csv')
MEMBUQ_Implicit_Noisy_Uncorrected = pd.read_csv(f'{path_ImpUQ}/{method}_{na}_{time_ImpUQ_na}.csv')
MEMBUQ_Implicit_NonNoisy_Corrected = pd.read_csv(f'{path_ImpUQ}/{cor}_{method}_{a}_{time_ImpUQ_a}.csv')
MEMBUQ_Implicit_Noisy_Corrected = pd.read_csv(f'{path_ImpUQ}/{cor}_{method}_{na}_{time_ImpUQ_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMBUQ_Implicit_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMBUQ_Implicit_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#    MEMBUQ Wipe (Implicit)   # 
###############################

method = "MEMBUQ_Implicit_Wipe"
path_ImpUQW = f"results_sim_arg/{method}" 
time_ImpUQW_a = "31-08-2023_10-29-35" 
time_ImpUQW_na = "31-08-2023_10-29-28" 

MEMBUQ_Implicit_Wipe_NonNoisy_Uncorrected = pd.read_csv(f'{path_ImpUQW}/{method}_{a}_{time_ImpUQW_a}.csv')
MEMBUQ_Implicit_Wipe_Noisy_Uncorrected = pd.read_csv(f'{path_ImpUQW}/{method}_{na}_{time_ImpUQW_na}.csv')
MEMBUQ_Implicit_Wipe_NonNoisy_Corrected = pd.read_csv(f'{path_ImpUQW}/{cor}_{method}_{a}_{time_ImpUQW_a}.csv')
MEMBUQ_Implicit_Wipe_Noisy_Corrected = pd.read_csv(f'{path_ImpUQW}/{cor}_{method}_{na}_{time_ImpUQW_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMBUQ_Implicit_Wipe_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_Wipe_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_Wipe_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMBUQ_Implicit_Wipe_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMBUQ_Implicit_Wipe_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_Wipe_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_Wipe_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMBUQ_Implicit_Wipe_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#    MEMBUQ (Explicit Naive)  #
###############################

method = "MEMBUQ_NLL_Explicit_Naive"
path_ExpUQ = f"results_sim_arg/{method}"
if method == "MEMBUQ_Explicit_Naive":
  time_ExpUQ_a = "02-08-2023_17-25-17"
  time_ExpUQ_na = "02-08-2023_17-40-17"
elif method == "MEMBUQ_NLL_Explicit_Naive":
  time_ExpUQ_a = "30-08-2023_14-49-50"
  time_ExpUQ_na = "30-08-2023_14-49-50"
elif method == "MEMBUQ2_NLL_Explicit_Naive":
  time_ExpUQ_a = "24-08-2023_16-49-18"
  time_ExpUQ_na = "24-08-2023_14-55-18"

MEMBUQ_Explicit_Naive_NonNoisy_Uncorrected = pd.read_csv(f'{path_ExpUQ}/{method}_{a}_{time_ExpUQ_a}.csv')
MEMBUQ_Explicit_Naive_Noisy_Uncorrected = pd.read_csv(f'{path_ExpUQ}/{method}_{na}_{time_ExpUQ_na}.csv')
MEMBUQ_Explicit_Naive_NonNoisy_Corrected = pd.read_csv(f'{path_ExpUQ}/{cor}_{method}_{a}_{time_ExpUQ_a}.csv')
MEMBUQ_Explicit_Naive_Noisy_Corrected = pd.read_csv(f'{path_ExpUQ}/{cor}_{method}_{na}_{time_ExpUQ_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMBUQ_Explicit_Naive_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Naive_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Naive_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Naive_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMBUQ_Explicit_Naive_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Naive_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Naive_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Naive_Noisy_Corrected.iloc[0, 3] = -0.12



##################################
#  MEMBUQ Wipe (Explicit Naive)  #
##################################

method = "MEMBUQ_NLL_Explicit_Naive_Wipe"
path_ExpUQW = f"results_sim_arg/{method}"
time_ExpUQW_a = "31-08-2023_10-05-22"
time_ExpUQW_na = "31-08-2023_10-29-31"

MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Uncorrected = pd.read_csv(f'{path_ExpUQW}/{method}_{a}_{time_ExpUQW_a}.csv')
MEMBUQ_Explicit_Wipe_Naive_Noisy_Uncorrected = pd.read_csv(f'{path_ExpUQW}/{method}_{na}_{time_ExpUQW_na}.csv')
MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Corrected = pd.read_csv(f'{path_ExpUQW}/{cor}_{method}_{a}_{time_ExpUQW_a}.csv')
MEMBUQ_Explicit_Wipe_Naive_Noisy_Corrected = pd.read_csv(f'{path_ExpUQW}/{cor}_{method}_{na}_{time_ExpUQW_na}.csv')

# Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0
MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Wipe_Naive_Noisy_Uncorrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Corrected.iloc[0, [2, 4]] = 0
MEMBUQ_Explicit_Wipe_Naive_Noisy_Corrected.iloc[0, [2, 4]] = 0
# Set the value of first row and columns 4 (max fitness) to -0.12
MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Wipe_Naive_Noisy_Uncorrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Corrected.iloc[0, 3] = -0.12
MEMBUQ_Explicit_Wipe_Naive_Noisy_Corrected.iloc[0, 3] = -0.12



###############################
#           MERGING           #
###############################

labels = [
  "ME", 
  "MES", 
  "MBME Implicit", 
  "MBME Explicit Naive", 
  "MBMEUQ Implicit", 
  "MBMEUQ Explicit Naive", 
  "MBME Implicit Wipe", 
  "MBME Explicit Naive Wipe", 
  "MBMEUQ Implicit Wipe", 
  "MBMEUQ Explicit Naive Wipe"
]

determ_unc_met = [
  ME_NonNoisy_Uncorrected, 
  MES_NonNoisy_Uncorrected, 
  MEMB_Implicit_NonNoisy_Uncorrected, 
  MEMB_Explicit_Naive_NonNoisy_Uncorrected, 
  MEMBUQ_Implicit_NonNoisy_Uncorrected, 
  MEMBUQ_Explicit_Naive_NonNoisy_Uncorrected,
  MEMB_Implicit_Wipe_NonNoisy_Uncorrected, 
  MEMB_Explicit_Wipe_Naive_NonNoisy_Uncorrected, 
  MEMBUQ_Implicit_Wipe_NonNoisy_Uncorrected, 
  MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Uncorrected
]

determ_cor_met = [
  ME_NonNoisy_Corrected, 
  MES_NonNoisy_Corrected, 
  MEMB_Implicit_NonNoisy_Corrected, 
  MEMB_Explicit_Naive_NonNoisy_Corrected, 
  MEMBUQ_Implicit_NonNoisy_Corrected, 
  MEMBUQ_Explicit_Naive_NonNoisy_Corrected,
  MEMB_Implicit_Wipe_NonNoisy_Corrected, 
  MEMB_Explicit_Wipe_Naive_NonNoisy_Corrected, 
  MEMBUQ_Implicit_Wipe_NonNoisy_Corrected, 
  MEMBUQ_Explicit_Wipe_Naive_NonNoisy_Corrected
]

uncert_unc_met = [
  ME_Noisy_Uncorrected, 
  MES_Noisy_Uncorrected, 
  MEMB_Implicit_Noisy_Uncorrected, 
  MEMB_Explicit_Naive_Noisy_Uncorrected, 
  MEMBUQ_Implicit_Noisy_Uncorrected, 
  MEMBUQ_Explicit_Naive_Noisy_Uncorrected,
  MEMB_Implicit_Wipe_Noisy_Uncorrected, 
  MEMB_Explicit_Wipe_Naive_Noisy_Uncorrected, 
  MEMBUQ_Implicit_Wipe_Noisy_Uncorrected, 
  MEMBUQ_Explicit_Wipe_Naive_Noisy_Uncorrected
]

uncert_cor_met = [
  ME_Noisy_Corrected, 
  MES_Noisy_Corrected, 
  MEMB_Implicit_Noisy_Corrected, 
  MEMB_Explicit_Naive_Noisy_Corrected, 
  MEMBUQ_Implicit_Noisy_Corrected, 
  MEMBUQ_Explicit_Naive_Noisy_Corrected,
  MEMB_Implicit_Wipe_Noisy_Corrected, 
  MEMB_Explicit_Wipe_Naive_Noisy_Corrected, 
  MEMBUQ_Implicit_Wipe_Noisy_Corrected, 
  MEMBUQ_Explicit_Wipe_Naive_Noisy_Corrected
]



###############################
#      A. DETERMINISTIC       #
###############################

determ_task_name = 'Deterministic Arm'
fig_determ_unc, axes_determ_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
determ_unc_path = 'results_sim_arg/Metrics_Comparison/Deterministic_Uncorrected_Metrics_Plots.png'
determ_metric_unc = 'Uncorrected'

plot_metrics_solo(
  fig=fig_determ_unc, 
  ax=axes_determ_unc, 
  methods=determ_unc_met, 
  labels=labels, 
  log_period=log_period, 
  sampling_size=sampling_size, 
  task_name=determ_task_name,
  task_met=determ_metric_unc, 
  path=determ_unc_path
)

fig_determ_cor, axes_determ_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
determ_cor_path = 'results_sim_arg/Metrics_Comparison/Deterministic_Corrected_Metrics_Plots.png'
determ_metric_cor = 'Corrected'

plot_metrics_solo(
  fig=fig_determ_cor, 
  ax=axes_determ_cor, 
  methods=determ_cor_met, 
  labels=labels, 
  log_period=log_period, 
  sampling_size=sampling_size, 
  task_name=determ_task_name,
  task_met=determ_metric_cor, 
  path=determ_cor_path
)



###############################
#       B. UNCERTAIN          #
###############################

uncert_task_name = 'Uncertain Arm'
fig_uncert_unc, axes_uncert_unc = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
uncert_unc_path = 'results_sim_arg/Metrics_Comparison/Uncertain_Uncorrected_Metrics_Plots.png'
uncert_metric_unc = 'Uncorrected'

plot_metrics_solo(
  fig=fig_uncert_unc, 
  ax=axes_uncert_unc, 
  methods=uncert_unc_met, 
  labels=labels, 
  log_period=log_period, 
  sampling_size=sampling_size, 
  task_name=uncert_task_name,
  task_met=uncert_metric_unc, 
  path=uncert_unc_path
)

fig_uncert_cor, axes_uncert_cor = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
uncert_cor_path = 'results_sim_arg/Metrics_Comparison/Uncertain_Corrected_Metrics_Plots.png'
uncert_metric_cor = 'Corrected'

plot_metrics_solo(
  fig=fig_uncert_cor, 
  ax=axes_uncert_cor, 
  methods=uncert_cor_met, 
  labels=labels, 
  log_period=log_period, 
  sampling_size=sampling_size, 
  task_name=uncert_task_name,
  task_met=uncert_metric_cor, 
  path=uncert_cor_path
)
