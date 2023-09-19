"""
This work is based on the QDax framework: https://github.com/adaptive-intelligent-robotics/QDax
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}

This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from annexed_methods import process_replication, plot_replicated, calculate_loss_metrics, plot_loss_metrics, process_replication_stds, plot_replicated_stds

methods_to_replicate = [
    'ME', 
    'MES', 
    'MEMB_Implicit', 'MEMB_Implicit_Wipe', 
    'MEMB_Explicit_Naive', 'MEMB_Explicit_Naive_Wipe', 
    'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 
    'MEMBUQ_Implicit', 'MEMBUQ_Implicit_Wipe'
] 

# Values use to have the same initialization between algorithms
preprocess_values = {
    'qd_score': 0,
    'max_fitness': -0.12,
    'coverage': 0
}

# Columns that we want to average and compute std between the 5 replications
metrics_columns = ['qd_score', 'max_fitness', 'coverage']

# All the columns in our new replicated DF
columns = ['iteration', 'qd_score', 'std_qd_score', 'max_fitness', 'std_max_fitness', 'coverage', 'std_coverage']

log_period = 10
sampling_size = 512
saving_path = 'results_sim_arg/Metrics_Comparison/Replicated'
take_avg = False

process_replication(
    methods=methods_to_replicate, 
    dict_values=preprocess_values, 
    metrics_columns=metrics_columns,
    rep_avg=take_avg
)


### No Reset
# methods_to_plot = ['ME', 'MES', 'MEMB_Implicit', 'MEMB_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_Implicit'] 
# check_reset = False

### Only Reset
methods_to_plot = ['ME', 'MES', 'MEMB_Implicit_Wipe', 'MEMB_Explicit_Naive_Wipe', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 'MEMBUQ_Implicit_Wipe'] 
# methods_to_plot = ['ME', 'MES', 'MEMBUQ_NLL_Explicit_Naive_Wipe']
check_reset = True

### Everything
# methods_to_plot = [
#     'ME', 
#     'MES', 
#     'MEMB_Implicit', 'MEMB_Implicit_Wipe', 
#     'MEMB_Explicit_Naive', 'MEMB_Explicit_Naive_Wipe', 
#     'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 
#     'MEMBUQ_Implicit', 'MEMBUQ_Implicit_Wipe'
# ] 
# check_reset = True # Don't really care here

if len(methods_to_plot) == 6 or len(methods_to_plot) == 3:
    check = False
elif len(methods_to_plot) == 10:
    check = True

plot_replicated(
    saving_path=saving_path, 
    all_columns=columns, 
    metrics_columns=metrics_columns, 
    log_period=log_period, 
    sampling_size=sampling_size, 
    methods=methods_to_plot,
    check_full=check,
    check_reset=check_reset # True indicates we use only reset results
)


### No Reset
# methods_loss = ['ME', 'MES', 'MEMB_Implicit', 'MEMB_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_Implicit'] 

### Only Reset
# methods_loss = ['ME', 'MES', 'MEMB_Implicit_Wipe', 'MEMB_Explicit_Naive_Wipe', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 'MEMBUQ_Implicit_Wipe'] 

### Everything
methods_loss = [
    'ME', 
    'MES', 
    'MEMB_Implicit', 'MEMB_Implicit_Wipe', 
    'MEMB_Explicit_Naive', 'MEMB_Explicit_Naive_Wipe', 
    'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 
    'MEMBUQ_Implicit', 'MEMBUQ_Implicit_Wipe'
] 

metrics_columns = ['qd_score', 'max_fitness', 'coverage']

calculate_loss_metrics(
    methods=methods_loss, 
    metrics_columns=metrics_columns
)

saving_path_loss = f'results_sim_arg/Loss_Comparison'

plot_loss_metrics(
    methods=methods_loss,
    saving_path=saving_path_loss
)


#################
#      STDS     #
#################

methods_to_replicate_stds = [
    'MES', 
    'MEMBUQ_NLL_Explicit_Naive', 'MEMBUQ_NLL_Explicit_Naive_Wipe', 
    'MEMBUQ_Implicit', 'MEMBUQ_Implicit_Wipe'
] 

# methods_to_replicate_stds = [
#     'MES', 
#     'MEMBUQ_NLL_Explicit_Naive_Wipe'
# ] 

# Columns that we want to average and compute std between the 5 replications
metrics_columns_stds = ['std_fitness','std_bd']

process_replication_stds(
    methods=methods_to_replicate_stds, 
    metrics_columns=metrics_columns_stds, 
    rep_avg=take_avg
)

saving_path_stds = 'results_sim_arg/Metrics_Comparison/Replicated_Stds'

# All the columns in our new replicated DF
columns_stds = ['iteration', 'std_fitness', 'std_std_fitness', 'std_bd', 'std_std_bd']

plot_replicated_stds(
    saving_path=saving_path_stds, 
    all_columns=columns_stds, 
    metrics_columns=metrics_columns_stds, 
    log_period=log_period, 
    sampling_size=sampling_size, 
    methods=methods_to_replicate_stds, 
)
