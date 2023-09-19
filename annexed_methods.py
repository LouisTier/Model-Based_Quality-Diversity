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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qdax.utils.plotting import plot_2d_map_elites_repertoire
import ast
import jax.numpy as jnp

# Used in main scripts (ME/S/MB_arm.py)
def create_directory(directory_path):
    """
    Create a folder given a path.

    Args:
        - directory_path (str): Path of the new folder to be created

    Returns:
        - Nothing but creates the corresponding folder.
    """

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Used in figure_handling.py
def retrieve_png(directory):
    """
    Retrieve all the png files in a folder given its path.

    Args:
        - directory (str): Path of the folder containing the png files

    Returns:
        - png_files (list): List containing all the png files
    """

    png_files = []
    for file in os.listdir(directory):
        if file.endswith(".png"):
            png_files.append(file)
    return png_files

# Used in figure_handling.py
def add_path_to_images(directory, images):
    """
    Merge the path of the folder with each png file it contains 

    Args:
        - directory (str): Path of the folder containing the png files
        - images (list): List of all the png files in the directory

    Returns:
        - img_path (list): List containing the path of each image
    """

    img_path = [os.path.join(directory, img) for img in images]
    return img_path

# Used in analysis_hyperparameters.py
def find_txt(path):
    """
    Retrieve all the txt files given the path of a folder 

    Args:
        - path (str): Path of the folder containing the txt files

    Returns:
        - txt_files (list): List containing all the txt files 
    """

    txt_files = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(path, file))
    return txt_files

# Used in NN_analysis.py
def replace_NAN(input_list, epsilon=1e-7):
    """
    Replace NaN values of a list by an epsilon values. 

    Args:
        - input_list (str): List to be analyzed
        - epsilon (float): Value used to replace NaN values

    Returns:
        - new_list (list): Same list as input without NaN values 
    """

    new_list = jnp.where(jnp.isnan(input_list), epsilon, input_list)
    return new_list

# Used in NN_analysis.py
def moving_avg(input_list, batch_size, bool_fit):
    """
    Calculate the average values of an input list whose window is based on a batch size. 

    Args:
        - input_list (str): List to be averaged
        - batch_size (int): Size of the window to average elements of the input list
        - bool_fit (boolean): Condition to deal with fitnesses (1D) or BDs (2D)

    Returns:
        - output_list (list): Averaged list 
    """

    epsilon = 1e-7
    output_list = []
    if bool_fit == True:
        for i in range(0, len(input_list), batch_size):
            avg_window = input_list[i:i+batch_size]
            avg_window = jnp.where(jnp.isnan(avg_window), epsilon, avg_window)
            # print(f"window n° {i}: {avg_window}")
            # print(f'window sum: {jnp.sum(avg_window)}')
            avg_value = jnp.mean(avg_window)
            output_list.append(avg_value)
            # print(f"value n° {i}: {avg_value}")
        output_list = jnp.array(output_list)
    else:
        for i in range(0, len(input_list), batch_size):
            avg_window = input_list[i:i+batch_size]
            avg_window = jnp.where(jnp.isnan(avg_window), epsilon, avg_window)
            avg_value = jnp.mean(avg_window, axis=0)
            output_list.append(avg_value)
        output_list = jnp.array(output_list)
    return output_list

# Used in NN_analysis.py
def plot_errors(input_list, marker_size, saving_path, bool_fit, method, name):
    """
    Plot errors values between evaluations of the scoring function and predictions of the model. 

    Args:
        - input_list (str): List to be plotted
        - marker_size (int): Size of the dots used in the plot
        - saving_path (str): Path and name of the plot to be saved
        - bool_fit (boolean): Condition to deal with fitnesses (1D) or BDs (2D)
        - method (str): Name of the algorithm, so MBME_Explicit/Implicit or UQ version
        - name (str): Name of the task, so arm or noisy_arm

    Returns:
        - Nothing but saves the corresponding plot 
    """

    fig = plt.figure(figsize=(10,8))
    plt.xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')

    if bool_fit == True:
        plt.scatter(range(len(input_list)), input_list, label='Avg Fitness Error', c='orange', s=marker_size)
        plt.ylabel('Average Absolute Fitness Error', fontsize=12, fontweight='bold')
        plt.title(f"Evolution of the average absolute fitness error with {method} \nwhen model is not training for the {name}", fontsize=14, fontweight='bold')
    else:
        plt.scatter(range(len(input_list)), input_list[:,0], label='Avg BD Error - X Position', c='orange', s=marker_size)
        plt.scatter(range(len(input_list)), input_list[:,1], label='Avg BD Error - Y Position', c='blue', s=marker_size)
        plt.ylabel('Average Absolute BD Errors', fontsize=12, fontweight='bold')
        plt.title(f"Evolution of the average absolute BD errors with {method} \nwhen model is not training for the {name}", fontsize=14, fontweight='bold')

    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(f'{saving_path}')

# Used in NN_analysis.py
def plot_errors_std(input_list, marker_size, saving_path, bool_gen, bool_fit, method, name):
    """
    Plot std errors values between evaluations of the scoring function and predictions of the model. 

    Args:
        - input_list (str): List to be plotted
        - marker_size (int): Size of the dots used in the plot
        - saving_path (str): Path and name of the plot to be saved
        - bool_gen (boolean): Condition to deal with final test set or predictions between generation
        - bool_fit (boolean): Condition to deal with fitnesses (1D) or BDs (2D)
        - method (str): Name of the algorithm, so MBME_Explicit/Implicit or UQ version
        - name (str): Name of the task, so arm or noisy_arm

    Returns:
        - Nothing but saves the corresponding plot
    """

    fig = plt.figure(figsize=(10,8))

    if bool_fit == True:
        plt.scatter(range(len(input_list)), input_list, label='Avg Fitness Std Error', c='orange', s=marker_size)
        plt.ylabel('Average Absolute Fitness Std Error', fontsize=12, fontweight='bold')
        if bool_gen == True:
            plt.xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
            plt.title(f"Evolution of the average absolute fitness Std error with {method} \nwhen model is not training for the {name}", fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Index of the predictions (last test set)', fontsize=12, fontweight='bold')
            plt.title(f'Average absolute fitness Std error with {method} \nfor the last training and the {name}', fontsize=14, fontweight='bold')
    
    else:
        plt.scatter(range(len(input_list)), input_list, label='Avg BD Std Error', c='orange', s=marker_size)
        plt.ylabel('Average Absolute BD Std Error', fontsize=12, fontweight='bold')
        if bool_gen == True:
            plt.xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
            plt.title(f"Evolution of the average absolute BD Std error with {method} \nwhen model is not training for the {name}", fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Index of the predictions (last test set)', fontsize=12, fontweight='bold')
            plt.title(f'Average absolute BD Std error with {method} \nfor the last training and the {name}', fontsize=14, fontweight='bold')

        plt.title(f"Evolution of the average absolute BD Std errors with {method} \nwhen model is not training for the {name}", fontsize=14, fontweight='bold')

    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(f'{saving_path}')

# Used in NN_analysis.py
def plot_comparison(saving_path, scoring_list, model_list, bool_gen, bool_fit, name, method):
    """
    Plot comparisons between std evaluations of the scoring function and std predictions of the model. 

    Args:
        - saving_path (str): Path and name of the plot to be saved
        - scoring_list (list): List containing the evaluation from the scoring function
        - model_list (list): List containing the predictions from the model
        - bool_gen (boolean): Condition to deal with final test set or predictions between generation
        - bool_fit (boolean): Condition to deal with fitnesses (1D) or BDs (2D)
        - method (str): Name of the algorithm, so MBME_Explicit/Implicit or UQ version
        - name (str): Name of the task, so arm or noisy_arm

    Returns:
        - Nothing but saves the corresponding plot
    """

    fig = plt.figure(figsize=(10,8))

    if bool_fit == True:
        plt.scatter(range(len(scoring_list)), scoring_list, label='Fit Std Scoring Function')
        plt.scatter(range(len(model_list)), model_list, label='Fit Std Model')
        plt.ylabel('Average Fitness Std', fontsize=12, fontweight='bold')
        if bool_gen == True:
            plt.xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
            plt.title(f"Comparison of fitness std predictions between scoring function and \nmodel outside trainings with {method} and {name}", fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Index of the predictions (last test set)', fontsize=12, fontweight='bold')
            plt.title(f"Comparison of fitness std predictions between scoring function and \nmodel with last test set, {method} and {name}", fontsize=14, fontweight='bold')

    else:
        plt.ylabel('Average BD Std', fontsize=12, fontweight='bold')
        plt.scatter(range(len(scoring_list)), scoring_list, label='BD Std Scoring Function')
        plt.scatter(range(len(model_list)), model_list, label='BD Std Model')
        if bool_gen == True:
            plt.xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
            plt.title(f"Comparison of BD std predictions between scoring function and \nmodel outside trainings with {method} and {name}", fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Index of the predictions (last test set)', fontsize=12, fontweight='bold')
            plt.title(f"Comparison of BD std predictions between scoring function and \nmodel with last test set, {method} and {name}", fontsize=14, fontweight='bold')

    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(f'{saving_path}')

# Used in analysis_repertoire.py
def plot_repertoires(fig, repertoires_fig, repertoires_names, min_bd, max_bd, path):
    """
    Plot repertoires obtained with different algorithms. 

    Args:
        - fig: Figure used to plot the graphs
        - repertoires_fig: List containing all the repertoires to be plotted
        - repertoires_names: List containing all the names of repertoires to be plotted
        - min_bd (float): Minimum values for the BD
        - max_bd (float): Maximum values for the BD
        - path (str): Path and name of the plot to be saved

    Returns:
        - Nothing but saves the corresponding plot
    """

    for i in range(1, 9):
        ax = fig.add_subplot(2, 4, i)
        repertoire = repertoires_fig[i-1]
        _, ax = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=min_bd,
            maxval=max_bd,
            repertoire_descriptors=repertoire.descriptors,
            ax=ax,
        )
        ax.set_xlabel("Behaviour Dimension 1: X", fontsize=12, fontweight='bold')
        ax.set_ylabel("Behaviour Dimension 2: Y", fontsize=12, fontweight='bold')
        ax.set_title(repertoires_names[i-1], fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_repertoire.py
def plot_repertoires_solo(fig, repertoires_fig, repertoires_names, min_bd, max_bd, report_display, path, min_fit=None, max_fit=None):
    """
    Plot repertoires obtained with different algorithms, but only given a specific task, so arm or noisy arm
    and specific metrics, so corrected or uncorrected. 

    Args:
        - fig: Figure used to plot the graphs
        - repertoires_fig: List containing all the repertoires to be plotted
        - repertoires_names: List containing all the names of repertoires to be plotted
        - min_bd (float): Minimum values for the BD
        - max_bd (float): Maximum values for the BD
        - report_display (boolean): Checking if displaying results 1x6 or 2x3
        - path (str): Path and name of the plot to be saved
        - min_fit (float): Minimum values for the fitness
        - max_fit (float): Maximum values for the fitness
        
    Returns:
        - Nothing but saves the corresponding plot
    """

    # for i in range(1, len(repertoires_names)+1):
    for i in range(1, 3*4+1):
        idx = i-3
        if report_display == True:
            # ax = fig.add_subplot(2, 3, i)
            ax = fig.add_subplot(3, 4, i)
        else:
            ax = fig.add_subplot(1, len(repertoires_names), i)

        if i == 2:
            idx = 0
        elif i == 3:
            idx = 1

        if i not in [1, 4]:
            repertoire = repertoires_fig[idx]
            _, ax = plot_2d_map_elites_repertoire(
                centroids=repertoire.centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=min_bd,
                maxval=max_bd,
                vmin=min_fit,
                vmax=max_fit,
                repertoire_descriptors=repertoire.descriptors,
                ax=ax,
            )
            ax.set_xlabel("Behaviour Dimension 1: X", fontsize=12, fontweight='bold')
            ax.set_ylabel("Behaviour Dimension 2: Y", fontsize=12, fontweight='bold')
            ax.set_title(repertoires_names[idx], fontsize=14, fontweight='bold')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_repertoire.py
def plot_repertoires_solo_best(fig, repertoires_fig, repertoires_names, min_bd, max_bd, path, min_fit=None, max_fit=None):
    """
    Plot repertoires obtained with MBME ad MBMEUQ algorithms, but only given a specific task, so arm or noisy arm
    and specific metrics, so corrected or uncorrected. 

    Args:
        - fig: Figure used to plot the graphs
        - repertoires_fig: List containing all the repertoires to be plotted
        - repertoires_names: List containing all the names of repertoires to be plotted
        - min_bd (float): Minimum values for the BD
        - max_bd (float): Maximum values for the BD
        - path (str): Path and name of the plot to be saved
        - min_fit (float): Minimum values for the fitness
        - max_fit (float): Maximum values for the fitness
        
    Returns:
        - Nothing but saves the corresponding plot
    """
    
    for i in range(1, len(repertoires_names)+1):
        ax = fig.add_subplot(1, 3, i)  
        repertoire = repertoires_fig[i-1]
        _, ax = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=min_bd,
            maxval=max_bd,
            vmin=min_fit,
            vmax=max_fit,
            repertoire_descriptors=repertoire.descriptors,
            ax=ax,
        )
        ax.set_xlabel("Behaviour Dimension 1: X", fontsize=12, fontweight='bold')
        ax.set_ylabel("Behaviour Dimension 2: Y", fontsize=12, fontweight='bold')
        ax.set_title(repertoires_names[i-1], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_repertoire.py
def plot_repertoires_solo_UQ(fig, repertoires_fig, repertoires_names, min_bd, max_bd, path, min_fit=None, max_fit=None):
    """
    Plot repertoires obtained with MBME ad MBMEUQ algorithms, but only given a specific task, so arm or noisy arm
    and specific metrics, so corrected or uncorrected. 

    Args:
        - fig: Figure used to plot the graphs
        - repertoires_fig: List containing all the repertoires to be plotted
        - repertoires_names: List containing all the names of repertoires to be plotted
        - min_bd (float): Minimum values for the BD
        - max_bd (float): Maximum values for the BD
        - path (str): Path and name of the plot to be saved
        - min_fit (float): Minimum values for the fitness
        - max_fit (float): Maximum values for the fitness
        
    Returns:
        - Nothing but saves the corresponding plot
    """
    
    for i in range(1, len(repertoires_names)+1):
        ax = fig.add_subplot(2, 2, i)  
        repertoire = repertoires_fig[i-1]
        _, ax = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=min_bd,
            maxval=max_bd,
            vmin=min_fit,
            vmax=max_fit,
            repertoire_descriptors=repertoire.descriptors,
            ax=ax,
        )
        ax.set_xlabel("Behaviour Dimension 1: X", fontsize=12, fontweight='bold')
        ax.set_ylabel("Behaviour Dimension 2: Y", fontsize=12, fontweight='bold')
        ax.set_title(repertoires_names[i-1], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_repertoire.py
def filter_repertoire(repertoire):
    """
    Filter repertoires from undefined values (-inf), keeping only fitnesses coming from an evaluation or a prediction. 

    Args:
        - repertoire (MapElitesRepertoire): Repertoire to be analyzed
        
    Returns:
        - filtered_repertoires (list): List containing the fitness values different from -inf
    """

    filtered_repertoires = [fit_val for fit_val in repertoire if fit_val != float("-inf")]
    return filtered_repertoires

# Used in analysis_metrics.py
def plot_metrics(fig, ax, methods, labels, log_period, sampling_size, task_name, path):
    """
    Plot metrics obtained with different algorithms. 

    Args:
        - fig: Figure used to plot the graphs
        - ax: Axes used to plot the graphs
        - methods (list): List containing the metrics of the different algorithms
        - labels (list): List containing the names of the different algorithms
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms
        - task_name (str): Name of the task, so arm or noisy_arm
        - path (str): Path and name of the plot to be saved
        
    Returns:
        - Nothing but saves the corresponding plot
    """

    counter = 0
    if task_name == 'Deterministic Arm':
        idx = 1
    elif task_name == 'Uncertain Arm':
        idx = 2

    for i in range(0,2):
        if i == 0:
            sample = 'Uncorrected'
        elif i == 1:
            sample = 'Corrected'

        for j in range(0,3):
            counter += 1
            if j == 0:
                metric = 'qd_score'
                title = 'QD Score'
                ylabel = title
            elif j == 1:
                metric = 'max_fitness'
                title = 'Max Fitness'
                ylabel = title
            elif j == 2:
                metric = 'coverage'
                title = 'Coverage'
                ylabel = 'Coverage in %'

            for k in range(len(methods[i])):
                ax[i,j].plot(methods[i][k]['loop']*log_period*sampling_size, methods[i][k][metric], label=labels[k])
            
            ax[i,j].set_xlabel("Environment Steps", fontsize=12, fontweight='bold')
            ax[i,j].set_ylabel(f"{sample} {ylabel}", fontsize=12, fontweight='bold')
            ax[i,j].set_title(f"{sample} {title} evolution during training \n {task_name} - F{idx}.{counter}", fontsize=14, fontweight='bold')
            ax[i,j].legend(loc='lower right')
    
    fig.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_metrics.py
def plot_metrics_solo(fig, ax, methods, labels, log_period, sampling_size, task_name, task_met, path):
    """
    Plot metrics obtained with different algorithms and specific metric calculations, so corrected or uncorrected. 

    Args:
        - fig: Figure used to plot the graphs
        - ax: Axes used to plot the graphs
        - methods (list): List containing the metrics of the different algorithms
        - labels (list): List containing the names of the different algorithms
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms
        - task_name (str): Name of the task, so arm or noisy_arm
        - task_met (str): Name of the metric, so corrected or uncorrected
        - path (str): Path and name of the plot to be saved
        
    Returns:
        - Nothing but saves the corresponding plot
    """

    counter = 0

    if task_name == 'Deterministic Arm':
        idx = 1
    elif task_name == 'Uncertain Arm':
        idx = 2

    if task_met == 'Uncorrected':
        sample = 'Uncorrected'
    elif task_met == 'Corrected':
        sample = 'Corrected'
        counter += 3

    for j in range(0,3):
        counter += 1
        if j == 0:
            metric = 'qd_score'
            title = 'QD Score'
            ylabel = title
        elif j == 1:
            metric = 'max_fitness'
            title = 'Max Fitness'
            ylabel = title
        elif j == 2:
            metric = 'coverage'
            title = 'Coverage'
            ylabel = 'Coverage in %'

        for k in range(len(methods)):
            ax[j].plot(methods[k]['loop']*log_period*sampling_size, methods[k][metric], label=labels[k])
        
        ax[j].set_xlabel("Environment Steps", fontsize=12, fontweight='bold')
        ax[j].set_ylabel(f"{ylabel}", fontsize=12, fontweight='bold')
        ax[j].set_title(f"{sample} {title} evolution during training \n {task_name} - F{idx}.{counter}", fontsize=14, fontweight='bold')
        ax[j].legend(loc='lower right')
    
    fig.tight_layout()
    plt.show()
    plt.savefig(path)

# Used in analysis_hyperparameters.py
def plot_metrics_solo_hyper(fig, ax, methods, labels, log_period, sampling_size, task_name, task_met, algo_name, show_legend=True):
    """
    Plot metrics obtained with one algorithm but different hyperparameters from the Grid Search, and specific metric calculations, 
    so corrected or uncorrected. 

    Args:
        - fig: Figure used to plot the graphs
        - ax: Axes used to plot the graphs
        - methods (list): List containing the metrics of the different simulations of a same algorithm
        - labels (list): List containing the names of the different hyperparameters combinations
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms
        - task_name (str): Name of the task, so arm or noisy_arm
        - task_met (str): Name of the metric, so corrected or uncorrected
        - algo_name (str): Name of the algorithm used
        - show_legend (boolean): Checking if dealing with the whole results or only the top k results
        
    Returns:
        - fig
        - ax 
    """

    counter = 0

    if task_name == 'Deterministic Arm':
        idx = 1
    elif task_name == 'Uncertain Arm':
        idx = 2

    if task_met == 'Uncorrected':
        sample = 'Uncorrected'
    elif task_met == 'Corrected':
        sample = 'Corrected'
        counter += 3

    for j in range(0,3):
        counter += 1
        if j == 0:
            metric = 'qd_score'
            title = 'QD Score'
            ylabel = title
        elif j == 1:
            metric = 'max_fitness'
            title = 'Max Fitness'
            ylabel = title
        elif j == 2:
            metric = 'coverage'
            title = 'Coverage'
            ylabel = 'Coverage in %'

        # for k in range(0,3):
        ax[j].plot(methods['loop']*log_period*sampling_size, methods[metric], label=labels)
        ax[j].set_xlabel("Environment Steps", fontsize=12, fontweight='bold')
        ax[j].set_ylabel(f"{ylabel}", fontsize=12, fontweight='bold')
        # ax[j].set_title(f"{sample} {title} evolution during training \n {task_name} - F{idx}.{counter}", fontsize=14, fontweight='bold')
        ax[j].set_title(f"{sample} {title} evolution during training \n {algo_name} - {task_name}", fontsize=14, fontweight='bold')
        if show_legend:
            ax[j].legend(loc='lower right')
    
    return fig, ax

# Used in analysis_hyperparameters.py
def retrieve_elements(txt_list, file_path, method, task_name, start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list):
    """
    Retrieve hyperparameters and results from a GridSearch simulation of a specific algorithm.

    Args:
        - txt_list (list): List of all the txt files containing the list of hyperparameters used to run a simulation
        - file_path (str): Name of the path where results are stored in csv
        - method (str): Name of the algorithm used, so MBME(UQ) Implicit or Explicit
        - task_name (str): Name of the task, so arm or noisy arm
        - start_training_list (list): List of the First Training hyperparameter
        - nb_epochs_list (list): List of the Number of Epochs hyperparameter
        - per_data_list (list): List of the % of Old Data retained between two trainings hyperparameter
        - time_list (list): List of the time of the different simulations
        - csv_unc_list (list): List of the csv where results are stored for the uncorrected metrics
        - csv_cor_list (list): List of the csv where results are stored for the corrected metrics      
        
    Returns:
        - start_training_list (list): List filled in with the corresponding hyperparameter for each simulation
        - nb_epochs_list (list): List filled in with the corresponding hyperparameter for each simulation
        - per_data_list (list): List filled in with the corresponding hyperparameter for each simulation
        - time_list (list): List filled in with the corresponding hyperparameter for each simulation
        - csv_unc_list, (list): List filled in with the corresponding preprocessed results and metrics for each simulation
        - csv_cor_list (list): List filled in with the corresponding preprocessed results and metrics for each simulation
    """

    # Looping over all txt file to retrieved values and csv
    for txt_path in txt_list:
        
        with open(txt_path, 'r') as file:
            txt_data = ast.literal_eval(file.read()) # Convert str into dict

        # Retrieving hyperparameters values
        time = txt_data['beginning_time']
        start_training = txt_data['first_train']
        nb_epochs = txt_data['num_epochs']
        per_data = txt_data['per_data']

        # Loading corresponding csvs: uncorrected and corrected
        csv_unc = pd.read_csv(f'{file_path}/{method}_{task_name}_{time}.csv')
        csv_cor = pd.read_csv(f'{file_path}/reeval_{method}_{task_name}_{time}.csv')
        
        # Set the value of first row and columns 3 (qd_score) and 5 (coverage) to 0, and columns 4 (max fitness) to -0.12
        csv_unc.iloc[0, [2, 4]] = 0
        csv_unc.iloc[0, 3] = -0.12
        csv_cor.iloc[0, [2, 4]] = 0
        csv_cor.iloc[0, 3] = -0.12

        start_training_list.append(start_training)
        nb_epochs_list.append(nb_epochs)
        per_data_list.append(per_data)
        csv_unc_list.append(csv_unc)
        csv_cor_list.append(csv_cor)
        time_list.append(time)
    
    return start_training_list, nb_epochs_list, per_data_list, time_list, csv_unc_list, csv_cor_list

# Used in analysis_hyperparameters.py
def plot_hyperparam(log_period, sampling_size, algo_name, full_name, save_unc, save_cor, fing_unc, ax_unc, fig_cor, ax_cor, start_training_list_unc, nb_epochs_list_unc, per_data_list_unc, csv_unc_list, csv_cor_list, start_training_list_cor, nb_epochs_list_cor, per_data_list_cor, show_legend):
    """
    Plot the uncorrected and corrected metrics of a specific GridSearch simulation, so a specific algorithm.

    Args:
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms 
        - algo_name (str): Name of the algorithm used for the GridSearch
        - full_name (str): Name of the task, so arm or noisy arm
        - save_unc (str): Path and name of the figure saved for the uncorrected metrics 
        - save_cor (str): Path and name of the figure saved for the corrected metrics
        - fing_unc: Figure used to plot the uncorrected metrics
        - ax_unc: Axes used to plot the uncorrected metrics
        - fig_cor: Figure used to plot the corrected metrics
        - ax_cor: Axes used to plot the corrected metrics
        - start_training_list_unc (list): List of the First Training hyperparameter for the uncorrected metrics
        - nb_epochs_list_unc (list): List of the Number of Epochs hyperparameter for the uncorrected metrics
        - per_data_list_unc (list): List of the % of Old Data retained between two trainings hyperparameter for the uncorrected metrics
        - csv_unc_list (list): List of the csv where results are stored for the uncorrected metrics
        - csv_cor_list (list): List of the csv where results are stored for the corrected metrics
        - start_training_list_cor (list): List of the First Training hyperparameter for the corrected metrics
        - nb_epochs_list_cor (list): List of the Number of Epochs hyperparameter for the corrected metrics 
        - per_data_list_cor (list): List of the % of Old Data retained between two trainings hyperparameter for the corrected metrics
        - show_legend (boolean): Checking if dealing with the whole results or only the top k results 

    Returns:
        - Nothing but saves the uncorrected and corrected metrics plots
    """

    # Looping over all simulations 
    for i in range(len(start_training_list_unc)):

        # Plot Uncorrected Metrics
        fing_unc, ax_unc = plot_metrics_solo_hyper(
            fig=fing_unc, 
            ax=ax_unc,
            methods=csv_unc_list[i], 
            labels=f"Start training: {start_training_list_unc[i]} | Nb epochs: {nb_epochs_list_unc[i]} | Old data: {100*per_data_list_unc[i]}%", 
            log_period=log_period, 
            sampling_size=sampling_size, 
            task_name=full_name, 
            task_met="Uncorrected", 
            show_legend=show_legend,
            algo_name=algo_name
        )

        # Plot Corrected Metrics
        fig_cor, ax_cor = plot_metrics_solo_hyper(
            fig=fig_cor, 
            ax=ax_cor,
            methods=csv_cor_list[i], 
            labels=f"Start training: {start_training_list_cor[i]} | Nb epochs: {nb_epochs_list_cor[i]} | Old data: {100*per_data_list_cor[i]}%", 
            log_period=log_period, 
            sampling_size=sampling_size, 
            task_name=full_name, 
            task_met="Corrected",
            show_legend=show_legend, 
            algo_name=algo_name
        )

    fing_unc.tight_layout()
    plt.show()
    fing_unc.savefig(save_unc)

    fig_cor.tight_layout()
    plt.show()
    fig_cor.savefig(save_cor)

# Used in analysis_hyperparameters.py
def best_qd_score(csv_list, top_k, start_training_list, nb_epochs_list, per_data_list, time_list):
    """
    Retrieve the list of hyperparameters and results from the top k best simulations of the GridSearch.

    Args:
        - csv_list (list): List of the csv where results are stored 
        - top_k (int): Number of best simuations we want to keep from the GridSearch
        - start_training_list (list): List of the First Training hyperparameter
        - nb_epochs_list (list): List of the Number of Epochs hyperparameter
        - per_data_list (list): List of the % of Old Data retained between two trainings hyperparameter
        - time_list (list): List of the time of the different simulations

    Returns:
        - best_indices (list): List of the indices of the top k best simulations from the GridSearch
        - best_lists (list): List of the csv where results are stored for the top k best simulations 
        - start_training_list (list): List of the First Training hyperparameter for the top k best simulations
        - nb_epochs_list (list): List of the Number of Epochs hyperparameter for the top k best simulations
        - per_data_list (list): List of the % of Old Data retained between two trainings hyperparameter for the top k best simulations
        - time_list (list): List of the time of the top k best simulations
    """

    last_value_list_unc = []
    for nb_sim in range(len(csv_list)):
        last_value_list_unc.append(csv_list[nb_sim]['qd_score'].iloc[-1])
        best_indices = np.argsort(last_value_list_unc)[-top_k:]

    best_lists = [csv_list[idx] for idx in best_indices]
    start_training_list = [start_training_list[idx] for idx in best_indices]
    nb_epochs_list = [nb_epochs_list[idx] for idx in best_indices]
    per_data_list = [per_data_list[idx] for idx in best_indices]
    time_list = [time_list[idx] for idx in best_indices]

    return best_indices, best_lists, start_training_list, nb_epochs_list, per_data_list, time_list

# Used in NN_analysis.py
def plot_NN_loss(loss_train, loss_val, method, name, path, bool_mse, num_epochs):
    """
    Plot the training and testing loss.

    Args:
        - loss_train (list): List containing the average loss values during training
        - loss_val (list): List containing the average loss values during testing
        - method (str): Name of the algorithm
        - name (str): Name of the task
        - path (str): Path and name of the figure to be saved
        - bool_mse (boolean): Checking if dealing with the training loss (regularized: Loss + L2 Regularization term) or the loss used by the model (only the Loss)
        - num_epochs (int): Number of epochs used to train the model

    Returns:
        - Nothing but saves the corresponding plot
    """

    fig = plt.figure(figsize=(10,8))
    plt.xlabel('Epoch', fontsize=12, fontweight='bold') 

    if bool_mse == True:     
        # plt.plot(range(len(loss_train)), loss_train, label='MSE Training Loss')
        plt.plot(range(len(loss_train)), loss_train, label='NLL Training Loss')
        # plt.scatter(jnp.arange(num_epochs-1, len(loss_train), num_epochs), loss_val, label='MSE Validation Loss', c='orange')
        plt.scatter(jnp.arange(num_epochs-1, len(loss_train), num_epochs), loss_val, label='NLL Testing Loss', c='orange')
        # plt.ylabel('MSE Loss', fontsize=12, fontweight='bold')
        plt.ylabel('NLL Loss', fontsize=12, fontweight='bold')
        # plt.title(f"Evolution of the average MSE losses of the model with {method} \n for the {name} with {len(loss_train)} epochs to train", fontsize=14, fontweight='bold')
        plt.title(f"Evolution of the average NLL losses of the model with \n {method} for the {name} with {len(loss_train)} epochs to train", fontsize=14, fontweight='bold')
    else:
        plt.plot(range(len(loss_train)), loss_train, label='Regularized Training Loss')
        plt.ylabel('Regularized Loss', fontsize=12, fontweight='bold')
        # plt.title(f"Evolution of the average MSE loss of the model with {method} \n for the {name} with {len(loss_train)} epochs to train", fontsize=14, fontweight='bold')
        plt.title(f"Evolution of the average NLL loss of the model with {method} \n for the {name} with {len(loss_train)} epochs to train", fontsize=14, fontweight='bold')

    plt.legend()
    plt.show()
    plt.savefig(path)

# Used in NN_analysis.py
def plot_BD_figures(BD_function, BD_model, name, algo_name, path_comparison, path_difference):
    """
    Plot the BD comparison and differences between the predictions of the model and the evaluations of the scoring function.

    Args:
        - BD_function (list): List containing the BD evaluations from the scoring function
        - BD_model (list): List containing the BD predictions from the model
        - name (str): Name of the task
        - algo_name (str): Name of the algorithm
        - path_comparison (str): Path and name of the comparison figure to be saved
        - path_difference (str): Path and name of the difference figure to be saved

    Returns:
        - Nothing but saves the corresponding plots
    """

    marker_size = 7

    BD_function_x = BD_function[:, 0]
    BD_function_y = BD_function[:, 1]
    BD_model_x = BD_model[:, 0]
    BD_model_y = BD_model[:, 1]

    fig = plt.figure(figsize=(10,10))
    plt.scatter(BD_function_x, BD_function_y, label='Function', s=marker_size)
    plt.scatter(BD_model_x, BD_model_y, label='Model', s=marker_size)

    # for i in range(0, len(BD_function_x), len(BD_function_x)//30):
    # # for i in range(0, len(BD_function_x), 5):
    #     plt.plot([BD_function_x[i], BD_model_x[i]], [BD_function_y[i], BD_model_y[i]], color='gray', linestyle='--')

    plt.xlabel('Behaviour Dimension 1: X', fontsize=12, fontweight='bold')
    plt.ylabel('Behaviour Dimension 2: Y', fontsize=12, fontweight='bold')
    plt.title(f'Comparison of the BDs with {algo_name} \n between the scoring function and the model for the {name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(path_comparison)

    diff_x = jnp.abs(BD_model_x - BD_function_x)
    diff_y = jnp.abs(BD_model_y - BD_function_y)

    new_fig = plt.figure(figsize=(10,10))
    plt.scatter(diff_x, diff_y, s=marker_size)
    # plt.axhline(0, color='black', linestyle='--')  # Horizontal line at y=0
    # plt.axvline(0, color='black', linestyle='--')  # Vertical line at x=0
    plt.xlabel('Behaviour Dimension 1: X (Absolute Difference)', fontsize=12, fontweight='bold')
    plt.ylabel('Behaviour Dimension 2: Y (Absolute Difference)', fontsize=12, fontweight='bold')
    plt.title(f'Absolute difference of the BDs with {algo_name} \n between the scoring function and the model for the {name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path_difference)
    
# Used in NN_analysis.py
def plot_fitness_figures(fitness_function, fitness_model, name, path):
    """
    Plot the BD comparison and differences between the predictions of the model and the evaluations of the scoring function.

    Args:
        - fitness_function (list): List containing the fitness evaluations from the scoring function
        - fitness_model (list): List containing the fitness predictions from the model
        - name (str): Name of the task
        - path (str): Path and name of the figure to be saved

    Returns:
        - Nothing but saves the corresponding plot
    """

    diff_fitness = jnp.abs(fitness_function - fitness_model)
    fig = plt.figure(figsize=(10,10))
    # plt.plot(diff_fitness)
    plt.scatter(list(range(len(diff_fitness))), diff_fitness, s=7)
    plt.xlabel('Number of the prediction (Index)', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness Value (Difference)', fontsize=12, fontweight='bold')
    plt.title(f'Difference of the fitnesses between the scoring function \n and the model for the {name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path)

# Used in NN_analysis.py
def plot_viobox_figures(fitness_func, fitness_mod, BD_function, BD_model, path_fitness_vio, path_fitness_box, path_BD_vio, path_BD_box, name, algo_name):
    """
    Plot the violin and box plots of the fitnesses and BD errors between the scoring function and the model.

    Args:
        - fitness_func (list): List containing the fitness evaluations from the scoring function
        - fitness_mod (list): List containing the fitness predictions from the model
        - BD_function (list): List containing the BD evaluations from the scoring function
        - BD_model (list): List containing the BD predictions from the model
        - path_fitness_vio (str): Path and name of the figure to be saved
        - path_fitness_box (str): Path and name of the figure to be saved
        - path_BD_vio (str): Path and name of the figure to be saved
        - path_BD_box (str): Path and name of the figure to be saved
        - name (str): Name of the task
        - algo_name (str): Name of the algorithm

    Returns:
        - Nothing but saves the corresponding plots
    """

    fitness_error = np.abs(fitness_func - fitness_mod)
    BD_error = np.linalg.norm(BD_function - BD_model, axis=1)

    df_fitness_error = pd.DataFrame(fitness_error)
    df_BD_error = pd.DataFrame(BD_error)

    fig = plt.figure(figsize=(10, 10))
    sns.violinplot(data=df_fitness_error, inner='box', palette='pastel')
    plt.xlabel('Fitness', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness Error (Absolute Difference)', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of the fitness error for the {name} with {algo_name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path_fitness_vio)

    fig_2 = plt.figure(figsize=(10, 10))
    sns.boxplot(data=df_fitness_error, palette='pastel')
    plt.xlabel('Fitness', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness Error (Absolute Difference)', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of the fitness error for the {name} with {algo_name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path_fitness_box)

    new_fig = plt.figure(figsize=(10, 10))
    sns.violinplot(data=df_BD_error, inner='box', palette='pastel')
    plt.xlabel('Behavioural Descriptor', fontsize=12, fontweight='bold')
    plt.ylabel('BD Error (Euclidian Distance)', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of the BD error for the {name} with {algo_name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path_BD_vio)

    new_fig_2 = plt.figure(figsize=(10, 10))
    sns.boxplot(data=df_BD_error, palette='pastel')
    plt.xlabel('Behavioural Descriptor', fontsize=12, fontweight='bold')
    plt.ylabel('BD Error (Euclidian Distance)', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of the BD error for the {name} with {algo_name}', fontsize=14, fontweight='bold')
    plt.show()
    plt.savefig(path_BD_box)

# Used in NN_analysis.py
def remove_outliers(BD_function, BD_model, fitness_func, fitness_mod, name):
    """
    Plot the violin and box plots of the fitnesses and BD errors between the scoring function and the model.

    Args:
        - BD_function (list): List containing the BD evaluations from the scoring function
        - BD_model (list): List containing the BD predictions from the model
        - fitness_func (list): List containing the fitness evaluations from the scoring function
        - fitness_mod (list): List containing the fitness predictions from the model
        - name (str): Name of the task

    Returns:
        - BD_function_no_outliers (list): List containing the BD evaluations without outliers from the scoring function
        - BD_model_no_outliers (list): List containing the BD predictions without outliers from the model
        - fitness_func_no_outliers (list): List containing the fitness evaluations without outliers from the scoring function
        - fitness_mod_no_outliers (list): List containing the fitness predictions without outliers from the model
    """

    fitness_error = np.abs(fitness_func - fitness_mod)
    BD_error = np.linalg.norm(BD_function - BD_model, axis=1)

    df_fitness_error = pd.DataFrame(fitness_error)
    df_BD_error = pd.DataFrame(BD_error)

    median_fitness = df_fitness_error.median().values[0]
    print(f'\n{name}: For the fitness, the median is {median_fitness:.5f}')
    median_BD = df_BD_error.median().values[0]
    print(f'{name}: For the BD, the median is {median_BD:.5f}')

    # Calculate number of outliers
    threshold_fitness = median_fitness + 1.5 * df_fitness_error.std()
    outliers_above_fitness = df_fitness_error[df_fitness_error > threshold_fitness].count().sum()
    outliers_below_fitness = df_fitness_error[df_fitness_error < -threshold_fitness].count().sum()
    total_outliers_fitness = outliers_above_fitness + outliers_below_fitness
    print(f'{name}: For the fitness, there are {total_outliers_fitness} outliers out of {df_fitness_error.size} elements so {total_outliers_fitness/df_fitness_error.size*100:.2f}%')

    threshold_BD = median_BD + 1.5 * df_BD_error.std()
    outliers_above_BD = df_BD_error[df_BD_error > threshold_BD].count().sum()
    outliers_below_BD = df_BD_error[df_BD_error < -threshold_BD].count().sum()
    total_outliers_BD = outliers_above_BD + outliers_below_BD
    print(f'{name}: For the BD, there are {total_outliers_BD} outliers out of {df_BD_error.size} elements so {total_outliers_BD/df_BD_error.size*100:.2f}%\n')

    outliers_fitness_indices = np.where((df_fitness_error > threshold_fitness) | (df_fitness_error < -threshold_fitness))[0]
    outliers_BD_indices = np.where((df_BD_error > threshold_BD) | (df_BD_error < -threshold_BD))[0]

    BD_function_no_outliers = np.delete(BD_function, outliers_BD_indices, axis=0)
    BD_model_no_outliers = np.delete(BD_model, outliers_BD_indices, axis=0)
    fitness_func_no_outliers = np.delete(fitness_func, outliers_fitness_indices)
    fitness_mod_no_outliers = np.delete(fitness_mod, outliers_fitness_indices)

    return BD_function_no_outliers, BD_model_no_outliers, fitness_func_no_outliers, fitness_mod_no_outliers

# Used in analysis_replication.py
def process_replication(methods, dict_values, metrics_columns, rep_avg):
    """
    Create a new csv for each algorithm wrt the task (arm or noisy_arm) and the metrics (corrected or uncorrected).

    Args:
        - methods (list): List containing the name of all the algorithms
        - dict_values (dictionary): Dict containing the values to preprocess first row value of csvs
        - metrics_columns (list): List containing the title of the columns that we want to compute the mean/median and std
        - rep_avg (boolean): Checking if we want to take the mean (TRUE) or the median (FALSE) of the 5 replications

    Returns:
        - Nothing but saves the corresponding replicated csv, with mean/median and std
    """

    for method in methods:

        path_a_unc = f'results_replications/methods/{method}/arm/unc/'
        path_a_cor = f'results_replications/methods/{method}/arm/cor/'
        path_na_unc = f'results_replications/methods/{method}/noisy_arm/unc/'
        path_na_cor = f'results_replications/methods/{method}/noisy_arm/cor/'
        
        all_paths = [path_a_unc, path_a_cor, path_na_unc, path_na_cor]

        for i, path in enumerate(all_paths):

            if i ==0:
                name = 'unc_arm'
                direct_path = 'arm/unc'
                save_path = 'results_replications/replicated_results/arm/unc'
            elif i==1:
                name = 'cor_arm'
                direct_path = 'arm/cor'
                save_path = 'results_replications/replicated_results/arm/cor'
            elif i==2:
                name = 'unc_noisy_arm'
                direct_path = 'noisy_arm/unc'
                save_path = 'results_replications/replicated_results/noisy_arm/unc'
            elif i==3:
                name = 'cor_noisy_arm'
                direct_path = 'noisy_arm/cor'
                save_path = 'results_replications/replicated_results/noisy_arm/cor'

            os.makedirs(save_path, exist_ok=True)

            all_files = os.listdir(path)

            # Retrieve the 5 replications of the method given a specific path
            csv_files = [file for file in all_files if file.endswith('.csv')]

            # Load CSV into DF
            csv1 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[0]}')
            csv2 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[1]}')
            csv3 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[2]}')
            csv4 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[3]}')
            csv5 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[4]}')

            # Preprocess first row of each csv
            for column, value in dict_values.items():
                csv1.iloc[0, csv1.columns.get_loc(column)] = value
                csv2.iloc[0, csv2.columns.get_loc(column)] = value
                csv3.iloc[0, csv3.columns.get_loc(column)] = value
                csv4.iloc[0, csv4.columns.get_loc(column)] = value
                csv5.iloc[0, csv5.columns.get_loc(column)] = value

            # Initialize a new dataframe to store the results
            replicate_df = pd.DataFrame()

            # Loop over the row of the DFs
            for row in range(len(csv1)):

                # Retrieve the values of each column at a given row for all the DFs
                values = [csv1.loc[row, metrics_columns],
                        csv2.loc[row, metrics_columns],
                        csv3.loc[row, metrics_columns],
                        csv4.loc[row, metrics_columns],
                        csv5.loc[row, metrics_columns]]

                # Calculate mean and std for each column at a given row
                if rep_avg == True:
                    mean_values = np.mean(values, axis=0)
                else:
                    mean_values = np.median(values, axis=0)
                std_values = np.std(values, axis=0)

                # Create a dictionary to store the results at a given row
                row_result = {'loop': int(csv1.loc[row, 'loop']),
                            'iteration': int(csv1.loc[row, 'iteration'])}
                
                # Add mean and std to the dictionary at a given row for each column
                for i, column in enumerate(metrics_columns):

                    row_result[column] = mean_values[i]
                    row_result[f'std_{column}'] = std_values[i]

                # Store the new row into the DF
                # replicate_df = replicate_df.append(row_result, ignore_index=True)
                replicate_df = pd.concat([replicate_df, pd.DataFrame([row_result])], ignore_index=True)


            # Save the result dataframe to a new CSV file
            replicate_df.to_csv(f'{save_path}/{method}_{name}_replicated.csv', index=False)

# Used in analysis_replication.py
def plot_replicated(saving_path, all_columns, metrics_columns, log_period, sampling_size, methods, check_full, check_reset):
    """
    Create the metrics plot for each task (arm and noisy_arm) and each metric (corrected and uncorrected), allowing a comparison between all algorithms.

    Args:
        - saving_path (str): Path and name of the plot to be saved
        - all_columns (list): List containing the title of all the columns of the new DF so with the mean/median and std
        - metrics_columns (list): List containing the title of the columns that we want to compute the mean/median and std
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms
        - methods (list): List containing the name of all the algorithms
        - check_full (boolean): Checking if we plot 6 or 10 replications. True = 10
        - check_reset (boolean): Checking if we compare Wipe or Not Wipe results when using only 6 methods

    Returns:
        - Nothing but saves the corresponding plot, with mean/median and std
    """
    os.makedirs(saving_path, exist_ok=True)

    for i in range(4):

        if i==0:
            direct_path = 'arm/unc'
            sample = 'Uncorrected'
            task_name = 'Deterministic Arm'
            full_saving_path = f'{saving_path}/Replicated_Deterministic_Uncorrected_Metrics.png'
        elif i==1:
            direct_path = 'arm/cor'
            sample = 'Corrected'
            task_name = 'Deterministic Arm'
            full_saving_path = f'{saving_path}/Replicated_Deterministic_Corrected_Metrics.png'
        elif i==2:
            direct_path = 'noisy_arm/unc'
            sample = 'Uncorrected'
            task_name = 'Uncertain Arm'
            full_saving_path = f'{saving_path}/Replicated_Uncertain_Uncorrected_Metrics.png'
        elif i==3:
            direct_path = 'noisy_arm/cor'
            sample = 'Corrected'
            task_name = 'Uncertain Arm'
            full_saving_path = f'{saving_path}/Replicated_Uncertain_Corrected_Metrics.png'

        full_path = f'results_replications/replicated_results/{direct_path}'
        all_files = os.listdir(f'{full_path}')

        # Retrieve the 6 different replications (one by algorithm) given a specific path (task and metric)
        csv_files = [file for file in all_files if file.endswith('.csv')]
        csv_files.sort(reverse=True)
        # print(csv_files)

        if check_full == True:

            # Load CSV into DF
            csv1 = pd.read_csv(f'{full_path}/{csv_files[0]}') # ME
            csv2 = pd.read_csv(f'{full_path}/{csv_files[1]}') # MES
            csv3 = pd.read_csv(f'{full_path}/{csv_files[2]}') # MEMB Imp
            csv4 = pd.read_csv(f'{full_path}/{csv_files[3]}') # MEMB Wipe Imp
            csv5 = pd.read_csv(f'{full_path}/{csv_files[4]}') # MEMB Exp
            csv6 = pd.read_csv(f'{full_path}/{csv_files[5]}') # MEMB Wipe Exp
            csv7 = pd.read_csv(f'{full_path}/{csv_files[6]}') # MEMBUQ Exp
            csv8 = pd.read_csv(f'{full_path}/{csv_files[7]}') # MEMBUQ Wipe Exp
            csv9 = pd.read_csv(f'{full_path}/{csv_files[8]}') # MEMBUQ Imp
            csv10 = pd.read_csv(f'{full_path}/{csv_files[9]}') # MEMBUQ Wipe Imp

            data1 = csv1[all_columns]
            data2 = csv2[all_columns]
            data3 = csv3[all_columns]
            data4 = csv4[all_columns]
            data5 = csv5[all_columns]
            data6 = csv6[all_columns]
            data7 = csv7[all_columns]
            data8 = csv8[all_columns]
            data9 = csv9[all_columns]
            data10 = csv10[all_columns]

            all_data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]

            fig_width = 24
            fig_height = 9

        elif check_full == False:

            if check_reset == False:

                # Load CSV into DF
                csv1 = pd.read_csv(f'{full_path}/{csv_files[0]}')
                csv2 = pd.read_csv(f'{full_path}/{csv_files[1]}')
                csv3 = pd.read_csv(f'{full_path}/{csv_files[2]}')
                csv4 = pd.read_csv(f'{full_path}/{csv_files[4]}')
                csv5 = pd.read_csv(f'{full_path}/{csv_files[6]}')
                csv6 = pd.read_csv(f'{full_path}/{csv_files[8]}')

                data1 = csv1[all_columns]
                data2 = csv2[all_columns]
                data3 = csv3[all_columns]
                data4 = csv4[all_columns]
                data5 = csv5[all_columns]
                data6 = csv6[all_columns]

                all_data = [data1, data2, data3, data4, data5, data6]

            elif check_reset == True:

                # Load CSV into DF
                csv1 = pd.read_csv(f'{full_path}/{csv_files[0]}')
                csv2 = pd.read_csv(f'{full_path}/{csv_files[1]}')
                csv3 = pd.read_csv(f'{full_path}/{csv_files[3]}')
                csv4 = pd.read_csv(f'{full_path}/{csv_files[5]}')
                csv5 = pd.read_csv(f'{full_path}/{csv_files[7]}')
                csv6 = pd.read_csv(f'{full_path}/{csv_files[9]}')

                data1 = csv1[all_columns]
                data2 = csv2[all_columns]
                data3 = csv3[all_columns]
                data4 = csv4[all_columns]
                data5 = csv5[all_columns]
                data6 = csv6[all_columns]

                all_data = [data1, data2, data3, data4, data5, data6]

                # # Load CSV into DF
                # csv1 = pd.read_csv(f'{full_path}/{csv_files[0]}')
                # csv2 = pd.read_csv(f'{full_path}/{csv_files[1]}')
                # csv3 = pd.read_csv(f'{full_path}/{csv_files[7]}')

                # data1 = csv1[all_columns]
                # data2 = csv2[all_columns]
                # data3 = csv3[all_columns]

                # all_data = [data1, data2, data3]

            fig_width = 20
            fig_height = 7.5
                   
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height))

        # Plot each metric with std deviation
        for j, metric in enumerate(metrics_columns):

            ax = axes[j]

            if j == 0:
                title = 'QD Score'
                ylabel = title
            elif j == 1:
                title = 'Max Fitness'
                ylabel = title
            elif j == 2:
                title = 'Coverage'
                ylabel = 'Coverage in %'

            for k in range(len(methods)):
                ax.plot(np.arange(len(all_data[k]))*log_period*sampling_size, all_data[k][metric], label=methods[k])
                ax.fill_between(np.arange(len(all_data[k]))*log_period*sampling_size, all_data[k][metric] - all_data[k][f'std_{metric}'], all_data[k][metric] + all_data[k][f'std_{metric}'], alpha=0.55)
            
            ax.set_xlabel("Environment Steps", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{ylabel}", fontsize=12, fontweight='bold')
            ax.set_title(f"Evolution of the {sample} {title} \n {task_name}", fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{full_saving_path}')

# Used in analysis_replication.py
def calculate_loss_metrics(methods, metrics_columns):
    """
    Create a new csv containing loss metrics for each algorithm wrt the noisy_arm and the metrics.

    Args:
        - methods (list): List containing the name of all the algorithms
        - metrics_columns (list): List containing the title of the columns that we want to compute the mean/median and std

    Returns:
        - Nothing but saves the corresponding loss metrics csv
    """
    
    # save_path_a = f'results_replications/loss_metrics/arm'
    save_path_na = f'results_replications/loss_metrics/noisy_arm'
    # os.makedirs(save_path_a, exist_ok=True)
    os.makedirs(save_path_na, exist_ok=True)

    for method in methods:
        
        # Path to access median csv of the 5 replications
        # path_a_unc = f'results_replications/replicated_results/arm/unc/'
        # path_a_cor = f'results_replications/replicated_results/arm/cor/'
        path_na_unc = f'results_replications/replicated_results/noisy_arm/unc/'
        path_na_cor = f'results_replications/replicated_results/noisy_arm/cor/'
        
        # CSV path for each method
        # method_path_a_unc = f"{path_a_unc}{method}_unc_arm_replicated.csv"
        # method_path_a_cor = f"{path_a_cor}{method}_cor_arm_replicated.csv"
        method_path_na_unc = f"{path_na_unc}{method}_unc_noisy_arm_replicated.csv"
        method_path_na_cor = f"{path_na_cor}{method}_cor_noisy_arm_replicated.csv"

        # Load CSV into DF
        # csv_a_unc = pd.read_csv(f'{method_path_a_unc}')
        # csv_a_cor = pd.read_csv(f'{method_path_a_cor}')
        csv_na_unc = pd.read_csv(f'{method_path_na_unc}')
        csv_na_cor = pd.read_csv(f'{method_path_na_cor}')

        # Initialize a new dataframe to store the results
        # loss_df_a = pd.DataFrame()
        loss_df_na = pd.DataFrame()

        # Loop over the row of the DFs
        for row in range(1, len(csv_na_unc)):

            # Retrieve the values of each column at a given row for all the DFs
            values = [
                # csv_a_unc.loc[row, metrics_columns],
                # csv_a_cor.loc[row, metrics_columns],
                csv_na_unc.loc[row, metrics_columns],
                csv_na_cor.loc[row, metrics_columns],
            ]

            # Initialize lists to store the loss values for each metric column
            loss_arm = []
            loss_noisy_arm = []

            # Calculate the loss metric for each metric column
            for i, column in enumerate(metrics_columns):
                # loss_arm.append((values[0][i] - values[1][i]) / values[0][i])
                loss_noisy_arm.append((values[0][i] - values[1][i]) / values[0][i])

            # Create a dictionary to store the results at a given row
            # row_result_a = {'loop': int(csv_a_unc.loc[row, 'loop']),
            #                 'iteration': int(csv_a_unc.loc[row, 'iteration'])}

            # Create a dictionary to store the results at a given row
            row_result_na = {'loop': int(csv_na_unc.loc[row, 'loop']),
                            'iteration': int(csv_na_unc.loc[row, 'iteration'])}

            # Add mean and std to the dictionary at a given row for each column
            for i, column in enumerate(metrics_columns):
                # row_result_a[f'Loss_{column}'] = loss_arm[i]
                row_result_na[f'Loss_{column}'] = loss_noisy_arm[i]

            # Store the new row into the DF
            # loss_df_a = pd.concat([loss_df_a, pd.DataFrame([row_result_a])], ignore_index=True)
            loss_df_na = pd.concat([loss_df_na, pd.DataFrame([row_result_na])], ignore_index=True)

        # Save the result dataframe to a new CSV file
        # loss_df_a.to_csv(f'{save_path_a}/{method}_arm_loss_metrics.csv', index=False)
        loss_df_na.to_csv(f'{save_path_na}/{method}_noisy_arm_loss_metrics.csv', index=False)

# Used in analysis_replication.py
def plot_loss_metrics(methods, saving_path):
    """
    Plot the loss metrics for the QD score, coverage and max fitness.

    Args:
        - saving_path (str): Path and name of the plot to be saved
        - methods (list): List containing the name of all the algorithms

    Returns:
        - Nothing but the corresponding plot
    """

    os.makedirs(saving_path, exist_ok=True)

    path = f'results_replications/loss_metrics/noisy_arm'
    all_dfs = []

    for method in methods:
        full_path = f'{path}/{method}_noisy_arm_loss_metrics.csv'
        all_dfs.append(pd.read_csv(full_path))

    # Multiply the loss columns by 100 to convert to percentages
    for df in all_dfs:
        df["Loss_qd_score"] *= 100
        # df["Loss_max_fitness"] *= 100
        df["Loss_coverage"] *= 100

    colors = sns.color_palette("Set3", n_colors=len(methods))

    # Create separate figures for each Loss column
    for i, loss_column in enumerate(["Loss_qd_score", "Loss_max_fitness", "Loss_coverage"]):

        if i == 0:
            title = 'Loss QD score'
        elif i== 1:
            title = 'Loss max fitness'
        elif i == 2:
            title = 'Loss coverage'
        
        data_to_plot = [df[loss_column] for df in all_dfs]

        plt.figure(figsize=(10, 8))
        
        box = plt.boxplot(data_to_plot, labels=methods, showfliers=False, patch_artist=True)
        for box_element, color in zip(box['boxes'], colors):
            box_element.set_facecolor(color)

        for median in box['medians']:
            median.set(color='black', linewidth=1.2)

        plt.title(f'{title} violin plots for the Uncertain Arm', fontsize=14, fontweight='bold')     
        plt.xlabel('Algorithms', fontsize=12, fontweight='bold')
        if loss_column == "Loss_max_fitness":
            plt.ylabel(f'{title}', fontsize=12, fontweight='bold')
        else:
            plt.ylabel(f'{title} (%)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=55)

        plt.tight_layout()
        plt.savefig(f'{saving_path}/{loss_column}.png')  
        plt.close()  

# Used in analysis_replication.py
def process_replication_stds(methods, metrics_columns, rep_avg):
    """
    Create a new csv for each algorithm wrt the task (arm or noisy_arm) and the metrics (corrected or uncorrected). Only for std_fit and std_bd metrics.

    Args:
        - methods (list): List containing the name of all the algorithms
        - metrics_columns (list): List containing the title of the columns that we want to compute the mean/median and std
        - rep_avg (boolean): Checking if we want to take the mean (TRUE) or the median (FALSE) of the 5 replications

    Returns:
        - Nothing but saves the corresponding replicated csv, with mean/median and std
    """

    for method in methods:

        path_a_unc = f'results_replications/methods/{method}/arm/unc/'
        path_a_cor = f'results_replications/methods/{method}/arm/cor/'
        path_na_unc = f'results_replications/methods/{method}/noisy_arm/unc/'
        path_na_cor = f'results_replications/methods/{method}/noisy_arm/cor/'
        
        all_paths = [path_a_unc, path_a_cor, path_na_unc, path_na_cor]

        for i, path in enumerate(all_paths):

            if i ==0:
                name = 'unc_arm'
                direct_path = 'arm/unc'
                save_path = 'results_replications/replicated_results_stds/arm/unc'
            elif i==1:
                name = 'cor_arm'
                direct_path = 'arm/cor'
                save_path = 'results_replications/replicated_results_stds/arm/cor'
            elif i==2:
                name = 'unc_noisy_arm'
                direct_path = 'noisy_arm/unc'
                save_path = 'results_replications/replicated_results_stds/noisy_arm/unc'
            elif i==3:
                name = 'cor_noisy_arm'
                direct_path = 'noisy_arm/cor'
                save_path = 'results_replications/replicated_results_stds/noisy_arm/cor'

            os.makedirs(save_path, exist_ok=True)

            all_files = os.listdir(path)

            # Retrieve the 5 replications of the method given a specific path
            csv_files = [file for file in all_files if file.endswith('.csv')]

            # Load CSV into DF
            csv1 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[0]}')
            csv2 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[1]}')
            csv3 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[2]}')
            csv4 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[3]}')
            csv5 = pd.read_csv(f'results_replications/methods/{method}/{direct_path}/{csv_files[4]}')

            # Initialize a new dataframe to store the results
            replicate_df = pd.DataFrame()

            # Loop over the row of the DFs
            for row in range(len(csv1)):

                # Retrieve the values of each column at a given row for all the DFs
                values = [csv1.loc[row, metrics_columns],
                        csv2.loc[row, metrics_columns],
                        csv3.loc[row, metrics_columns],
                        csv4.loc[row, metrics_columns],
                        csv5.loc[row, metrics_columns]]

                # Calculate mean and std for each column at a given row
                if rep_avg == True:
                    mean_values = np.mean(values, axis=0)
                else:
                    mean_values = np.median(values, axis=0)
                std_values = np.std(values, axis=0)

                # Create a dictionary to store the results at a given row
                row_result = {'loop': int(csv1.loc[row, 'loop']),
                            'iteration': int(csv1.loc[row, 'iteration'])}
                
                # Add mean and std to the dictionary at a given row for each column
                for i, column in enumerate(metrics_columns):

                    row_result[column] = mean_values[i]
                    row_result[f'std_{column}'] = std_values[i]

                # Store the new row into the DF
                # replicate_df = replicate_df.append(row_result, ignore_index=True)
                replicate_df = pd.concat([replicate_df, pd.DataFrame([row_result])], ignore_index=True)


            # Save the result dataframe to a new CSV file
            replicate_df.to_csv(f'{save_path}/{method}_{name}_replicated.csv', index=False)

# Used in analysis_replication.py
def plot_replicated_stds(saving_path, all_columns, metrics_columns, log_period, sampling_size, methods):
    """
    Create the metrics plot for each task (arm and noisy_arm) and each metric (corrected and uncorrected), allowing a comparison between all algorithms. Only for std_fit and std_bd.

    Args:
        - saving_path (str): Path and name of the plot to be saved
        - all_columns (list): List containing the title of all the columns of the new DF so with the mean/median and std
        - metrics_columns (list): List containing the title of the columns that we want to compute the mean/median and std
        - log_period (int): Generation frequency to collect new metrics while running main algorithms
        - sampling_size (int): Resources allocation used to compare different algorithms
        - methods (list): List containing the name of all the algorithms

    Returns:
        - Nothing but saves the corresponding plot, with mean/median and std
    """
    os.makedirs(saving_path, exist_ok=True)

    for i in range(4):

        if i==0:
            direct_path = 'arm/unc'
            sample = 'Uncorrected'
            task_name = 'Deterministic Arm'
            full_saving_path = f'{saving_path}/Stds_Replicated_Deterministic_Uncorrected_Metrics.png'
        elif i==1:
            direct_path = 'arm/cor'
            sample = 'Corrected'
            task_name = 'Deterministic Arm'
            full_saving_path = f'{saving_path}/Stds_Replicated_Deterministic_Corrected_Metrics.png'
        elif i==2:
            direct_path = 'noisy_arm/unc'
            sample = 'Uncorrected'
            task_name = 'Uncertain Arm'
            full_saving_path = f'{saving_path}/Stds_Replicated_Uncertain_Uncorrected_Metrics.png'
        elif i==3:
            direct_path = 'noisy_arm/cor'
            sample = 'Corrected'
            task_name = 'Uncertain Arm'
            full_saving_path = f'{saving_path}/Stds_Replicated_Uncertain_Corrected_Metrics.png'

        full_path = f'results_replications/replicated_results_stds/{direct_path}'
        all_files = os.listdir(f'{full_path}')

        # Retrieve the 6 different replications (one by algorithm) given a specific path (task and metric)
        csv_files = [file for file in all_files if file.endswith('.csv')]
        csv_files.sort(reverse=True)
        # print(csv_files)

        
        # Load CSV into DF
        csv1 = pd.read_csv(f'{full_path}/{csv_files[0]}')
        csv2 = pd.read_csv(f'{full_path}/{csv_files[1]}')
        csv3 = pd.read_csv(f'{full_path}/{csv_files[2]}')
        csv4 = pd.read_csv(f'{full_path}/{csv_files[3]}')
        csv5 = pd.read_csv(f'{full_path}/{csv_files[4]}')

        data1 = csv1[all_columns]
        data2 = csv2[all_columns]
        data3 = csv3[all_columns]
        data4 = csv4[all_columns]
        data5 = csv5[all_columns]

        all_data = [data1, data2, data3, data4, data5]

        fig_width = 12
        fig_height = 6
                
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))

        # Plot each metric with std deviation
        for j, metric in enumerate(metrics_columns):

            ax = axes[j]

            if j == 0:
                title = 'Std Fitness'
                ylabel = title
            elif j == 1:
                title = 'Std BD'
                ylabel = title

            for k in range(len(methods)):
                ax.plot(np.arange(len(all_data[k]))*log_period*sampling_size, all_data[k][metric], label=methods[k])
                ax.fill_between(np.arange(len(all_data[k]))*log_period*sampling_size, all_data[k][metric] - all_data[k][f'std_{metric}'], all_data[k][metric] + all_data[k][f'std_{metric}'], alpha=0.55)
            
            ax.set_xlabel("Environment Steps", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{ylabel}", fontsize=12, fontweight='bold')
            ax.set_title(f"Evolution of the {sample} {title} \n {task_name}", fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{full_saving_path}')

