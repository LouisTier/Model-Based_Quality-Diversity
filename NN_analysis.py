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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from annexed_methods import plot_NN_loss, plot_BD_figures, plot_fitness_figures, plot_viobox_figures, create_directory, remove_outliers, moving_avg, plot_errors, plot_errors_std, plot_comparison, replace_NAN

# MEMB_Explicit_Naive, MEMB_Explicit_Naive_Wipe, MEMB_Implicit, MEMB_Implicit_Wipe, MEMBUQ_Implicit, MEMBUQ_Implicit_Wipe, MEMBUQ_NLL_Explicit_Naive, MEMBUQ_NLL_Explicit_Naive_Wipe
# MEMBUQ2_NLL_Explicit_Naive (Standardization), MEMBUQ_NLL_Explicit_Naive_Both_Wipe (Both SF)
method = "MEMBUQ_NLL_Explicit_Naive_Both_Wipe" 
if method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":
    task_name = ['both']
else:
    task_name = ['arm','noisy_arm'] # ['arm', 'noisy_arm'] 

# current_path = os.path.dirname(os.path.realpath(__file__))
# result_path = os.path.join(current_path, "results_sim_arg")
# method_path = os.path.join(result_path, method)

# loss_path = os.path.join(method_path, "Losses")
# arm_loss_path = os.path.join(loss_path, "arm")
# noisy_arm_loss_path = os.path.join(loss_path, "noisy_arm")
# both_loss_path = os.path.join(loss_path, "both")

# error_path = os.path.join(method_path, "Errors")
# arm_error_path = os.path.join(error_path, "arm")
# noisy_arm_error_path = os.path.join(error_path, "noisy_arm")
# both_error_path = os.path.join(error_path, "both")

# error_std_path = os.path.join(method_path, "Errors_Std")
# arm_error_std_path = os.path.join(error_std_path, "arm")
# noisy_arm_error_std_path = os.path.join(error_std_path, "noisy_arm")
# both_error_std_path = os.path.join(error_std_path, "both")

# comparison_path = os.path.join(method_path, "Comparisons")
# arm_comparison_path = os.path.join(comparison_path, "arm")
# noisy_arm_comparison_path = os.path.join(comparison_path, "noisy_arm")
# both_comparison_path = os.path.join(comparison_path, "both")

# create_directory(result_path), create_directory(method_path)
# create_directory(loss_path), create_directory(arm_loss_path), create_directory(noisy_arm_loss_path)
# create_directory(error_path), create_directory(arm_error_path), create_directory(noisy_arm_error_path)
# create_directory(error_std_path), create_directory(arm_error_std_path), create_directory(noisy_arm_error_std_path) 
# create_directory(comparison_path), create_directory(arm_comparison_path), create_directory(noisy_arm_comparison_path) 

# create_directory(both_loss_path), create_directory(both_error_path), create_directory(both_error_std_path), create_directory(both_comparison_path)

direct_1 = ['BD_Comparison', 'Fitness_Comparison', 'Viobox_Comparison']
direct_2 = ['arm', 'noisy_arm']
# direct_2 = ['both']
direct_3 = ['In_Training', 'Out_Training']
direct_4 = ['No_Outliers', 'Outliers']

# for i in direct_1:
#     path_1 = os.path.join(method_path, i)
#     create_directory(path_1)
#     # print("path_1: ", path_1)
#     for j in direct_2:
#         path_2 = os.path.join(path_1, j)
#         create_directory(path_2)
#         # print("path_2: ", path_2)
#         for k in direct_3:
#             path_3 = os.path.join(path_2, k)
#             create_directory(path_3)
#             # print("path_3: ", path_3)
#             for l in direct_4:
#                 path_4 = os.path.join(path_3, l)
#                 create_directory(path_4)
#                 # print("path_4: ", path_4)


backup = f'results_sim_arg/{method}/Backup/{method}_'
saving_title = '' # In case I want to modify beginning of saving path results
l2_rate = 0.000001 
marker_size = 5 

for name in task_name:

    if method == "MEMB_Explicit_Naive":
        num_epochs = 125
        batch_size = 64 
        if name == "arm":
            time_format = "30-08-2023_14-39-43"  
        elif name =="noisy_arm":
            time_format = "30-08-2023_14-39-43" 

    elif method == "MEMB_Explicit_Naive_Wipe":
        num_epochs = 125
        batch_size = 64
        if name == "arm":
            time_format = "30-08-2023_12-29-43" 
        elif name =="noisy_arm":
            time_format = "30-08-2023_12-29-48" 

    elif method == "MEMB_Implicit":
        num_epochs = 125
        batch_size = 512
        if name == "arm":
            time_format = "30-08-2023_18-16-40" 
        elif name =="noisy_arm":
            time_format = "30-08-2023_18-16-42" 

    elif method == "MEMB_Implicit_Wipe":
        num_epochs = 125
        batch_size = 512
        if name == "arm":
            time_format = "30-08-2023_15-55-47" 
        elif name =="noisy_arm":
            time_format = "30-08-2023_15-55-46" 

    elif method == "MEMBUQ_Implicit":
        num_epochs = 225
        batch_size = 512
        if name == "arm":
            time_format = "30-08-2023_16-08-30"
        elif name =="noisy_arm":
            time_format = "30-08-2023_16-08-40"

    elif method == "MEMBUQ_Implicit_Wipe":
        num_epochs = 225
        batch_size = 512
        if name == "arm":
            time_format = "31-08-2023_10-29-35"
        elif name =="noisy_arm":
            time_format = "31-08-2023_10-29-28"

    elif method == "MEMBUQ_NLL_Explicit_Naive": 
        num_epochs = 175
        batch_size = 64 
        if name == "arm":
            time_format = "30-08-2023_14-49-50" 
        elif name =="noisy_arm":
            time_format = "30-08-2023_14-49-50" 

    elif method == "MEMBUQ_NLL_Explicit_Naive_Wipe": 
        num_epochs = 175
        batch_size = 64 
        if name == "arm":
            time_format = "31-08-2023_10-05-22" 
        elif name =="noisy_arm":
            time_format = "31-08-2023_10-29-31" 

    elif method == "MEMBUQ2_NLL_Explicit_Naive": 
        num_epochs = 175
        batch_size = 64 
        if name == "arm":
            time_format = "24-08-2023_16-49-18" 
        elif name =="noisy_arm":
            time_format = "24-08-2023_14-55-18" 

    elif method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe": 
        num_epochs = 175
        batch_size = 64 
        if name == "both":
            time_format = "31-08-2023_17-05-28"  
    
    print("\n",method, name, num_epochs, batch_size, time_format)

    #################
    #    OBJECTS    #
    #################

    training_loss = jnp.load(f'{backup}{name}_training_loss_{time_format}.npy') # Regularized: MSE/NGLL + L2
    val_loss = jnp.load(f'{backup}{name}_val_loss_{time_format}.npy')
    # print("training loss: ", training_loss)
    # print("val loss: ", val_loss)

    ### Predictions between trainings 
    if name == 'both':
        outside_training_model = jnp.load(f'{backup}{name}_OT_model_predictions_{time_format}.npy') 
        outside_training_function_arm = jnp.load(f'{backup}{name}_OT_function_predictions_arm_{time_format}.npy') 
        outside_training_function_noisy_arm = jnp.load(f'{backup}{name}_OT_function_predictions_noisy_arm_{time_format}.npy') 
        outside_training_function = outside_training_function_noisy_arm
    else:
        outside_training_model = jnp.load(f'{backup}{name}_OT_model_predictions_{time_format}.npy') 
        outside_training_function = jnp.load(f'{backup}{name}_OT_function_predictions_{time_format}.npy')

    if method != "MEMBUQ_Implicit" and method != "MEMBUQ_Implicit_Wipe" and method != "MEMBUQ_NLL_Explicit_Naive" and method != "MEMBUQ_NLL_Explicit_Naive_Wipe" and method != "MEMBUQ2_NLL_Explicit_Naive" and method != "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":

        ### Training & validations losses 
        train_loss = jnp.load(f'{backup}{name}_mse_loss_{time_format}.npy') # Not regularized: Only MSE

        ### Predictions of the last test set (no std)
        BD_function = jnp.load(f'{backup}{name}_BD_func_{time_format}.npy')
        BD_model = jnp.load(f'{backup}{name}_BD_mod_{time_format}.npy')
        fitness_function = jnp.ravel(jnp.load(f'{backup}{name}_fit_func_{time_format}.npy'))
        fitness_model = jnp.ravel(jnp.load(f'{backup}{name}_fit_mod_{time_format}.npy')) 

        path_loss = f'results_sim_arg/{method}/Losses/{name}/{saving_title}{name}_MSE_losses.png' 

        # Retrieve BD and fitness stds for both IT and OT when using a UQ algorithm
        if method == "MEMBUQ_Explicit_Naive":
            
            ### Predictions of the last test set (only std)
            BD_var_function = jnp.load(f'{backup}{name}_BD_var_func_{time_format}.npy')
            BD_var_model = jnp.load(f'{backup}{name}_BD_var_mod_{time_format}.npy')
            fitness_var_function = jnp.ravel(jnp.load(f'{backup}{name}_fit_var_func_{time_format}.npy'))
            fitness_var_model = jnp.ravel(jnp.load(f'{backup}{name}_fit_var_mod_{time_format}.npy'))
            
            ### Retrieving corresponding predictions between trainings (fit/bd + fit/bd_std)
            OT_fit_mod = outside_training_model[:, 0] 
            OT_BD_mod = outside_training_model[:, 1:3] 
            OT_fit_var_mod = outside_training_model[:, 3:4] 
            OT_BD_var_mod = outside_training_model[:, 4:5]

            OT_fit_func = outside_training_function[:, 0]
            OT_BD_func = outside_training_function[:, 1:3]
            OT_fit_var_func = outside_training_function[:, 3:4]
            OT_BD_var_func = outside_training_function[:, 4:5]
        
        # Retrieve BD and fitness without std when using a NON-UQ algorithm
        else:

            ### Retrieving corresponding predictions between trainings (only fit/bd, no fit/bd_std)
            OT_fit_mod = outside_training_model[:, 0] 
            OT_BD_mod = outside_training_model[:, 1:] 
            OT_fit_func = outside_training_function[:, 0]
            OT_BD_func = outside_training_function[:, 1:]

    # Specifically modify variables when using MEMBUQ Implicit (NGLL, mean, stds)
    elif method == "MEMBUQ_Implicit" or method == "MEMBUQ_Implicit_Wipe" or method == "MEMBUQ_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Wipe" or method == "MEMBUQ2_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":

        train_loss = jnp.load(f'{backup}{name}_NGLL_loss_{time_format}.npy')

        ### Predictions of the last test set (no std)
        BD_function = jnp.load(f'{backup}{name}_BD_func_{time_format}.npy') 
        BD_model = jnp.load(f'{backup}{name}_mean_BD_mod_{time_format}.npy') 
        fitness_function = jnp.ravel(jnp.load(f'{backup}{name}_fit_func_{time_format}.npy')) 
        fitness_model = jnp.ravel(jnp.load(f'{backup}{name}_mean_fitness_mod_{time_format}.npy'))
        fitness_var_model = jnp.exp(jnp.ravel(jnp.load(f'{backup}{name}_std_fitness_mod_{time_format}.npy')))
        BD_var_model = jnp.exp(jnp.ravel(jnp.load(f'{backup}{name}_std_BD_mod_{time_format}.npy')))

        # Model has Fitness, BD and associated stds
        OT_fit_mod = outside_training_model[:, 0] 
        OT_BD_mod = outside_training_model[:, 1:3] 
        OT_fit_var_mod = jnp.exp(outside_training_model[:, 3:4]) 
        OT_BD_var_mod = jnp.exp(outside_training_model[:, 4:5]) 

        # Function has only Fitness and BD, not stds
        OT_fit_func = outside_training_function[:, 0] 
        OT_BD_func = outside_training_function[:, 1:3]

        if method == "MEMBUQ_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Wipe" or method == "MEMBUQ2_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":
            OT_fit_var_func = outside_training_function[:, 3:4]
            OT_BD_var_func = outside_training_function[:, 4:5]
            fitness_var_function = jnp.ravel(jnp.load(f'{backup}{name}_std_fit_func_{time_format}.npy'))
            BD_var_function = jnp.ravel(jnp.load(f'{backup}{name}_std_BD_func_{time_format}.npy'))

        path_loss = f'results_sim_arg/{method}/Losses/{name}/{saving_title}{name}_NGLL_losses.png'  

    #################
    #     PATHS     #
    #################

    path_regu_loss = f'results_sim_arg/{method}/Losses/{name}/{saving_title}{name}_training_losses.png'
    
    # Path to store errors of the predictions (no stds, only fit and BD values)
    path_errors_BD = f'results_sim_arg/{method}/Errors/{name}/{saving_title}{name}_BD_Errors_OT.png'
    path_errors_fitness = f'results_sim_arg/{method}/Errors/{name}/{saving_title}{name}_Fitness_Errors_OT.png'
    
    # Additional paths for UQ algorithms for errors: between training and last test set (IT)
    path_errors_BD_std = f'results_sim_arg/{method}/Errors_Std/{name}/{saving_title}{name}_BD_Stds_Errors_OT.png'
    path_errors_fitness_std = f'results_sim_arg/{method}/Errors_Std/{name}/{saving_title}{name}_Fitness_Stds_Errors_OT.png'
    path_errors_BD_IN_std = f'results_sim_arg/{method}/Errors_Std/{name}/{saving_title}{name}_BD_Stds_Errors_IT.png'
    path_errors_fitness_IN_std = f'results_sim_arg/{method}/Errors_Std/{name}/{saving_title}{name}_Fitness_Stds_Errors_IT.png'
    
    # Additional paths for UQ algorithms for prediction comparison: between training and last test set (IT)
    path_comparison_BD_std = f'results_sim_arg/{method}/Comparisons/{name}/{saving_title}{name}_BD_Stds_Comparison_OT.png'
    path_comparison_fitness_std = f'results_sim_arg/{method}/Comparisons/{name}/{saving_title}{name}_Fitness_Stds_Comparison_OT.png'
    path_comparison_BD_IN_std = f'results_sim_arg/{method}/Comparisons/{name}/{saving_title}{name}_BD_Stds_Comparison_IT.png'
    path_comparison_fitness_IN_std = f'results_sim_arg/{method}/Comparisons/{name}/{saving_title}{name}_Fitness_Stds_Comparison_IT.png'

    # Path for BD, Fitness, Viobox plots w/ (O) and w/o (NO) outliers for both between trainings (OT) and last test set (IT)
    path_BD_task_IT_NO = f'results_sim_arg/{method}/{direct_1[0]}/{name}/{direct_3[0]}/{direct_4[0]}/'
    path_BD_task_IT_O = f'results_sim_arg/{method}/{direct_1[0]}/{name}/{direct_3[0]}/{direct_4[1]}/'
    path_BD_task_OT_NO = f'results_sim_arg/{method}/{direct_1[0]}/{name}/{direct_3[1]}/{direct_4[0]}/'
    path_BD_task_OT_O = f'results_sim_arg/{method}/{direct_1[0]}/{name}/{direct_3[1]}/{direct_4[1]}/'
    path_fitness_task_IT_NO = f'results_sim_arg/{method}/{direct_1[1]}/{name}/{direct_3[0]}/{direct_4[0]}/'
    path_fitness_task_IT_O = f'results_sim_arg/{method}/{direct_1[1]}/{name}/{direct_3[0]}/{direct_4[1]}/'
    path_fitness_task_OT_NO = f'results_sim_arg/{method}/{direct_1[1]}/{name}/{direct_3[1]}/{direct_4[0]}/'
    path_fitness_task_OT_O = f'results_sim_arg/{method}/{direct_1[1]}/{name}/{direct_3[1]}/{direct_4[1]}/'
    path_viobox_task_IT_NO = f'results_sim_arg/{method}/{direct_1[2]}/{name}/{direct_3[0]}/{direct_4[0]}/'
    path_viobox_task_IT_O = f'results_sim_arg/{method}/{direct_1[2]}/{name}/{direct_3[0]}/{direct_4[1]}/'
    path_viobox_task_OT_NO = f'results_sim_arg/{method}/{direct_1[2]}/{name}/{direct_3[1]}/{direct_4[0]}/'
    path_viobox_task_OT_O = f'results_sim_arg/{method}/{direct_1[2]}/{name}/{direct_3[1]}/{direct_4[1]}/'

    # Plot training and validation losses, then plot regularized loss
    plot_NN_loss(
        loss_train=train_loss, 
        loss_val=val_loss, 
        method=method, 
        name=name, 
        path=path_loss, 
        bool_mse=True, 
        num_epochs=num_epochs
    )
    plot_NN_loss(
        loss_train=training_loss, 
        loss_val=None, 
        method=method, 
        name=name, 
        path=path_regu_loss, 
        bool_mse=False, 
        num_epochs=num_epochs
    )

    # Average fitness error at each generation so between training
    difference_fitness = jnp.abs(OT_fit_mod - OT_fit_func)
    avg_fit_mod = moving_avg(input_list=difference_fitness, batch_size=batch_size, bool_fit=True)
    plot_errors(
        input_list=avg_fit_mod, 
        marker_size=marker_size, 
        saving_path=path_errors_fitness, 
        bool_fit=True, 
        method=method, 
        name=name
    )
    
    # Average BD error at each generation so between training
    difference_bd = jnp.abs(OT_BD_mod - OT_BD_func) 
    avg_BD_mod = moving_avg(input_list=difference_bd, batch_size=batch_size, bool_fit=False)
    plot_errors(
        input_list=avg_BD_mod, 
        marker_size=marker_size, 
        saving_path=path_errors_BD, 
        bool_fit=False, 
        method=method, 
        name=name
    )

    # Additional plots for UQ algorithms (only Explicit: we don't have access to Fitness/BD_stds with the scoring function with Implicit)
    if method == "MEMBUQ_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Wipe" or method == "MEMBUQ2_NLL_Explicit_Naive" or method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":

        #################
        #     ERRORS    #
        #################

        # Average fitness STD error at each generation so between training
        difference_fitness_var = jnp.abs(OT_fit_var_mod - OT_fit_var_func)
        avg_fit_var_mod = moving_avg(input_list=difference_fitness_var, batch_size=batch_size, bool_fit=True)
        plot_errors_std(
            input_list=avg_fit_var_mod, 
            marker_size=marker_size, 
            saving_path=path_errors_fitness_std, 
            bool_gen=True, 
            bool_fit=True, 
            method=method, 
            name=name
        )

        # Average BD STD error at each generation so between training
        difference_bd_var = jnp.abs(OT_BD_var_mod - OT_BD_var_func) 
        avg_BD_var_mod = moving_avg(input_list=difference_bd_var, batch_size=batch_size, bool_fit=False)
        plot_errors_std(
            input_list=avg_BD_var_mod, 
            marker_size=marker_size, 
            saving_path=path_errors_BD_std, 
            bool_gen=True, 
            bool_fit=False, 
            method=method, 
            name=name
        )

        # Average fitness STD error with last test set
        difference_fitness_in_var = jnp.abs(fitness_var_model - fitness_var_function)
        avg_fit_in_var_mod = moving_avg(input_list=difference_fitness_in_var, batch_size=batch_size, bool_fit=True)
        plot_errors_std(
            input_list=avg_fit_in_var_mod, 
            marker_size=marker_size, 
            saving_path=path_errors_fitness_IN_std, 
            bool_gen=False, 
            bool_fit=True, 
            method=method, 
            name=name
        )

        # Average BD STD error with last test set
        difference_bd_in_var = jnp.abs(BD_var_model - BD_var_function) 
        avg_BD_in_var_mod = moving_avg(input_list=difference_bd_in_var, batch_size=batch_size, bool_fit=False)
        plot_errors_std(
            input_list=avg_BD_in_var_mod, 
            marker_size=marker_size, 
            saving_path=path_errors_BD_IN_std, 
            bool_gen=False, 
            bool_fit=False, 
            method=method, 
            name=name
        )
        
        #################
        #  COMPARISONS  #
        #################

        # Comparison of fitness std predictions between model and scoring function between training 
        avg_OT_fit_var_func = moving_avg(input_list=OT_fit_var_func, batch_size=batch_size, bool_fit=True)
        avg_OT_fit_var_mod = moving_avg(input_list=OT_fit_var_mod, batch_size=batch_size, bool_fit=True)
        plot_comparison(
            saving_path=path_comparison_fitness_std, 
            scoring_list=avg_OT_fit_var_func, 
            model_list=avg_OT_fit_var_mod, 
            bool_gen=True, 
            bool_fit=True, 
            name=name, 
            method=method
        )
        
        # Comparison of BD std predictions between model and scoring function between training
        avg_OT_BD_var_func = moving_avg(input_list=OT_BD_var_func, batch_size=batch_size, bool_fit=False)
        avg_OT_BD_var_mod = moving_avg(input_list=OT_BD_var_mod, batch_size=batch_size, bool_fit=False)
        plot_comparison(
            saving_path=path_comparison_BD_std, 
            scoring_list=avg_OT_BD_var_func, 
            model_list=avg_OT_BD_var_mod, 
            bool_gen=True, 
            bool_fit=False, 
            name=name, 
            method=method
        )
        
        # Comparison of fitness std predictions between model and scoring function with last test set
        avg_fitness_var_function = moving_avg(input_list=fitness_var_function, batch_size=batch_size, bool_fit=True)
        avg_fitness_var_model = moving_avg(input_list=fitness_var_model, batch_size=batch_size, bool_fit=True)
        plot_comparison(
            saving_path=path_comparison_fitness_IN_std, 
            scoring_list=avg_fitness_var_function, 
            model_list=avg_fitness_var_model, 
            bool_gen=False, 
            bool_fit=True, 
            name=name, 
            method=method
        )
        
        # Comparison of BD std predictions between model and scoring function with last test set
        avg_BD_var_function = moving_avg(input_list=BD_var_function, batch_size=batch_size, bool_fit=False)
        avg_BD_var_model = moving_avg(input_list=BD_var_model, batch_size=batch_size, bool_fit=False)
        plot_comparison(
            saving_path=path_comparison_BD_IN_std, 
            scoring_list=avg_BD_var_function, 
            model_list=avg_BD_var_model, 
            bool_gen=False, 
            bool_fit=False, 
            name=name, 
            method=method
        )

    #################
    #    Outliers   #
    #################
    
    # During training
    plot_BD_figures(
        BD_function=BD_function, 
        BD_model=BD_model,  
        name=name, 
        algo_name=method,
        path_comparison=f'{path_BD_task_IT_O}{saving_title}{name}_BD_comparison.png', 
        path_difference=f'{path_BD_task_IT_O}{saving_title}{name}_BD_differences.png'
    )
    
    plot_fitness_figures(
        fitness_function=fitness_function, 
        fitness_model=fitness_model, 
        name=name, 
        path=f'{path_fitness_task_IT_O}{saving_title}{name}_Fitness_differences.png'
    )

    plot_viobox_figures(
        fitness_func=fitness_function,     
        fitness_mod=fitness_model, 
        BD_function=BD_function, 
        BD_model=BD_model, 
        path_fitness_vio=f'{path_viobox_task_IT_O}{saving_title}{name}_violin_fitness.png', 
        path_fitness_box=f'{path_viobox_task_IT_O}{saving_title}{name}_box_fitness.png',
        path_BD_vio=f'{path_viobox_task_IT_O}{saving_title}{name}_violin_BD.png', 
        path_BD_box=f'{path_viobox_task_IT_O}{saving_title}{name}_box_BD.png', 
        name=name,
        algo_name=method
    )
    
    # Outside training
    plot_BD_figures(
        BD_function=OT_BD_func, 
        BD_model=OT_BD_mod,  
        name=name, 
        algo_name=method,
        path_comparison=f'{path_BD_task_OT_O}{saving_title}{name}_BD_comparison.png', 
        path_difference=f'{path_BD_task_OT_O}{saving_title}{name}_BD_differences.png'
    )

    plot_fitness_figures(
        fitness_function=OT_fit_func, 
        fitness_model=OT_fit_mod, 
        name=name, 
        path=f'{path_fitness_task_OT_O}{saving_title}{name}_Fitness_differences.png'
    )

    plot_viobox_figures(
        fitness_func=OT_fit_func,     
        fitness_mod=OT_fit_mod, 
        BD_function=OT_BD_func, 
        BD_model=OT_BD_mod, 
        path_fitness_vio=f'{path_viobox_task_OT_O}{saving_title}{name}_violin_fitness.png', 
        path_fitness_box=f'{path_viobox_task_OT_O}{saving_title}{name}_box_fitness.png',
        path_BD_vio=f'{path_viobox_task_OT_O}{saving_title}{name}_violin_BD.png', 
        path_BD_box=f'{path_viobox_task_OT_O}{saving_title}{name}_box_BD.png', 
        name=name,
        algo_name=method
    )
    
    #################
    #  No Outliers  #
    #################

    # During training
    BD_function_no_outliers, BD_model_no_outliers, fitness_func_no_outliers, fitness_mod_no_outliers = remove_outliers(
        BD_function=BD_function, 
        BD_model=BD_model, 
        fitness_func=fitness_function, 
        fitness_mod=fitness_model, 
        name=name
    )

    plot_BD_figures(
        BD_function=BD_function_no_outliers, 
        BD_model=BD_model_no_outliers, 
        name=name, 
        algo_name=method,
        path_comparison=f'{path_BD_task_IT_NO}{saving_title}{name}_BD_comparison.png', 
        path_difference=f'{path_BD_task_IT_NO}{saving_title}{name}_BD_differences.png'
    )

    plot_fitness_figures(
        fitness_function=fitness_func_no_outliers, 
        fitness_model=fitness_mod_no_outliers, 
        name=name, 
        path=f'{path_fitness_task_IT_NO}{saving_title}{name}_Fitness_differences.png'
    )

    plot_viobox_figures(
        fitness_func=fitness_func_no_outliers,     
        fitness_mod=fitness_mod_no_outliers, 
        BD_function=BD_function_no_outliers, 
        BD_model=BD_model_no_outliers, 
        path_fitness_vio=f'{path_viobox_task_IT_NO}{saving_title}{name}_violin_fitness.png', 
        path_fitness_box=f'{path_viobox_task_IT_NO}{saving_title}{name}_box_fitness.png', 
        path_BD_vio=f'{path_viobox_task_IT_NO}{saving_title}{name}_violin_BD.png', 
        path_BD_box=f'{path_viobox_task_IT_NO}{saving_title}{name}_box_BD.png', 
        name=name,
        algo_name=method
    )
    
    # Outside training
    OT_BD_function_no_outliers, OT_BD_model_no_outliers, OT_fitness_func_no_outliers, OT_fitness_mod_no_outliers = remove_outliers(
        BD_function=OT_BD_func, 
        BD_model=OT_BD_mod, 
        fitness_func=OT_fit_func, 
        fitness_mod=OT_fit_mod, 
        name=name
    )

    plot_BD_figures(
        BD_function=OT_BD_function_no_outliers, 
        BD_model=OT_BD_model_no_outliers, 
        name=name, 
        algo_name=method,
        path_comparison=f'{path_BD_task_OT_NO}{saving_title}{name}_BD_comparison.png', 
        path_difference=f'{path_BD_task_OT_NO}{saving_title}{name}_BD_differences.png'
    )

    plot_fitness_figures(
        fitness_function=OT_fitness_func_no_outliers, 
        fitness_model=OT_fitness_mod_no_outliers, 
        name=name, 
        path=f'{path_fitness_task_OT_NO}{saving_title}{name}_Fitness_differences.png'
    )

    plot_viobox_figures(
        fitness_func=OT_fitness_func_no_outliers,     
        fitness_mod=OT_fitness_mod_no_outliers, 
        BD_function=OT_BD_function_no_outliers, 
        BD_model=OT_BD_model_no_outliers, 
        path_fitness_vio=f'{path_viobox_task_OT_NO}{saving_title}{name}_violin_fitness.png', 
        path_fitness_box=f'{path_viobox_task_OT_NO}{saving_title}{name}_box_fitness.png', 
        path_BD_vio=f'{path_viobox_task_OT_NO}{saving_title}{name}_violin_BD.png', 
        path_BD_box=f'{path_viobox_task_OT_NO}{saving_title}{name}_box_BD.png', 
        name=name,
        algo_name=method
    )

if method == "MEMBUQ_NLL_Explicit_Naive_Both_Wipe":

    OT_fit_mod = outside_training_model[:, 0] 
    OT_BD_mod = outside_training_model[:, 1:3]

    OT_fit_func_arm = outside_training_function_arm[:,0]
    OT_BD_func_arm = outside_training_function_arm[:,1:3]

    OT_fit_func_noisy_arm = outside_training_function_noisy_arm[:,0]
    OT_BD_func_noisy_arm = outside_training_function_noisy_arm[:,1:3]

    # Difference between (1) Model predictions in the Uncertain Environment and Deterministic Scoring Function, 
    # Difference between (2) Deterministic Scoring Function and Uncertain Scoring Function
    diff_fit_mod_arm = jnp.abs(OT_fit_mod - OT_fit_func_arm)
    diff_BD_mod_arm = jnp.abs(OT_BD_mod - OT_BD_func_arm)
    diff_fit_arm_noisy_arm = jnp.abs(OT_fit_func_arm - OT_fit_func_noisy_arm)
    diff_BD_arm_noisy_arm = jnp.abs(OT_BD_func_arm - OT_BD_func_noisy_arm)

    # Average of the error by batch size to have only one value per generation
    avg_fit_mod_arm = moving_avg(input_list=diff_fit_mod_arm, batch_size=batch_size, bool_fit=True)
    avg_BD_mod_arm = moving_avg(input_list=diff_BD_mod_arm, batch_size=batch_size, bool_fit=False)
    avg_fit_arm_noisy_arm = moving_avg(input_list=diff_fit_arm_noisy_arm, batch_size=batch_size, bool_fit=True)
    avg_BD_arm_noisy_arm = moving_avg(input_list=diff_BD_arm_noisy_arm, batch_size=batch_size, bool_fit=False)

    # Violin Plots of the two errors for Fitness, BDX and BDY
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 11))
    data1 = [avg_BD_mod_arm[:, 0], avg_BD_arm_noisy_arm[:, 0]]
    data2 = [avg_BD_mod_arm[:, 1], avg_BD_arm_noisy_arm[:, 1]]
    data3 = [avg_fit_mod_arm, avg_fit_arm_noisy_arm]

    # ax1.set_xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Absolute BD-X Errors', fontsize=12, fontweight='bold')
    ax1.set_title(f"BD-X errors between model, deterministic & uncertain arm \nwith MEMBUQ Explicit with Reset when model is not training", fontsize=14, fontweight='bold')
    sns.violinplot(data=data1, palette=["orange", "blue"], ax=ax1)
    ax1.legend(labels=['Avg BD Error X - Model & Deterministic Arm', 'Avg BD Error X - Deterministic Arm & Uncertain Arm'], loc="upper right")

    # ax2.set_xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Absolute BD-Y Errors', fontsize=12, fontweight='bold')
    ax2.set_title(f"BD-Y errors between model, deterministic & uncertain arm \nwith MEMBUQ Explicit with Reset when model is not training", fontsize=14, fontweight='bold')
    sns.violinplot(data=data2, palette=["orange", "blue"], ax=ax2)
    ax2.legend(labels=['Avg BD Error Y - Model & Deterministic Arm', 'Avg BD Error Y - Deterministic Arm & Uncertain Arm'], loc="upper right")

    # ax3.set_xlabel('Index of the predictions (generation)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Absolute Fitness Error', fontsize=12, fontweight='bold')
    ax3.set_title(f"Fitness errors between model, deterministic & uncertain arm  \nwith MEMBUQ Explicit with Reset when model is not training for the {name}", fontsize=14, fontweight='bold')
    sns.violinplot(data=data3, palette=["orange", "blue"], ax=ax3)
    ax3.legend(labels=['Avg Fitness Error - Model & Deterministic Arm', 'Avg Fitness Error - Deterministic Arm & Uncertain Arm'], loc="upper right")

    plt.tight_layout()
    plt.savefig(f'results_sim_arg/{method}/Errors/Combined_Errors_OT.png')
