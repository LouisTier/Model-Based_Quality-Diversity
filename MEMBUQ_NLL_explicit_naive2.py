"""
This work is based on the QDax framework: https://github.com/adaptive-intelligent-robotics/QDax
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}

This work is also based on the UQD framework: https://github.com/adaptive-intelligent-robotics/Uncertain_Quality_Diversity
@misc{flageat2023uncertain,
      title={Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains}, 
      author={Manon Flageat and Antoine Cully},
      year={2023},
      eprint={2302.00463},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}

This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

"""Test default rastrigin using MAP Elites"""

import argparse
import functools
import time
import datetime
import tqdm

from matplotlib import pyplot as plt
from qdax.core.neuroevolution.buffers.buffer import TransitionBuffer_UQ

import jax
import jax.numpy as jnp
import optax

import pickle

from qdax.core.containers.mapelites_repertoire_UQ import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter, MixingEmitter_UQ
from qdax.core.map_elites_MBUQ_NLL_explicit_naive2 import MAPElites
from qdax.tasks.arm import arm_scoring_function, noisy_arm_scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics_UQ
from qdax.utils.plotting import plot_map_elites_results_UQ
from qdax.core.stochasticity_utils_UQ import reevaluation_function, sampling
from qdax.core.neuroevolution.networks.networks import MLP


from annexed_methods import create_directory
import os

method = "MEMBUQ2_NLL_Explicit_Naive"

# current_path = os.path.dirname(os.path.realpath(__file__))
# result_path = os.path.join(current_path, "results_sim_arg")
# create_directory(result_path)
# MEMB_path = os.path.join(result_path, method)
# create_directory(MEMB_path)
# MEMB_rep_path = os.path.join(MEMB_path, "repertoires")
# create_directory(MEMB_rep_path)
# arm_path = os.path.join(MEMB_rep_path, "arm")
# create_directory(arm_path)
# arm_correct_path = os.path.join(arm_path, "corrected")
# create_directory(arm_correct_path)
# arm_uncorrect_path = os.path.join(arm_path, "uncorrected")
# create_directory(arm_uncorrect_path)
# noisy_arm_path = os.path.join(MEMB_rep_path, "noisy_arm")
# create_directory(noisy_arm_path) 
# noisy_arm_correct_path = os.path.join(noisy_arm_path, "corrected")
# create_directory(noisy_arm_correct_path)
# noisy_arm_uncorrect_path = os.path.join(noisy_arm_path, "uncorrected")
# create_directory(noisy_arm_uncorrect_path)
# metric_path = os.path.join(result_path, "Metrics_Comparison")
# create_directory(metric_path)
# container_path = os.path.join(result_path, "Containers_Comparison")
# create_directory(container_path)
# losses_path = os.path.join(MEMB_path, "Losses")
# create_directory(losses_path)
# fitness_path = os.path.join(MEMB_path, "Fitness_Comparison")
# create_directory(fitness_path)
# bd_path = os.path.join(MEMB_path, "BD_Comparison")
# create_directory(bd_path)
# backup_path = os.path.join(MEMB_path, "Backup")
# create_directory(backup_path)

### Define scoring function (fitness or another score) ==> Evaluate the environment
scoring_functions = {
    "arm": functools.partial(arm_scoring_function),
    "noisy_arm": functools.partial(
        noisy_arm_scoring_function,
        fit_variance=0.01, ### Noise added to the fitnesses
        desc_variance=0.05, ### Noise added to the descriptors
        params_variance=0., ### Noise added to the parameters of the controllers (0.05) 
    ), ### fit + desc OR params alone
}

def test_arm(
    task_name: str, 
    batch_size: int,
    num_gen: int,
    num_dof: int,
    first_train: int,
    num_epochs: int,
    num_reeval: int,
    layer_dim: int,
    learning_rate: float,
    log_period: int,
    train_split: float,
    l2_rate: float,
    num_reeval_metrics: int,
    save_dir: str,
    keep_data: float,
    time_format: str,
) -> None:
    """
    Run MEMBUQ Explicit Naive algorithm with NLL loss

    Args:
        - task_name: Name of the task, so arm or noisy_arm
        - batch_size: Number of elements to be evaluated in parallel
        - num_gen: Number of generations
        - num_dof: Number of degrees of freedom (joints)
        - first_train: Generation number of the first training of the model
        - num_epochs: Number of epochs to train the model
        - num_reeval: Number of reevaluations of the solutions
        - layer_dim: Dimension of a hidden layer of the model
        - learning_rate: LR of the optimizer
        - log_period: Generation frequency to collect new metrics while running main algorithms
        - train_split: Repartition of the data between training and testing
        - l2_rate: Weight of the L2 regularization term of the training loss
        - num_reeval_metrics: Number of reevaluations of each solution for the corrected metrics
        - save_dir: Path to save final results
        - keep_data: Percentage of data kept between two trainings
        - time_format: Time of the beginning of the simulation
        
    Returns:
        - Nothing but saves the corresponding csv metrics and png 
    """
    
    ### Randomness of the simulation (numpy)
    seed = 42

    ### Number of DoF of the arm.
    num_param_dimensions = num_dof

    ### Solution/ Sample considered at each forward pass (or main loop/generation) and evaluated in parallel.
    batch_size = batch_size

    # 1 - 1/x = % to keep old data so 3 ==> 2/3 of old data is kept between each training, 2 ==> 1/2 is kept, 4 ==> 3/4 is kept, 1.4 ==> 30% is kept
    keeping_size = 1 / (1-keep_data)

    ### Number of generations (= batch_size episodes). Main QD loop = selection, mutation, evaluation, addition  
    num_iterations = num_gen # Number of generations: 3000
    model_start_training = first_train # Number of generations to fill container before training the model: 500
    freq_training = model_start_training/2 # Number of generations between 2 training except the first one at model_start_training generations: 250
    buffer_size = int(keeping_size * freq_training * batch_size) # Size of the buffer to store genotypes and scores  
    num_epochs = num_epochs # Number of epochs at each training: 125 
    nb_reeval = num_reeval # Number of reevaluations for the scoring function and the model: 8

    layer_size = layer_dim # Number of neurons in the Dense layer of the NN: 8
    output_size = 6 # Fitness, BD1, BD2, Fitness_Std, BD1_Std and BD2_Std for the NN predictions: 6
    learning_rate = learning_rate # Learning rate of the optimizer: 0.007
    log_period = log_period # Update frequency of metrics: every 10 containers
    train_split = train_split # Train split for the buffer: 0.7
    l2_rate = l2_rate # Weight of the L2 loss in the training: 0.000001

    print("\nNumber of trainings: ", (num_iterations/freq_training)) 
    print("Maximal size of the buffer: ", buffer_size)
    print("Percentage of data to be retained: ", keep_data, "keeping_size: ", keeping_size)

    ### Buffer to store transitions (genotypes + scores) obtained via the scoring function
    replay_buffer = TransitionBuffer_UQ(
        buffer_size=buffer_size
    ) 

    buffer = None
    
    ### Will be use to calculate the number of centroids. It is the number of centroids along X and Y axis
    grid_shape = (100, 100) 

    ### Scale Normalization for the controllers (population) and so the variations
    min_param = 0.0
    max_param = 1.0

    ### Scale Normalization for the behavioural descriptor
    min_bd = 0.0
    max_bd = 1.0

    ### Init a random key to consume (Jax = local randomness) based on the global randomness (Numpy)
    random_key = jax.random.PRNGKey(seed)

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    init_variables = jax.random.uniform(
        subkey,
        shape=(batch_size, num_param_dimensions),
        minval=min_param, 
        maxval=max_param, 
    )

    # Prepare the scoring function with the corresponding task (arm or noisy_arm)
    scoring_fn = scoring_functions[task_name]
    sampling_fn = functools.partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=nb_reeval,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=0.05,
        line_sigma=0.1,
        minval=min_param, 
        maxval=max_param, 
    )

    ### Defining how to evolve a generation
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y), 
        variation_fn=variation_fn, 
        variation_percentage=1.0, 
        batch_size=batch_size,
    )

    # Define a metrics function ==> Scoring can be novelty, fitness, curiosity and Metrics can be diversity, performance, convergence speed or QD score.
    metrics_fn = functools.partial(
        default_qd_metrics_UQ, 
        qd_offset=1, 
    ) 

    # Defining the NN that will be used
    random_key, subkey = jax.random.split(random_key)
    MLP_layer_sizes = (layer_size, layer_size) + (output_size,) 
    MLP_network = MLP(
        layer_sizes=MLP_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=None, 
    )

    # # Progressively decay the learning rate: decayed_value = init_value * decay_rate ^ (count / transition_steps) `
    # scheduler = optax.exponential_decay(
    #     init_value=learning_rate, 
    #     transition_steps=250, 
    #     decay_rate=0.9, # 0.99 | 1 if we don't want decay
    #     transition_begin=1, 
    #     staircase=False, 
    #     end_value=None
    # )

    # Defining the optimizer that will be used
    network_optimizer = optax.adam(learning_rate=learning_rate)
    # network_optimizer = optax.adam(learning_rate=scheduler)
    
    # Instantiate MAP-Elites
    map_elites = MAPElites( 
        scoring_function=sampling_fn, 
        emitter=mixing_emitter, 
        metrics_function=metrics_fn, 
        network=MLP_network,
        network_optimizer = network_optimizer,
        l2_rate=l2_rate,
        reevaluations=nb_reeval
    )

    # Compute the centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd, 
        maxval=max_bd,
    ) 

    # Compute initial repertoire
    repertoire, emitter_state, random_key, model_state = map_elites.init(
        init_variables, centroids, random_key
    )

    # print("model dict parameters after init: \n", model_state['network_params'])

    ### Dictionnary to store the different metrics of the algorithm
    all_metrics = {}
    all_metrics_reeval = {}
    os.makedirs(f"{save_dir}/{method}", exist_ok=True)

    ### Save uncorrected metrics of an experiment in a csv file during the training process.
    csv_logger = CSVLogger(
        f"{save_dir}/{method}/{method}_{task_name}_{time_format}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "std_fitness", "std_bd", "time"]
    )

    ### Save corrected metrics of an experiment in a csv file during the training process.
    csv_logger_reeval = CSVLogger(
        f"{save_dir}/{method}/reeval_{method}_{task_name}_{time_format}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "std_fitness", "std_bd", "time"]
    )

    # print()

    NN_training_loss = []
    NN_ngll_loss = []
    NN_val_loss = []
    OT_model_predictions = []
    OT_function_predictions = []

    ### Perform MAP-Elites for num_loops generations
    for i in tqdm.tqdm(range(num_iterations)):
    # for i in range(num_iterations):
       
        if i==0:
            print(f"\n\nBeginning with {method}, the {task_name} and {num_iterations} generations with a batch of size {batch_size} at {time_format} \n")
        
        # For the first generations, we only use the scoring function for both container and buffer
        if i <= model_start_training:

            ### Keeping track of the time required for each generation
            start_time = time.time()

            repertoire, genotypes, fitnesses, fitnesses_var, descriptors, descriptors_var, emitter_state, metrics, random_key = map_elites.update_buffer_beginning(
                repertoire=repertoire,
                emitter_state=emitter_state,
                random_key=random_key,
            )
            
            ### Keeping track of the time required for each generation
            timelapse = time.time() - start_time

            ### Adding scores and genotypes in the buffer
            buffer = replay_buffer.add_transitions_explicit(
                buffer=buffer,
                fitnesses_var=fitnesses_var,
                descriptors_var=descriptors_var,
                genotypes=genotypes,
                fitnesses=fitnesses,
                # descriptors=descriptors
                bd1=descriptors[:,0],
                bd2=descriptors[:,1]
            )

            # print("buffer: ", buffer)
        
        # After the first generations, buffer via scoring function and container via model 
        else:
            
            start_time = time.time()

            ### Producing genotypes and associated scores with the scoring function
            genotypes, fitnesses, fitnesses_var, descriptors, descriptors_var, random_key = map_elites.generate_batch(
                repertoire=repertoire,
                emitter_state=emitter_state,
                random_key=random_key,
            )

            ### Adding scores and genotypes in the buffer
            buffer = replay_buffer.add_transitions_explicit(
                buffer=buffer,
                fitnesses_var=fitnesses_var,
                descriptors_var=descriptors_var,
                genotypes=genotypes,
                fitnesses=fitnesses,
                # descriptors=descriptors
                bd1=descriptors[:,0],
                bd2=descriptors[:,1]
            )

            ### Predicting scores with the model
            mean_fitness, mean_bd1, mean_bd2, std_fitness, std_bd1, std_bd2 = map_elites.model_batch_predict(
                model_dict=model_state,
                genotypes=genotypes,
                norm_param=norm_parameters
            )

            mean_bd = jnp.stack((mean_bd1, mean_bd2), axis=1)
            std_bd = (std_bd1 + std_bd2) / 2

            # NEED TO MODIFY TO ADD STDS IN CONTAINER
            repertoire, emitter_state, metrics = map_elites.update_via_model(
                repertoire=repertoire,
                emitter_state=emitter_state,
                genotypes=genotypes,
                fitnesses=mean_fitness,
                descriptors=mean_bd,
                fitnesses_var=std_fitness,
                descriptors_var=std_bd,
                extra_scores=None,
            )

            timelapse = time.time() - start_time 

        # Every freq_training generations and the final one: train the model 
        if (i != 0 and i % freq_training == 0 and i - model_start_training > 0) or i == model_start_training or i == num_iterations-1:
            
            norm_buffer, norm_parameters, buffer = replay_buffer.normalize_buffer_explicit(buffer=buffer)

            ### Spliting the buffer intro 70% training set and 30% testing set. 
            training_set, validation_set, random_key = replay_buffer.split_buffer(
                buffer=norm_buffer,
                random_key=random_key,
                train_ratio = train_split,
            )

            ### Training the model
            training_loss, ngll_loss, model_state = map_elites.train_model_v2(
                model_dict=model_state,
                norm_param=norm_parameters,
                training_set=training_set,
                num_epochs=num_epochs,
                batch_size=batch_size,
                l2_rate=l2_rate
            )

            ### Testing the model
            val_mean_fitness_mod, val_mean_bd1_mod, val_mean_bd2_mod, val_std_fitness_mod, val_std_bd1_mod, val_std_bd2_mod, val_fit_func, val_bd1_func, val_bd2_func, val_std_fit_func, val_std_bd_func, val_ngll = map_elites.check_model(
                model_dict=model_state,
                norm_param = norm_parameters,
                validation_set=validation_set,
                batch_size=batch_size, 
            )

            val_mean_bd_mod = jnp.stack((val_mean_bd1_mod, val_mean_bd2_mod), axis=1)
            val_std_bd_mod = (val_std_bd1_mod + val_std_bd2_mod) / 2

            val_bd_func = jnp.stack((val_bd1_func, val_bd2_func), axis=1)

            # print("training_loss: ", training_loss)
            # print("ngll_loss: ", ngll_loss)
            # print("val_ngll: ", val_ngll)

            ### Keeping track of training and test losses
            NN_training_loss.append(training_loss)
            NN_ngll_loss.append(ngll_loss)
            NN_val_loss.append(val_ngll)
            

        # When model is not training, use it to predict and compare its predictions with the scoring function in the future
        # to_predict generation before training, the NN uses the repertoire to predict some scores 
        elif i != 0 and i > model_start_training:

            preds_model = jnp.concatenate((mean_fitness.reshape((-1,1)), mean_bd, std_fitness.reshape((-1,1)), std_bd.reshape((-1,1))), axis=1)
            preds_func = jnp.concatenate((fitnesses.reshape((-1,1)), descriptors, fitnesses_var.reshape((-1,1)), descriptors_var.reshape((-1,1))), axis=1)
            OT_model_predictions.append(preds_model)
            OT_function_predictions.append(preds_func)

        # Each 10 loops
        if i%log_period == 0:

            ### Keeping track of the time required for each generation
            start_time_reeval = time.time()
            
            ### Performing corrected MAP Elites
            (reeval_repertoire, _, _, _, _, random_key) = reevaluation_function(
                repertoire=repertoire,
                metric_repertoire=repertoire,
                random_key=random_key,
                scoring_fn=scoring_fn, 
                num_reevals=num_reeval_metrics
            )
            
            ### Keeping track of the time required for each generation
            timelapse_reeval = time.time() - start_time_reeval

            ##### Need to rewrite metrics like above
            reeval_metrics = metrics_fn(reeval_repertoire)

            # log metrics
            logged_metrics = {"time": timelapse, "loop": int(1+i/log_period), "iteration": 1 + i}

            ### For all metrics ==> what are the metrics (and the order) ? It is "loop", "iteration", "qd_score", "max_fitness", "coverage", "time"
            for key, value in metrics.items():

                ### Take last value of corresponding metric
                logged_metrics[key] = value

                ### Concatenate current value with previous ones
                if key in all_metrics.keys():
                    all_metrics[key] = jnp.append(all_metrics[key], value)

                ### If this is the first value of the corresponding metric, no need to concatenate
                else:
                    all_metrics[key] = value

            ### Logging/writing new metrics to the csv file every 10 generations (log_period).
            csv_logger.log(logged_metrics)

            # log metrics
            logged_metrics_reeval = {"time": timelapse_reeval, "loop": int(1+i/log_period), "iteration": 1 + i}

            ## For all metrics ==> what are the metrics (and the order) ? It is "loop", "iteration", "qd_score", "max_fitness", "coverage", "time"
            for key, value in reeval_metrics.items():

                ### Take last value of corresponding metric
                logged_metrics_reeval[key] = value

                ### Concatenate current value with previous ones
                if key in all_metrics_reeval.keys():
                    all_metrics_reeval[key] = jnp.append(all_metrics_reeval[key], value)

                ### If this is the first value of the corresponding metric, no need ot concatenate
                else:
                    all_metrics_reeval[key] = value

            csv_logger_reeval.log(logged_metrics_reeval)

    # Number of evaluations performed during the whole algorithm (uncorrected)
    env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period * nb_reeval

    # create the plots and the grid with corresponding uncorrected metric and final repertoire
    fig, axes = plot_map_elites_results_UQ(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'{save_dir}/{method}/{method}_{task_name}_{time_format}.png')

    # Number of evaluations performed during the whole algorithm (corrected)
    env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period * nb_reeval

    # create the plots and the grid with corresponding orrected metric and final repertoire
    fig_reeval, axes_reeval = plot_map_elites_results_UQ(env_steps=env_steps, metrics=all_metrics_reeval, repertoire=reeval_repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'{save_dir}/{method}/reeval_{method}_{task_name}_{time_format}.png')

    ### Adjusting elements
    NN_training_loss = jnp.concatenate(NN_training_loss)
    NN_ngll_loss = jnp.concatenate(NN_ngll_loss)
    OT_model_predictions = jnp.concatenate(OT_model_predictions) 
    OT_function_predictions = jnp.concatenate(OT_function_predictions) 

    full_path = f'{save_dir}/{method}/Backup'
    os.makedirs(full_path, exist_ok=True)

    ### Neural Network
    with open(f'{full_path}/{method}_{task_name}_trained_model_{time_format}.pkl', 'wb') as f:
        pickle.dump(model_state, f) 

    ### Training & validations losses 
    jnp.save(f'{full_path}/{method}_{task_name}_training_loss_{time_format}.npy', NN_training_loss)
    jnp.save(f'{full_path}/{method}_{task_name}_NGLL_loss_{time_format}.npy', NN_ngll_loss)
    jnp.save(f'{full_path}/{method}_{task_name}_val_loss_{time_format}.npy', NN_val_loss)
    
    ### Predictions outside training
    jnp.save(f'{full_path}/{method}_{task_name}_OT_model_predictions_{time_format}.npy', OT_model_predictions) 
    jnp.save(f'{full_path}/{method}_{task_name}_OT_function_predictions_{time_format}.npy', OT_function_predictions) 
    
    ### Predictions during training (final test set)
    jnp.save(f'{full_path}/{method}_{task_name}_mean_fitness_mod_{time_format}.npy', val_mean_fitness_mod)
    jnp.save(f'{full_path}/{method}_{task_name}_mean_BD_mod_{time_format}.npy', val_mean_bd_mod)
    jnp.save(f'{full_path}/{method}_{task_name}_std_fitness_mod_{time_format}.npy', val_std_fitness_mod)
    jnp.save(f'{full_path}/{method}_{task_name}_std_BD_mod_{time_format}.npy', val_std_bd_mod)
    jnp.save(f'{full_path}/{method}_{task_name}_fit_func_{time_format}.npy', val_fit_func)
    jnp.save(f'{full_path}/{method}_{task_name}_BD_func_{time_format}.npy', val_bd_func)
    jnp.save(f'{full_path}/{method}_{task_name}_std_fit_func_{time_format}.npy', val_std_fit_func)
    jnp.save(f'{full_path}/{method}_{task_name}_std_BD_func_{time_format}.npy', val_std_bd_func)

    ### Containers
    global_path = f"{save_dir}/{method}/repertoires/{task_name}/"
    final_path_unc = global_path + "uncorrected/"
    final_path_cor = global_path + "corrected/"
    os.makedirs(global_path, exist_ok=True)
    os.makedirs(final_path_unc, exist_ok=True)
    os.makedirs(final_path_cor, exist_ok=True)
    repertoire.save(path=final_path_unc)
    reeval_repertoire.save(path=final_path_cor)

def specify_arg():

    parser = argparse.ArgumentParser()

    # Usually fixed for now
    parser.add_argument("--num_gen", default=3000, type=int, help="Number of generations")
    parser.add_argument("--num_dof", default=8, type=int, help="Number of joints to control")
    parser.add_argument("--batch_size", default=64, type=int, help="Number of solutions to be evaluated at each generation")
    parser.add_argument("--num_reeval", default=8, type=int, help="Number of reevaluations of the solutions")
    parser.add_argument("--layer_dim", default=8, type=int, help="Size of MLP's hidden layers")
    parser.add_argument("--learning_rate", default=0.007, type=float, help="Learning rate of the optimizer (Adam)")
    parser.add_argument("--log_period", default=10, type=int, help="Number of generations to be scanned")
    parser.add_argument("--train_split", default=0.7, type=float, help="Percentage of elements in the dataset used for training")
    parser.add_argument("--l2_rate", default=0.000001, type=float, help="Weight of the Regularized Loss in the L2 Loss)")
    parser.add_argument("--num_reeval_metrics", default=256, type=int, help="Number of reevaluations of each soluton for the corrected metrics")

    # To specify
    parser.add_argument("--save_dir", default="", type=str, help="Directory to save the results")
    parser.add_argument("--task_name", default="", type=str, help="Name of the task/environment")

    # To fine tune
    parser.add_argument("--first_train", default=1000, type=int, help="Number of the generation when the model is trained for the first time")
    parser.add_argument("--num_epochs", default=175, type=int, help="Number of epochs to train the model")
    parser.add_argument("--per_data", default=0.2, type=float, help="Percentage of old data to be retained for next training session")

    return parser.parse_args()

if __name__ == "__main__":

    #################
    #    Classic    #
    #################

    # python3 MEMBUQ_NLL_explicit_naive.py --save_dir test_grid_UQ --task_name arm --first_train 1000 --num_epochs 125 --per_data 0.2
    # python3 MEMBUQ_NLL_explicit_naive.py --save_dir results_sim_arg --task_name noisy_arm --first_train 1000 --num_epochs 125 --per_data 0.2

    current_time = datetime.datetime.now()
    time_format = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    # Read and retrieve input parameter from command line
    args = specify_arg()

    os.makedirs(f'{args.save_dir}/{method}', exist_ok=True) #save dir = $CURPATH/results/$PATHNAME where PATHNAME = 2023-07-25_10_30_45_12345
    
    # Save inputs parameters into txt 
    with open(f'{args.save_dir}/{method}/{args.task_name}_{time_format}.txt', 'w') as file:
        args_with_time = vars(args).copy()  
        args_with_time['beginning_time'] = time_format
        print(args_with_time, file=file)

    start_time = time.time()
    test_arm(
        task_name=args.task_name, 
        batch_size=args.batch_size,
        num_gen=args.num_gen,
        num_dof=args.num_dof,
        first_train=args.first_train,
        num_epochs=args.num_epochs,
        num_reeval=args.num_reeval,
        layer_dim=args.layer_dim,
        learning_rate=args.learning_rate,
        log_period=args.log_period,
        train_split=args.train_split,
        l2_rate=args.l2_rate,
        num_reeval_metrics=args.num_reeval_metrics,
        save_dir=args.save_dir,
        keep_data=args.per_data,
        time_format=time_format
    )
    timelapse = time.time() - start_time 
    print(f"It took {timelapse/60:.2f} minutes for method: {method} | task: {args.task_name} | 1st training: {args.first_train} | nb of epochs: {args.num_epochs} | old data retained: {100*args.per_data}% | time: {time_format}")
    print()

    # #################
    # #      Grid     #
    # #################

    # # python3 MEMBUQ_NLL_explicit_naive.py --save_dir test_grid  --task_name arm

    # # first_train_list = [200, 500, 800, 1000]  
    # # num_epochs_list = [75, 125, 175, 225]         
    # # per_data_list = [0.2, 0.4, 0.6, 0.8] 

    # first_train_list = [500, 800, 1000]  
    # num_epochs_list = [125, 175, 225]         
    # per_data_list = [0.2, 0.4, 0.6, 0.8] 

    # # first_train_list = [200, 500, 1000]  
    # # num_epochs_list = [5, 10]         
    # # per_data_list = [0.1, 0.3] 

    # simulation = 0    
    # total_simulation = len(first_train_list) * len(num_epochs_list) * len(per_data_list)

    # for first_train in first_train_list:
    #     for num_epochs in num_epochs_list:
    #         for per_data in per_data_list:

    #             current_time = datetime.datetime.now()
    #             time_format = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    #             print(f"Simulation: {simulation}/{total_simulation}")
    #             print(f'Time: {time_format}')

    #             # Read and retrieve input parameter from command line
    #             args = specify_arg()

    #             args.first_train = first_train
    #             args.num_epochs = num_epochs
    #             args.per_data = per_data

    #             os.makedirs(f'{args.save_dir}/{method}', exist_ok=True) #save dir = $CURPATH/results/$PATHNAME where PATHNAME = 2023-07-25_10_30_45_12345
                
    #             # Save inputs parameters into txt 
    #             with open(f'{args.save_dir}/{method}/{args.task_name}_sim_{simulation}_{time_format}.txt', 'w') as file:
    #                 args_with_time = vars(args).copy()  
    #                 args_with_time['beginning_time'] = time_format
    #                 print(args_with_time, file=file)

    #             start_time = time.time()
    #             test_arm(
    #                 task_name=args.task_name, 
    #                 batch_size=args.batch_size,
    #                 num_gen=args.num_gen,
    #                 num_dof=args.num_dof,
    #                 first_train=args.first_train,
    #                 num_epochs=args.num_epochs,
    #                 num_reeval=args.num_reeval,
    #                 layer_dim=args.layer_dim,
    #                 learning_rate=args.learning_rate,
    #                 log_period=args.log_period,
    #                 train_split=args.train_split,
    #                 l2_rate=args.l2_rate,
    #                 num_reeval_metrics=args.num_reeval_metrics,
    #                 save_dir=args.save_dir,
    #                 keep_data=args.per_data,
    #                 time_format=time_format
    #             )
    #             timelapse = time.time() - start_time 
    #             print(f"It took {timelapse/60:.2f} minutes for method: {method} | task: {args.task_name} | 1st training: {args.first_train} | nb of epochs: {args.num_epochs} | old data retained: {100*args.per_data}% | sim: {simulation} | time: {time_format}")
    #             simulation += 1

    # #################
    # #     Grid 2    #
    # #################

    # learning_rate_list = [0.0002, 0.0007, 0.002, 0.007, 0.02]
    # l2_rate_list = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    # simulation = 0    
    # total_simulation = len(learning_rate_list) * len(l2_rate_list)

    # for learning_rate in learning_rate_list:
    #     for l2_rate in l2_rate_list:
        
    #         current_time = datetime.datetime.now()
    #         time_format = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    #         print(f"Simulation: {simulation}/{total_simulation}")
    #         print(f'Time: {time_format}')

    #         # Read and retrieve input parameter from command line
    #         args = specify_arg()

    #         args.learning_rate = learning_rate
    #         args.l2_rate = l2_rate

    #         os.makedirs(f'{args.save_dir}/{method}', exist_ok=True) #save dir = $CURPATH/results/$PATHNAME where PATHNAME = 2023-07-25_10_30_45_12345
            
    #         # Save inputs parameters into txt 
    #         with open(f'{args.save_dir}/{method}/{args.task_name}_sim_{simulation}_{time_format}.txt', 'w') as file:
    #             args_with_time = vars(args).copy()  
    #             args_with_time['beginning_time'] = time_format
    #             print(args_with_time, file=file)

    #         start_time = time.time()
    #         test_arm(
    #             task_name=args.task_name, 
    #             batch_size=args.batch_size,
    #             num_gen=args.num_gen,
    #             num_dof=args.num_dof,
    #             first_train=args.first_train,
    #             num_epochs=args.num_epochs,
    #             num_reeval=args.num_reeval,
    #             layer_dim=args.layer_dim,
    #             learning_rate=args.learning_rate,
    #             log_period=args.log_period,
    #             train_split=args.train_split,
    #             l2_rate=args.l2_rate,
    #             num_reeval_metrics=args.num_reeval_metrics,
    #             save_dir=args.save_dir,
    #             keep_data=args.per_data,
    #             time_format=time_format
    #         )
    #         timelapse = time.time() - start_time 
    #         print(f"It took {timelapse/60:.2f} minutes for method: {method} | task: {args.task_name} | learning_rate: {args.learning_rate} | l2_rate: {args.l2_rate} | sim: {simulation} | time: {time_format}")
    #         simulation += 1
