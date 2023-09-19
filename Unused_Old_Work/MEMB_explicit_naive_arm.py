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

import functools
import time
import datetime
import tqdm

from matplotlib import pyplot as plt
from qdax.core.neuroevolution.buffers.buffer import TransitionBuffer

import jax
import jax.numpy as jnp
import optax

# # Not working
# import logging
# logging.getLogger("jax").setLevel(logging.DEBUG)

# # Working
# import jax.config
# jax.config.update('jax_log_compiles', True)

import pytest
import numpy as np
import pickle

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from QDax.qdax.core.map_elites_MB_explicit_naive_old import MAPElites
from qdax.tasks.arm import arm_scoring_function, noisy_arm_scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results
from qdax.core.stochasticity_utils import reevaluation_function, sampling
from qdax.core.neuroevolution.networks.networks import MLP


from annexed_methods import create_directory
import os

method = "MEMB_Explicit_Naive"

# current_path = os.path.dirname(os.path.realpath(__file__))
# result_path = os.path.join(current_path, "results_sim")
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

### Retrieving current time at the execution
current_time = datetime.datetime.now()
time_format = current_time.strftime("%d-%m-%Y_%H-%M-%S")

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

### Parameterizing arguments of the test_arm function to run the test against multiple sets of inputs
@pytest.mark.parametrize(
    "task_name, batch_size",
    [("arm", 128), ("noisy_arm", 128)],
)
def test_arm(task_name: str, batch_size: int) -> None:

    # a = jnp.ones(1)
    # print()
    # print(a.device())
    # jax.debug.print("Test")
    
    ### Randomness of the simulation (numpy)
    seed = 42

    ### Number of DoF of the arm. 8 is an easy task (Cully), 100 is medium and 1000 is hard
    ### dimension of genotype / number of angles to control
    num_param_dimensions = 8  

    ### Just a random initialization, could be w/e
    init_batch_size = batch_size

    ### Solution/ Sample considered at each forward pass (or main loop/generation) and evaluated in parallel.
    ### Each solution has its own episode and there is no interaction between them (world). Here an episode has a length of 1 (instantaneous).
    batch_size = batch_size

    # F) 500, 250, 4, 250 | G) 400, 200, 4, 150 | H) 500, 250, 7, 150 | I) 500, 250, 7, 250
    ### Number of generations (= batch_size episodes). Main QD loop = selection, mutation, evaluation, addition  
    num_iterations = 3000 # Number of generations: 3000
    model_start_training = 500 # Number of generations to fill container before training the model: 500 | 200 (42') | 80 (67') | 80 (100') | 500 (30') | 500 (30') | 500 (X') | 500 (X') 
    freq_training = 250 # Number of generations between 2 training except the first one at model_start_training generations: 250 | 100 | 40 | 40 | 250 | 250 | 250 | 250
    # 3 ==> 2/3 of old data is kept between each training, 2 ==> 1/2 is kept, 4 ==> 3/4 is kept, 1.4 ==> 30% is kept, 1.33 ==> 25%, 1.25 ==> 20%, 7 ==> 85%
    buffer_size = int(7 * freq_training * batch_size) # Size of the buffer to store genotypes and scores:  2 | 2 | 2 | 2 | 2 | 1.25 | 2 | 4 |
    num_epochs = 250 # Number of epochs at each training: 125 | 150 | 100 | 150 | 250 | 250 | 250
    nb_reeval = 8 # Number of reevaluations for the scoring function and the model: 8
    sampling_size = batch_size * nb_reeval

    layer_size = 8 # Number of neurons in the Dense layer of the NN: 8
    output_size = 3 # Fitness, BD1, BD2 for the NN predictions: 3
    learning_rate = 0.007 # Learning rate of the optimizer: 0.007
    log_period = 10 # Update frequency of metrics: every 10 containers
    # to_predict = 3 # Number of generation before each training to compare predictions outside the training: 3
    train_split = 0.7 # Train split for the buffer: 0.7
    l2_rate = 0.000001 # Weight of the L2 loss in the training: 0.000001
    # title = f"Loss_Buffer_test_{layer_size}x{layer_size}_" # Title to save data at the end of the simulation

    print("\nNumber of trainings: ", (num_iterations/freq_training)) 
    print("Maximal size of the buffer: ", buffer_size)
    print("Sampling size: ", sampling_size,"\n")

    ### Buffer to store transitions (genotypes + scores) obtained via the scoring function
    replay_buffer = TransitionBuffer(
        buffer_size=buffer_size
    ) 
    
    ### Will be use to calculate the number of centroids. It is the number of centroids along X and Y axis
    ### High number = little centroids (squares), Low number = huge centroids
    grid_shape = (100, 100) ### (num_centroids, num_descriptors)

    ### Scale Normalization for the controllers (population) and so the variations
    ### Parameter space is normalized between [0;1] which corresponds to [0;2*Pi]
    min_param = 0.0
    max_param = 1.0

    ### Scale Normalization for the behavioural descriptor
    ### Descriptor space (end-effector x-y position) is normalized between 
    min_bd = 0.0
    max_bd = 1.0

    ### Name to save the simulation (.csv and .png)
    saving_path = method + "_" + str(num_iterations) + "_" + str(batch_size) + "_" + task_name + "_" + time_format

    ### Init a random key to consume (Jax = local randomness) based on the global randomness (Numpy)
    random_key = jax.random.PRNGKey(seed)

    # Init population of controllers
    ### This is the genotype used to initialized MAP-Elites
    random_key, subkey = jax.random.split(random_key)
    init_variables = jax.random.uniform(
        subkey, ### Handling Jax randomness
        shape=(sampling_size, num_param_dimensions), ### Population at each generation
        minval=min_param, ### Minimum value of the genotype parameters
        maxval=max_param, ### Maximum value of the genotype parameters
    )

    # Prepare the scoring function with the corresponding task (arm or noisy_arm)
    ### Value of fitness should be between - 0.1 and 0
    scoring_fn = scoring_functions[task_name]

    # Prepare the scoring functions for the offspring generated folllowing
    # the approximated gradient (each of them is evaluated nb_reeval times)
    sampling_fn = functools.partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=nb_reeval,
    )

    # Define emitter
    ### Param for variation function (mutation) ==> Mutation + Crossover (really performant)
    ### Sample 2 parents from archive, line between the parents, add noise (gaussian)
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=0.05, ### Spread parameter (noise)
        line_sigma=0.1, ### Line parameter (direction of the new genotype)
        minval=min_param, ### Minimum value to clip the genotypes
        maxval=max_param, ### Maximum value to clip the genotypes
    )

    ### Defining how to evolve a generation. Difference between mutation and variation:
    ### Mutation (1 parent + noise) and crossover (two parents that we mix) to generate offspring
    ### Variation = mix of the two above (line between two parents + noise)
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y), ### Mutation is doing nothing here?
        variation_fn=variation_fn, ### Over a set of pairs of genotypes
        variation_percentage=1.0, ### 1 = full cross over, 0 = full mutation
        batch_size=batch_size, ### Number of solutions to evolve
    )
    ### Emitter that performs both mutation and variation. Two batches of variation_percentage * batch_size 
    ### genotypes are sampled in the repertoire, copied and cross-over to obtain new offsprings. 
    ### One batch of (1.0 - variation_percentage) * batch_size genotypes are sampled in the repertoire, copied and mutated.

    # Define a metrics function ==> Scoring can be novelty, fitness, curiosity and Metrics can be diversity, performance, convergence speed or QD score.
    ### Scoring = to evaluate sample/solution, metric = to evaluate the algorithm (only for us, the algo doesn't use them)
    metrics_fn = functools.partial(
        default_qd_metrics, ### qd_score, max_fitness, coverage
        qd_offset=1, ### Offset used to ensure that the QD score will be positive and increasing with the number of individuals.
    ) ### Add 1 or -1, lets check default_qd

    # Defining the NN that will be used
    random_key, subkey = jax.random.split(random_key)
    MLP_layer_sizes = (layer_size, layer_size) + (output_size,) #3 = fitness, BD1 and BD2 = env.action_size
    MLP_network = MLP(
        layer_sizes=MLP_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=None, # None = linear activation
    )

    # Defining the optimizer that will be used
    network_optimizer = optax.adam(learning_rate=learning_rate)

    # Instantiate MAP-Elites
    map_elites = MAPElites( #scoring_fn or sampling_fn
        scoring_function=sampling_fn, ### How to evaluate the env. The score is defined intrinsically  by the environment selected (the task)
        emitter=mixing_emitter, ### How to evole the population at each mutation step
        metrics_function=metrics_fn, ### How to evaluate the algorithm (only for the user)
        network=MLP_network,
        network_optimizer = network_optimizer,
        l2_rate=l2_rate,
        reevaluations=nb_reeval
    )

    # Compute the centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape, ### Number of centroids per BD dimension
        minval=min_bd, ### Minimum descriptors value
        maxval=max_bd, ### Maximum descriptors value
    ) ### Returns centroids with shape (num_centroids, num_descriptors)

    # Compute initial repertoire
    ### Can we see the emitter_state as the state of the controllers?
    repertoire, emitter_state, random_key, model_state = map_elites.init(
        init_variables, centroids, random_key
    )

    ### Dictionnary to store the different metrics of the algorithm
    all_metrics = {}
    all_metrics_reeval = {}

    ### Save uncorrected metrics of an experiment in a csv file during the training process.
    csv_logger = CSVLogger(
        f"results_sim/{method}/{saving_path}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )

    ### Save corrected metrics of an experiment in a csv file during the training process.
    csv_logger_reeval = CSVLogger(
        f"results_sim/{method}/reeval_{saving_path}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )

    print()

    NN_training_loss = []
    NN_mse_loss = []
    NN_val_loss = []
    OT_model_predictions = []
    OT_function_predictions = []

    ### Perform MAP-Elites for num_loops generations
    for i in tqdm.tqdm(range(num_iterations)):
       
        if i==0:
            print(f"\n\nBeginning with {task_name} and {num_iterations} generations with a batch of size {batch_size} at {time_format} \n")
        
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
            replay_buffer.add_transitions_2(
                fitnesses_var=fitnesses_var,
                descriptors_var=descriptors_var,
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors
            )

            # print("BEFORE GENERATIONS, number of cells occupied in the rep: ", len(repertoire.fitnesses[jnp.where(repertoire.fitnesses != float('-inf'))[0]]))

        
        # After the first generations, buffer via scoring function and container via model 
        else:
            
            # print(f'iteration {i}: Starting to predict with the trained model and add in the container')

            start_time = time.time()

            list_geno, list_fit, list_fit_var, list_bd, list_bd_var = [], [], [], [], []

            for reeval in range(nb_reeval):

                ### Producing genotypes and associated scores with the scoring function
                genotypes, fitnesses, fitnesses_var, descriptors, descriptors_var, random_key = map_elites.generate_batch(
                    repertoire=repertoire,
                    emitter_state=emitter_state,
                    random_key=random_key,
                )

                # print(f"reveal: {reeval}, \ngenotypes: ", genotypes[:5,:4])

                list_geno.append(genotypes)
                list_fit.append(fitnesses)
                list_fit_var.append(fitnesses_var)
                list_bd.append(descriptors)
                list_bd_var.append(descriptors_var)
            
            list_geno = jnp.concatenate(list_geno)
            list_fit = jnp.concatenate(list_fit)
            list_fit_var = jnp.concatenate(list_fit_var)
            list_bd = jnp.concatenate(list_bd)
            list_bd_var = jnp.concatenate(list_bd_var)

            ### Adding scores and genotypes in the buffer
            replay_buffer.add_transitions_2(
                fitnesses_var=list_fit_var,
                descriptors_var=list_bd_var,
                genotypes=list_geno,
                fitnesses=list_fit,
                descriptors=list_bd
            )

            ### Predicting scores with the model
            fitness_model, descriptor_model = map_elites.model_batch_predict(
                model_dict=model_state,
                genotypes=list_geno,
                # reevaluations=nb_reeval
            )

            repertoire, emitter_state, metrics = map_elites.update_via_model(
                    repertoire=repertoire,
                    emitter_state=emitter_state,
                    genotypes=list_geno,
                    fitnesses=fitness_model,
                    descriptors=descriptor_model,
                    extra_scores=None
            )

            # for reeval in range(nb_reeval):
            #     start_idx = reeval * batch_size
            #     end_idx = (reeval+1) * batch_size
            #     # print(start_idx, end_idx)
            #     ### Adding predictions + genotypes in the container. Also updating metrics and emitter
            #     repertoire, emitter_state, metrics = map_elites.update_via_model(
            #         repertoire=repertoire,
            #         emitter_state=emitter_state,
            #         genotypes=list_geno[start_idx:end_idx],
            #         fitnesses=fitness_model[start_idx:end_idx],
            #         descriptors=descriptor_model[start_idx:end_idx],
            #         extra_scores=None
            #     )

            # print("AFTER GENERATIONS, number of cells occupied in the rep: ", len(repertoire.fitnesses[jnp.where(repertoire.fitnesses != float('-inf'))[0]]))

            timelapse = time.time() - start_time # Warning: It includes batch addition

        # Every freq_training generations and the final one: train the model 
        if (i != 0 and i % freq_training == 0 and i - model_start_training > 0) or i == model_start_training or i == num_iterations-1:
            
            # print(f"iteration {i}: Beginning of training")

            ### Spliting the buffer intro 70% training set and 30% testing set
            training_set, validation_set, random_key = replay_buffer.split_buffer(
                random_key=random_key,
                train_ratio = train_split,
            )

            ### Training the model
            training_loss, mse_loss, model_state = map_elites.train_model_v2(
                model_dict=model_state,
                training_set=training_set,
                num_epochs=num_epochs,
                batch_size=sampling_size,
                # reevaluations=nb_reeval,
                l2_rate=l2_rate
            )
            
            ### Testing the model
            fit_mod, fit_func, BD_mod, BD_func, val_loss = map_elites.check_model(
                model_dict=model_state,
                validation_set=validation_set,
                batch_size=sampling_size, 
                # reevaluations=nb_reeval,
            )

            ### Keeping track of training and test losses
            NN_training_loss.append(training_loss)
            NN_mse_loss.append(mse_loss)
            NN_val_loss.append(val_loss)
            

        # When model is not training, use it to predict and compare its predictions with the scoring function in the future
        # to_predict generation before training, the NN uses the repertoire to predict some scores 
        # elif i != 0 and i > model_start_training and i != (freq_training - to_predict):
        elif i != 0 and i > model_start_training:
        # elif i != 0 and i > model_start_training and i != (freq_training - to_predict) and (i + to_predict) % freq_training == 0:
   
            # print(f"iteration {i}: Outside training - predict to compare scoring function & model")
            # print("genotypes: \n", genotypes)

            ### Predicting scores with the model
            fitness_model, descriptor_model = map_elites.model_batch_predict(
                model_dict=model_state,
                genotypes=list_geno,
                # reevaluations=nb_reeval
            )

            preds_model = jnp.concatenate((fitness_model.reshape((-1,1)), descriptor_model), axis=1)
            # print("preds_model: \n", preds_model)
            preds_func = jnp.concatenate((list_fit.reshape((-1,1)), list_bd), axis=1)
            # print("preds_func: \n", preds_func)

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
                scoring_fn=scoring_fn, #No sampling
                num_reevals=256
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
    env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period * nb_reeval # WITH SAMPLING
    # env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period

    # create the plots and the grid with corresponding uncorrected metric and final repertoire
    fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'results_sim/{method}/{saving_path}.png')

    # Number of evaluations performed during the whole algorithm (corrected)
    env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period * nb_reeval # WITH SAMPLING
    # env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period # equal

    # create the plots and the grid with corresponding orrected metric and final repertoire
    fig_reeval, axes_reeval = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics_reeval, repertoire=reeval_repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'results_sim/{method}/reeval_{saving_path}.png')

    ### Adjusting elements
    NN_training_loss = jnp.concatenate(NN_training_loss)
    NN_mse_loss = jnp.concatenate(NN_mse_loss)
    OT_model_predictions = jnp.concatenate(OT_model_predictions) 
    OT_function_predictions = jnp.concatenate(OT_function_predictions) 

    ### Neural Network
    with open(f'results_sim/{method}/Backup/{method}_{task_name}_trained_model.pkl', 'wb') as f:
        pickle.dump(model_state, f) 
    
    ### Training & validations losses 
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_training_loss.npy', NN_training_loss)
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_mse_loss.npy', NN_mse_loss)
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_val_loss.npy', NN_val_loss)
    
    ### Predictions outside training
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_OT_model_predictions.npy', OT_model_predictions) 
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_OT_function_predictions.npy', OT_function_predictions) 
    
    ### Predictions during training
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_BD_func.npy', BD_func)
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_BD_mod.npy', BD_mod)
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_fit_func.npy', fit_func)
    jnp.save(f'results_sim/{method}/Backup/{method}_{task_name}_fit_mod.npy', fit_mod)

    ### Containers
    global_path = f"results_sim/{method}/repertoires/{task_name}/"
    final_path_unc = global_path + "uncorrected/"
    final_path_cor = global_path + "corrected/"
    repertoire.save(path=final_path_unc)
    reeval_repertoire.save(path=final_path_cor)

if __name__ == "__main__":

    test_arm(task_name="arm", batch_size=64)
    test_arm(task_name="noisy_arm", batch_size=64)
    print()