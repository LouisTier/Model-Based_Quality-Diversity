"""  
This code comes from the work provided by Louis BERTHIER as part of his Independent Study Option at Imperial College London.
@article{berthier2023iso,
  title={Model-Based Uncertainty Quantification in the context of Reinforcement Learning and Quality-Diversity},
  author={Berthier, Louis and Lim, Bryan and Flageat Manon and Cully, Antoine},
  year={2023}
}

It is based on QDax framework: 
    A. https://github.com/adaptive-intelligent-robotics/QDax/tree/main
    B. https://github.com/adaptive-intelligent-robotics/QDax/blob/main/examples/mapelites.ipynb
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}

This work has been slightly adapted and modified by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

"""Test default rastrigin using MAP Elites"""

import functools
import time
import datetime
import tqdm
import argparse

from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.arm import arm_scoring_function, noisy_arm_scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.stochasticity_utils import reevaluation_function, sampling

from annexed_methods import create_directory
import os

method = "ME"

# good_path = "results_replications" # results_replications or results_sim

# current_path = os.path.dirname(os.path.realpath(__file__))
# result_path = os.path.join(current_path, good_path)
# create_directory(result_path)
# ME_path = os.path.join(result_path, "ME")
# create_directory(ME_path)
# ME_rep_path = os.path.join(ME_path, "repertoires")
# create_directory(ME_rep_path)
# arm_path = os.path.join(ME_rep_path, "arm")
# create_directory(arm_path)
# arm_correct_path = os.path.join(arm_path, "corrected")
# create_directory(arm_correct_path)
# arm_uncorrect_path = os.path.join(arm_path, "uncorrected")
# create_directory(arm_uncorrect_path)
# noisy_arm_path = os.path.join(ME_rep_path, "noisy_arm")
# create_directory(noisy_arm_path) 
# noisy_arm_correct_path = os.path.join(noisy_arm_path, "corrected")
# create_directory(noisy_arm_correct_path)
# noisy_arm_uncorrect_path = os.path.join(noisy_arm_path, "uncorrected")
# create_directory(noisy_arm_uncorrect_path)
# metric_path = os.path.join(result_path, "Metrics_Comparison")
# create_directory(metric_path)
# container_path = os.path.join(result_path, "Containers_Comparison")
# create_directory(container_path)


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
def test_arm(task_name: str, batch_size: int, nb_seed: int, save_dir: str, time_format: str) -> None:
    """
    Run ME algorithm. 

    Args:
        - task_name: Name of the task, so arm or noisy_arm
        - batch_size: Number of elements to be evaluated in parallel
        
    Returns:
        - Nothing but saves the corresponding csv metrics and png 
    """

    ### Randomness of the simulation (numpy)
    seed = nb_seed

    print("seed: ", seed)

    ### Number of DoF of the arm. 8 is an easy task (Cully), 100 is medium and 1000 is hard
    ### dimension of genotype / number of angles to control
    num_param_dimensions = 8  

    ### Just a random initialization, could be w/e
    init_batch_size = batch_size

    ### Solution/ Sample considered at each forward pass (or main loop/generation) and evaluated in parallel.
    ### Each solution has its own episode and there is no interaction between them (world). Here an episode has a length of 1 (instantaneous).
    batch_size = batch_size

    ### Number of generations (= batch_size episodes). Main QD loop = selection, mutation, evaluation, addition  
    num_iterations = 3000 # 3000

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
    saving_path = "ME_" + task_name + "_" + time_format
    # print(saving_path)

    ### Init a random key to consume (Jax = local randomness) based on the global randomness (Numpy)
    random_key = jax.random.PRNGKey(seed)

    # Init population of controllers
    ### This is the genotype used to initialized MAP-Elites
    random_key, subkey = jax.random.split(random_key)
    init_variables = jax.random.uniform(
        subkey, ### Handling Jax randomness
        shape=(init_batch_size, num_param_dimensions), ### Population at each generation
        minval=min_param, ### Minimum value of the genotype parameters
        maxval=max_param, ### Maximum value of the genotype parameters
    )

    # Prepare the scoring function with the corresponding task (arm or noisy_arm)
    ### Value of fitness should be between - 0.1 and 0
    scoring_fn = scoring_functions[task_name]

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

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn, ### How to evaluate the env. The score is defined intrinsically  by the environment selected (the task)
        emitter=mixing_emitter, ### How to evole the population at each mutation step
        metrics_function=metrics_fn, ### How to evaluate the algorithm (only for the user)
    )

    # Compute the centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape, ### Number of centroids per BD dimension
        minval=min_bd, ### Minimum descriptors value
        maxval=max_bd, ### Maximum descriptors value
    ) ### Returns centroids with shape (num_centroids, num_descriptors)

    # Compute initial repertoire
    ### Can we see the emitter_state as the state of the controllers?
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
      
    ### Why defining a log period instead of directly defining the num_loops ? Mainly for speed reason
    log_period = 10 # one scan update of 10 generations
    num_loops = int(num_iterations / log_period) # 1000 / 10

    ### Dictionnary to store the different metrics of the algorithm
    all_metrics = {}
    all_metrics_reeval = {}
    os.makedirs(f"{save_dir}/{method}", exist_ok=True)

    ### Save metrics of an experiment in a csv file during the training process.
    csv_logger = CSVLogger(
        f"{save_dir}/ME/{saving_path}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )

    csv_logger_reeval = CSVLogger(
        f"{save_dir}/ME/reeval_{saving_path}.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )

    print()
    ### Perform MAP-Elites for num_loops generations
    for i in tqdm.tqdm(range(num_loops)):
        
        if i==0:
            print(f"\n\nBeginning with {task_name} and {num_iterations} generations with a batch of size {batch_size} at {time_format} \n")
        
        # if (i%10 == 0) or (i == num_loops-1):
        #     print(f'Loop: {i} | Iteration: {i*log_period}/{num_iterations}')

        ### Keeping track of the time required for each generation
        start_time = time.time()

        ### MAIN ITERATION
        ### jax.lax.scan will scan the 'map_elites_scan_update' function over the leading array axes.
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            map_elites.scan_update, ### function that will be scanned
            (repertoire, emitter_state, random_key), ### variables used for the initialization
            (), ### array type returned after the function ==> tuple
            length=log_period, ### Number of 'scans' so number of repetitions of the function (loop iterations)
        )

        ### Keeping track of the time required for each generation
        timelapse = time.time() - start_time



        ### Keeping track of the time required for each generation
        start_time_reeval = time.time()
        
        (reeval_repertoire, _, _, _, _, random_key) = reevaluation_function(repertoire=repertoire,
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
        logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": 1 + i*log_period}

        ### For all metrics ==> what are the metrics (and the order) ? It is "loop", "iteration", "qd_score", "max_fitness", "coverage", "time"
        for key, value in metrics.items():

            ### Take last value of corresponding metric
            logged_metrics[key] = value[-1]

            # take all values

            ### Concatenate current value with previous ones
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])

            ### If this is the first value of the corresponding metric, no need ot concatenate
            else:
                all_metrics[key] = value

        ### Logging/writing new metrics to the csv file every 10 generations (log_period).
        csv_logger.log(logged_metrics)



        # log metrics
        logged_metrics_reeval = {"time": timelapse_reeval, "loop": 1+i, "iteration": 1 + i*log_period}

        # print("items: ",reeval_metrics.items())

        ## For all metrics ==> what are the metrics (and the order) ? It is "loop", "iteration", "qd_score", "max_fitness", "coverage", "time"
        for key, value in reeval_metrics.items():

            # print("key reeval: ", key, "value reeval: ", value)
            ### Take last value of corresponding metric
            logged_metrics_reeval[key] = value
            # print("value: ", logged_metrics_reeval[key])

            # print(type(logged_metrics_reeval[key]))

            # take all values

            ### Concatenate current value with previous ones
            if key in all_metrics_reeval.keys():

                # print("concatenate reeval: ", all_metrics_reeval[key], "and ", value)
                # all_metrics_reeval[key] = jnp.concatenate([all_metrics_reeval[key], value])
                all_metrics_reeval[key] = jnp.append(all_metrics_reeval[key], value)
                # all_metrics_reeval[key] = jnp.append([all_metrics_reeval[key], value])

            ### If this is the first value of the corresponding metric, no need ot concatenate
            else:
                all_metrics_reeval[key] = value

        csv_logger_reeval.log(logged_metrics_reeval)


        
    # create the x-axis array
    ### num_iterations = 1000 generations, 
    ### batch_size = 100 solutions with their own episode for each generation to be evaluated, 
    ### episode_length = 100 timesteps in one evaluation (one offspring evaluated) so for each solution
    ### No episode ==> Toy task, easy, instantaneous. Not an RL problem, can't apply RL algo here, NO MDP here for instance.
    env_steps = jnp.arange(num_iterations) * batch_size 

    # create the plots and the grid with corresponding metric and final repertoire
    fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'{save_dir}/ME/{saving_path}.png')


    # env_steps = jnp.arange(num_iterations/log_period) * batch_size # factor 10 between uncorrected and here (corrected)
    env_steps = jnp.arange(num_iterations/log_period) * batch_size * log_period # equal
    # env_steps = jnp.arange(num_iterations) * batch_size

    # print("metrics_reeval: ", all_metrics_reeval)

    # create the plots and the grid with corresponding metric and final repertoire
    fig_reeval, axes_reeval = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics_reeval, repertoire=reeval_repertoire, min_bd=min_bd, max_bd=max_bd)

    ### Saving corresponding figure (3 metrics + final archive)
    plt.savefig(f'{save_dir}/ME/reeval_{saving_path}.png')

    
    # global_path = f"{good_path}/ME/repertoires/"
    # # saving_rep_unc = f"results_sim/ME/repertoires/{task_name}_unc_rep"
    # # saving_rep_cor = f"results_sim/ME/repertoires/{task_name}_cor_rep"

    # if task_name == "arm":
    #     final_path = global_path + f"{task_name}/"
    #     final_path_unc = final_path + "uncorrected/"
    #     final_path_cor = final_path + "corrected/"
    #     # print(final_path)
    #     repertoire.save(path=final_path_unc)
    #     reeval_repertoire.save(path=final_path_cor)
    # else:
    #     final_path = global_path + f"{task_name}/"
    #     final_path_unc = final_path + "uncorrected/"
    #     final_path_cor = final_path + "corrected/"
    #     # print(final_path)
    #     repertoire.save(path=final_path_unc)
    #     reeval_repertoire.save(path=final_path_cor)

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
    parser.add_argument("--batch_size", default=512, type=int, help="Number of solutions to be evaluated at each generation")

    # To specify
    parser.add_argument("--save_dir", default="", type=str, help="Directory to save the results")
    parser.add_argument("--task_name", default="", type=str, help="Name of the task/environment")
    parser.add_argument("--seed", default=42, type=int, help="Number of the seed used to fix randomness")

    return parser.parse_args()

if __name__ == "__main__":

    # test_arm(task_name="arm", batch_size=512)
    # test_arm(task_name="noisy_arm", batch_size=512)

    # print()

    #################
    #    Classic    #
    #################

    current_time = datetime.datetime.now()
    time_format = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    # Read and retrieve input parameter from command line
    args = specify_arg()

    os.makedirs(f'{args.save_dir}/{method}', exist_ok=True) #save dir = $CURPATH/results/$PATHNAME where PATHNAME = 2023-07-25_10_30_45_12345    

    start_time = time.time()
    test_arm(
        task_name=args.task_name, 
        batch_size=args.batch_size,
        nb_seed=args.seed,
        save_dir=args.save_dir,
        time_format=time_format,
    )
    print()    