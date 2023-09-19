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

This work has been adapted and modified by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import os
from matplotlib import pyplot as plt
import jax
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from annexed_methods import plot_repertoires, plot_repertoires_solo, plot_repertoires_solo_UQ, filter_repertoire, plot_repertoires_solo_best
import time


###############################
#       INITIALIZATION        #
###############################

os.makedirs("results_sim_arg/Containers_Comparison/Best_Case", exist_ok=True)
# os.makedirs("results_sim_arg/Containers_Comparison_UQ", exist_ok=True)

### Fix randomness of the simulation (Numpy = global randomness)
seed = 42

### Init a random key that will be consumed (Jax = local randomness) 
random_key = jax.random.PRNGKey(seed)
random_key, subkey = jax.random.split(random_key)

### Parameters of initizaliation
init_batch_size = 512
num_param_dimensions = 8
min_param = 0.0
max_param = 1.0
min_bd = 0.0
max_bd = 1.0

### Genotype used to initialized MAP-Elites, first individual/controller
init_variables = jax.random.uniform(
    subkey, ### Handling Jax randomness
    shape=(1, num_param_dimensions), ### Population at each generation
    minval=min_param, ### Minimum value of the genotype parameters
    maxval=max_param, ### Maximum value of the genotype parameters
)

### Flatten the init_variables array into a single flat array
_, reconstruction_fn = ravel_pytree(init_variables)

a = "arm"
na = "noisy_arm"
unc = "uncorrected"
cor = "corrected"

path_ME = "results_sim_arg/ME"
path_MES = "results_sim_arg/MES"
path_ExpUQW = "results_sim_arg/MEMBUQ_NLL_Explicit_Naive_Wipe"

# True = 2x3 figures and False = 1x6 figures. Let it always TRUE
report_analysis = True

if report_analysis == True:
  fig_size_width = 16 # 18 for 2x3, 22.5 for 3x4
  fig_size_height = 5 # 11 for 2x3, 15.5 for 3x4
else:
  fig_size_width = 43 # 24 for 1x4, 43 for 1x6
  fig_size_height = 7 # 6 for 1x4, 7 for 1x6

###############################
#     LOADING REPERTOIRES     #
###############################

### Loading repertoires of corresponding methods, tasks and metrics
ME_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ME}/repertoires/{a}/{cor}/")
ME_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ME}/repertoires/{a}/{unc}/")
ME_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ME}/repertoires/{na}/{cor}/")
ME_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ME}/repertoires/{na}/{unc}/")

MES_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_MES}/repertoires/{a}/{cor}/")
MES_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_MES}/repertoires/{a}/{unc}/")
MES_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_MES}/repertoires/{na}/{cor}/")
MES_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_MES}/repertoires/{na}/{unc}/")

MEMBUQ_explicit_naive_wipe_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{a}/{cor}/")
MEMBUQ_explicit_naive_wipe_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{a}/{unc}/")
MEMBUQ_explicit_naive_wipe_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{na}/{cor}/")
MEMBUQ_explicit_naive_wipe_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{na}/{unc}/")



###############################
#       MERGING & NAMING      #
###############################

### Grouping and naming experiments that will be compared 
determ_unc_rep = [
  ME_arm_unc_repertoire, MES_arm_unc_repertoire, MEMBUQ_explicit_naive_wipe_arm_unc_repertoire
]
determ_unc_names = [
    "Uncorrected ME \n Deterministic Arm Repertoire",
    "Uncorrected MES \n Deterministic Arm Repertoire",
    "Uncorrected MBMEUQ Explicit Wipe \n Deterministic Arm Repertoire"
]

determ_cor_rep = [
  ME_arm_cor_repertoire, MES_arm_cor_repertoire, MEMBUQ_explicit_naive_wipe_arm_cor_repertoire
]
determ_cor_names = [
    "Corrected ME \n Deterministic Arm Repertoire",
    "Corrected MES \n Deterministic Arm Repertoire",
    "Corrected MBMEUQ Explicit Wipe \n Deterministic Arm Repertoire"
]


uncert_unc_rep = [
  ME_noisy_arm_unc_repertoire, MES_noisy_arm_unc_repertoire, MEMBUQ_explicit_naive_wipe_noisy_arm_unc_repertoire
]
uncert_unc_names = [
    "Uncorrected ME \n Uncertain Repertoire",
    "Uncorrected MES \n Uncertain Repertoire",
    "Uncorrected MBMEUQ Explicit Wipe \n Uncertain Repertoire" 
]
  
uncert_cor_rep = [
  ME_noisy_arm_cor_repertoire, MES_noisy_arm_cor_repertoire, MEMBUQ_explicit_naive_wipe_noisy_arm_cor_repertoire
]
uncert_cor_names = [
    "Corrected ME \n Uncertain Repertoire",
    "Corrected MES \n Uncertain Repertoire",
    "Corrected MBMEUQ Explicit Wipe \n Uncertain Repertoire",   
]


###############################
#          MIN & MAX          #
###############################

all_reps = [determ_unc_rep, determ_cor_rep, uncert_unc_rep, uncert_cor_rep]
min_values = []
max_values = []

# start_time = time.time()
for counter, repertoires in enumerate(all_reps):

  print(f"Finding Min and Max of list of repertoires number {counter}")

  filtered_reps = []
  min_list = []
  max_list = []

  for rep in repertoires:
      filtered_fitnesses = filter_repertoire(rep.fitnesses)
      filtered_reps.append(filtered_fitnesses)

  for fit_list in filtered_reps:
      min_list.append(min(fit_list))
      max_list.append(max(fit_list))

  min_of_min = min(min_list)
  max_of_max = max(max_list)
  # print("min_list: ", min_list)
  # print("max_list: ", max_list)
  # print("min_of_min: ", min_of_min)
  # print("max_of_max: ", max_of_max)
  min_values.append(min_of_min)
  max_values.append(max_of_max)
# timelapse = time.time() - start_time 
# print(f"It took {timelapse/60:.2f} minutes to find the overall minimum and maximum") # 2.84 minutes

# print("min_values: ", min_values)
# print("max_values: ", max_values)
min_determ_unc = min_values[0]
min_determ_cor = min_values[1]
min_uncert_unc = min_values[2]
min_uncert_cor = min_values[3]
max_determ_unc = max_values[0]
max_determ_cor = max_values[1]
max_uncert_unc = max_values[2]
max_uncert_cor = max_values[3]



##############################
#       DETERMINISTIC        #
##############################

fig_determ_unc = plt.figure(figsize=(fig_size_width, fig_size_height))
determ_unc_path = 'results_sim_arg/Containers_Comparison/Best_Case/Best_Case_Deterministic_Uncorrected_Repertoires_Plots.png'

print("\nPlotting uncorrected deterministic repertoires\n")
start_time = time.time()
plot_repertoires_solo_best(
  fig=fig_determ_unc, 
  repertoires_fig=determ_unc_rep, 
  repertoires_names=determ_unc_names, 
  min_bd=min_bd, 
  max_bd=max_bd, 
  min_fit=None, # min_determ_unc | None
  max_fit=None, # max_determ_unc | None
  path=determ_unc_path
)
timelapse = time.time() - start_time 
print(f"It took {timelapse/60:.2f} minutes to plot uncorrected deterministic repertoires") # 3.48 minutes

fig_determ_cor = plt.figure(figsize=(fig_size_width, fig_size_height))
determ_cor_path = 'results_sim_arg/Containers_Comparison/Best_Case/Best_Case_Deterministic_Corrected_Repertoires_Plots.png'

print("\nPlotting corrected deterministic repertoires\n")
plot_repertoires_solo_best(
  fig=fig_determ_cor, 
  repertoires_fig=determ_cor_rep, 
  repertoires_names=determ_cor_names, 
  min_bd=min_bd, 
  max_bd=max_bd, 
  min_fit=min_determ_cor,
  max_fit=max_determ_cor,
  path=determ_cor_path
)



###############################
#         UNCERTAIN           #
###############################

fig_uncert_unc = plt.figure(figsize=(fig_size_width, fig_size_height))
uncert_unc_path = 'results_sim_arg/Containers_Comparison/Best_Case/Best_Case_Uncertain_Uncorrected_Repertoires_Plots.png'

print("\nPlotting uncorrected uncertain repertoires\n")
plot_repertoires_solo_best(
  fig=fig_uncert_unc, 
  repertoires_fig=uncert_unc_rep, 
  repertoires_names=uncert_unc_names, 
  min_bd=min_bd, 
  max_bd=max_bd, 
  min_fit=None, # min_uncert_unc | None
  max_fit=None, # max_uncert_unc | None
  path=uncert_unc_path
)

fig_uncert_cor = plt.figure(figsize=(fig_size_width, fig_size_height))
uncert_cor_path = 'results_sim_arg/Containers_Comparison/Best_Case/Best_Case_Uncertain_Corrected_Repertoires_Plots.png'

print("\nPlotting corrected uncertain repertoires\n")
plot_repertoires_solo_best(
  fig=fig_uncert_cor, 
  repertoires_fig=uncert_cor_rep, 
  repertoires_names=uncert_cor_names, 
  min_bd=min_bd, 
  max_bd=max_bd,
  min_fit=min_uncert_cor,
  max_fit=max_uncert_cor, 
  path=uncert_cor_path
)
