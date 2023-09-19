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
from annexed_methods import plot_repertoires, plot_repertoires_solo, plot_repertoires_solo_UQ, filter_repertoire
import time


###############################
#       INITIALIZATION        #
###############################

# os.makedirs("results_sim_arg/Containers_Comparison", exist_ok=True)
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
path_Imp = "results_sim_arg/MEMB_Implicit"
path_Exp = "results_sim_arg/MEMB_Explicit_Naive"
path_ImpUQ = "results_sim_arg/MEMBUQ_Implicit"
path_ExpUQ = "results_sim_arg/MEMBUQ_NLL_Explicit_Naive"
path_ImpW = "results_sim_arg/MEMB_Implicit_Wipe"
path_ExpW = "results_sim_arg/MEMB_Explicit_Naive_Wipe"
path_ImpUQW = "results_sim_arg/MEMBUQ_Implicit_Wipe"
path_ExpUQW = "results_sim_arg/MEMBUQ_NLL_Explicit_Naive_Wipe"

# True = Analysis with ME, MES, MBME and MBMEUQ and False = Analysis with MBME and MBMEUQ
multi_analysis = True

# True = 2x3 figures and False = 1x6 figures. Let it always TRUE
report_analysis = True

if report_analysis == True:
  fig_size_width = 22.5 # 18 for 2x3, 22.5 for 3x4
  fig_size_height = 15.5 # 11 for 2x3, 15.5 for 3x4
else:
  fig_size_width = 43 # 24 for 1x4, 43 for 1x6
  fig_size_height = 7 # 6 for 1x4, 7 for 1x6

fig_size_width_UQ = 12 # 12 for 2x2
fig_size_height_UQ = 11 # 10 for 2x2

if multi_analysis == True:
  
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

  MEMB_implicit_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{a}/{cor}/")
  MEMB_implicit_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{a}/{unc}/")
  MEMB_implicit_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{na}/{cor}/")
  MEMB_implicit_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{na}/{unc}/")

  MEMB_explicit_naive_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{a}/{cor}/")
  MEMB_explicit_naive_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{a}/{unc}/")
  MEMB_explicit_naive_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{na}/{cor}/")
  MEMB_explicit_naive_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{na}/{unc}/")

  MEMBUQ_implicit_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{a}/{cor}/")
  MEMBUQ_implicit_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{a}/{unc}/")
  MEMBUQ_implicit_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{na}/{cor}/")
  MEMBUQ_implicit_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{na}/{unc}/")

  MEMBUQ_explicit_naive_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{a}/{cor}/")
  MEMBUQ_explicit_naive_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{a}/{unc}/")
  MEMBUQ_explicit_naive_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{na}/{cor}/")
  MEMBUQ_explicit_naive_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{na}/{unc}/")

  MEMB_implicit_wipe_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpW}/repertoires/{a}/{cor}/")
  MEMB_implicit_wipe_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpW}/repertoires/{a}/{unc}/")
  MEMB_implicit_wipe_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpW}/repertoires/{na}/{cor}/")
  MEMB_implicit_wipe_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpW}/repertoires/{na}/{unc}/")

  MEMB_explicit_naive_wipe_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpW}/repertoires/{a}/{cor}/")
  MEMB_explicit_naive_wipe_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpW}/repertoires/{a}/{unc}/")
  MEMB_explicit_naive_wipe_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpW}/repertoires/{na}/{cor}/")
  MEMB_explicit_naive_wipe_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpW}/repertoires/{na}/{unc}/")

  MEMBUQ_implicit_wipe_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQW}/repertoires/{a}/{cor}/")
  MEMBUQ_implicit_wipe_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQW}/repertoires/{a}/{unc}/")
  MEMBUQ_implicit_wipe_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQW}/repertoires/{na}/{cor}/")
  MEMBUQ_implicit_wipe_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQW}/repertoires/{na}/{unc}/")

  MEMBUQ_explicit_naive_wipe_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{a}/{cor}/")
  MEMBUQ_explicit_naive_wipe_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{a}/{unc}/")
  MEMBUQ_explicit_naive_wipe_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{na}/{cor}/")
  MEMBUQ_explicit_naive_wipe_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQW}/repertoires/{na}/{unc}/")



  ###############################
  #       MERGING & NAMING      #
  ###############################

  ### Grouping and naming experiments that will be compared 
  determ_unc_rep = [
    ME_arm_unc_repertoire, MES_arm_unc_repertoire, 
    MEMB_explicit_naive_arm_unc_repertoire, MEMB_explicit_naive_wipe_arm_unc_repertoire, MEMBUQ_explicit_naive_arm_unc_repertoire, MEMBUQ_explicit_naive_wipe_arm_unc_repertoire,
    MEMB_implicit_arm_unc_repertoire, MEMB_implicit_wipe_arm_unc_repertoire, MEMBUQ_implicit_arm_unc_repertoire, MEMBUQ_implicit_wipe_arm_unc_repertoire 
  ]
  determ_unc_names = [
      "Uncorrected ME \n Deterministic Arm Repertoire",
      "Uncorrected MES \n Deterministic Arm Repertoire",
      "Uncorrected MBME Explicit \n Deterministic Arm Repertoire",
      "Uncorrected MBME Explicit Wipe \n Deterministic Arm Repertoire",
      "Uncorrected MBMEUQ Explicit \n Deterministic Arm Repertoire",
      "Uncorrected MBMEUQ Explicit Wipe \n Deterministic Arm Repertoire",
      "Uncorrected MBME Implicit \n Deterministic Arm Repertoire",
      "Uncorrected MBME Implicit Wipe \n Deterministic Arm Repertoire",
      "Uncorrected MBMEUQ Implicit \n Deterministic Arm Repertoire",
      "Uncorrected MBMEUQ Implicit Wipe \n Deterministic Arm Repertoire"
  ]

  determ_cor_rep = [
    ME_arm_cor_repertoire, MES_arm_cor_repertoire, 
    MEMB_explicit_naive_arm_cor_repertoire, MEMB_explicit_naive_wipe_arm_cor_repertoire, MEMBUQ_explicit_naive_arm_cor_repertoire, MEMBUQ_explicit_naive_wipe_arm_cor_repertoire,
    MEMB_implicit_arm_cor_repertoire, MEMB_implicit_wipe_arm_cor_repertoire, MEMBUQ_implicit_arm_cor_repertoire, MEMBUQ_implicit_wipe_arm_cor_repertoire
  ]
  determ_cor_names = [
      "Corrected ME \n Deterministic Arm Repertoire",
      "Corrected MES \n Deterministic Arm Repertoire",
      "Corrected MBME Explicit \n Deterministic Arm Repertoire",
      "Corrected MBME Explicit Wipe \n Deterministic Arm Repertoire",
      "Corrected MBMEUQ Explicit \n Deterministic Arm Repertoire",
      "Corrected MBMEUQ Explicit Wipe \n Deterministic Arm Repertoire",
      "Corrected MBME Implicit \n Deterministic Arm Repertoire",
      "Corrected MBME Implicit Wipe \n Deterministic Arm Repertoire",
      "Corrected MBMEUQ Implicit \n Deterministic Arm Repertoire",
      "Corrected MBMEUQ Implicit Wipe \n Deterministic Arm Repertoire"
  ]


  uncert_unc_rep = [
    ME_noisy_arm_unc_repertoire, MES_noisy_arm_unc_repertoire, 
    MEMB_explicit_naive_noisy_arm_unc_repertoire, MEMB_explicit_naive_wipe_noisy_arm_unc_repertoire, MEMBUQ_explicit_naive_noisy_arm_unc_repertoire, MEMBUQ_explicit_naive_wipe_noisy_arm_unc_repertoire,
    MEMB_implicit_noisy_arm_unc_repertoire, MEMB_implicit_wipe_noisy_arm_unc_repertoire, MEMBUQ_implicit_noisy_arm_unc_repertoire, MEMBUQ_implicit_wipe_noisy_arm_unc_repertoire 
  ]
  uncert_unc_names = [
      "Uncorrected ME \n Uncertain Repertoire",
      "Uncorrected MES \n Uncertain Repertoire",
      "Uncorrected MBME Explicit \n Uncertain Repertoire",
      "Uncorrected MBME Explicit Wipe \n Uncertain Repertoire",
      "Uncorrected MBMEUQ Explicit \n Uncertain Repertoire",
      "Uncorrected MBMEUQ Explicit Wipe \n Uncertain Repertoire",
      "Uncorrected MBME Implicit \n Uncertain Repertoire",
      "Uncorrected MBME Implicit Wipe \n Uncertain Repertoire",
      "Uncorrected MBMEUQ Implicit \n Uncertain Repertoire",
      "Uncorrected MBMEUQ Implicit Wipe \n Uncertain Repertoire"      
  ]
    
  uncert_cor_rep = [
    ME_noisy_arm_cor_repertoire, MES_noisy_arm_cor_repertoire, 
    MEMB_explicit_naive_noisy_arm_cor_repertoire, MEMB_explicit_naive_wipe_noisy_arm_cor_repertoire, MEMBUQ_explicit_naive_noisy_arm_cor_repertoire, MEMBUQ_explicit_naive_wipe_noisy_arm_cor_repertoire,
    MEMB_implicit_noisy_arm_cor_repertoire, MEMB_implicit_wipe_noisy_arm_cor_repertoire, MEMBUQ_implicit_noisy_arm_cor_repertoire, MEMBUQ_implicit_wipe_noisy_arm_cor_repertoire    , 
  ]
  uncert_cor_names = [
      "Corrected ME \n Uncertain Repertoire",
      "Corrected MES \n Uncertain Repertoire",
      "Corrected MBME Explicit \n Uncertain Repertoire",
      "Corrected MBME Explicit Wipe \n Uncertain Repertoire",
      "Corrected MBMEUQ Explicit \n Uncertain Repertoire",
      "Corrected MBMEUQ Explicit Wipe \n Uncertain Repertoire",
      "Corrected MBME Implicit \n Uncertain Repertoire",
      "Corrected MBME Implicit Wipe \n Uncertain Repertoire",
      "Corrected MBMEUQ Implicit \n Uncertain Repertoire",
      "Corrected MBMEUQ Implicit Wipe \n Uncertain Repertoire"      
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
  determ_unc_path = 'results_sim_arg/Containers_Comparison/Deterministic_Uncorrected_Repertoires_Plots.png'

  print("\nPlotting uncorrected deterministic repertoires\n")
  start_time = time.time()
  plot_repertoires_solo(
    fig=fig_determ_unc, 
    repertoires_fig=determ_unc_rep, 
    repertoires_names=determ_unc_names, 
    min_bd=min_bd, 
    max_bd=max_bd, 
    min_fit=None, # min_determ_unc | None
    max_fit=None, # max_determ_unc | None
    report_display=report_analysis,
    path=determ_unc_path
  )
  timelapse = time.time() - start_time 
  print(f"It took {timelapse/60:.2f} minutes to plot uncorrected deterministic repertoires") # 3.48 minutes

  fig_determ_cor = plt.figure(figsize=(fig_size_width, fig_size_height))
  determ_cor_path = 'results_sim_arg/Containers_Comparison/Deterministic_Corrected_Repertoires_Plots.png'

  print("\nPlotting corrected deterministic repertoires\n")
  plot_repertoires_solo(
    fig=fig_determ_cor, 
    repertoires_fig=determ_cor_rep, 
    repertoires_names=determ_cor_names, 
    min_bd=min_bd, 
    max_bd=max_bd, 
    min_fit=min_determ_cor,
    max_fit=max_determ_cor,
    report_display=report_analysis,
    path=determ_cor_path
  )



  ###############################
  #         UNCERTAIN           #
  ###############################

  fig_uncert_unc = plt.figure(figsize=(fig_size_width, fig_size_height))
  uncert_unc_path = 'results_sim_arg/Containers_Comparison/Uncertain_Uncorrected_Repertoires_Plots.png'

  print("\nPlotting uncorrected uncertain repertoires\n")
  plot_repertoires_solo(
    fig=fig_uncert_unc, 
    repertoires_fig=uncert_unc_rep, 
    repertoires_names=uncert_unc_names, 
    min_bd=min_bd, 
    max_bd=max_bd, 
    min_fit=None, # min_uncert_unc | None
    max_fit=None, # max_uncert_unc | None
    report_display=report_analysis,
    path=uncert_unc_path
  )

  fig_uncert_cor = plt.figure(figsize=(fig_size_width, fig_size_height))
  uncert_cor_path = 'results_sim_arg/Containers_Comparison/Uncertain_Corrected_Repertoires_Plots.png'

  print("\nPlotting corrected uncertain repertoires\n")
  plot_repertoires_solo(
    fig=fig_uncert_cor, 
    repertoires_fig=uncert_cor_rep, 
    repertoires_names=uncert_cor_names, 
    min_bd=min_bd, 
    max_bd=max_bd,
    min_fit=min_uncert_cor,
    max_fit=max_uncert_cor, 
    report_display=report_analysis,
    path=uncert_cor_path
  )

else:
   
  ###############################
  #     LOADING REPERTOIRES     #
  ###############################

  ### Loading repertoires of corresponding methods, tasks and metrics
  MEMB_implicit_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{a}/{cor}/")
  MEMB_implicit_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{a}/{unc}/")
  MEMB_implicit_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{na}/{cor}/")
  MEMB_implicit_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Imp}/repertoires/{na}/{unc}/")

  MEMB_explicit_naive_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{a}/{cor}/")
  MEMB_explicit_naive_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{a}/{unc}/")
  MEMB_explicit_naive_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{na}/{cor}/")
  MEMB_explicit_naive_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_Exp}/repertoires/{na}/{unc}/")

  MEMBUQ_implicit_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{a}/{cor}/")
  MEMBUQ_implicit_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{a}/{unc}/")
  MEMBUQ_implicit_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{na}/{cor}/")
  MEMBUQ_implicit_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ImpUQ}/repertoires/{na}/{unc}/")

  MEMBUQ_explicit_naive_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{a}/{cor}/")
  MEMBUQ_explicit_naive_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{a}/{unc}/")
  MEMBUQ_explicit_naive_noisy_arm_cor_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{na}/{cor}/")
  MEMBUQ_explicit_naive_noisy_arm_unc_repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path_ExpUQ}/repertoires/{na}/{unc}/")



  ###############################
  #       MERGING & NAMING      #
  ###############################

  ### Grouping and naming experiments that will be compared 
  determ_unc_rep = [MEMB_implicit_arm_unc_repertoire, MEMB_explicit_naive_arm_unc_repertoire, MEMBUQ_implicit_arm_unc_repertoire, MEMBUQ_explicit_naive_arm_unc_repertoire]
  determ_unc_names = [
      "Uncorrected MBME Implicit Repertoire \n Deterministic Arm - F5.1",
      "Uncorrected MBME Explicit Repertoire \n Deterministic Arm - F5.2",
      "Uncorrected MBMEUQ Implicit Repertoire \n Deterministic Arm - F5.3",
      "Uncorrected MBMEUQ Explicit Repertoire \n Deterministic Arm - F5.4"
  ]

  determ_cor_rep = [MEMB_implicit_arm_cor_repertoire, MEMB_explicit_naive_arm_cor_repertoire, MEMBUQ_implicit_arm_cor_repertoire, MEMBUQ_explicit_naive_arm_cor_repertoire]
  determ_cor_names = [
      "Corrected MBME Implicit Repertoire \n Deterministic Arm - F5.5",
      "Corrected MBME Explicit Repertoire \n Deterministic Arm - F5.6",
      "Corrected MBMEUQ Implicit Repertoire \n Deterministic Arm - F5.7",
      "Corrected MBMEUQ Explicit Repertoire \n Deterministic Arm - F5.8"
  ]


  uncert_unc_rep = [MEMB_implicit_noisy_arm_unc_repertoire, MEMB_explicit_naive_noisy_arm_unc_repertoire, MEMBUQ_implicit_noisy_arm_unc_repertoire, MEMBUQ_explicit_naive_noisy_arm_unc_repertoire]
  uncert_unc_names = [
      "Uncorrected MBME Implicit Repertoire \n Noisy Arm - F6.1",
      "Uncorrected MBME Explicit Repertoire \n Noisy Arm - F6.2",
      "Uncorrected MBMEUQ Implicit Repertoire \n Noisy Arm - F6.3",
      "Uncorrected MBMEUQ Explicit Repertoire \n Noisy Arm - F6.4"
  ]
    
  uncert_cor_rep = [MEMB_implicit_noisy_arm_cor_repertoire, MEMB_explicit_naive_noisy_arm_cor_repertoire, MEMBUQ_implicit_noisy_arm_cor_repertoire, MEMBUQ_explicit_naive_noisy_arm_cor_repertoire]
  uncert_cor_names = [
      "Corrected MBME Implicit Repertoire \n Noisy Arm - F6.5",
      "Corrected MBME Explicit Repertoire \n Noisy Arm - F6.6",
      "Corrected MBMEUQ Implicit Repertoire \n Noisy Arm - F6.7",
      "Corrected MBMEUQ Explicit Repertoire \n Noisy Arm - F6.8"
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

  fig_determ_unc = plt.figure(figsize=(fig_size_width_UQ, fig_size_height_UQ))
  determ_unc_path = 'results_sim_arg/Containers_Comparison_UQ/Deterministic_Uncorrected_Repertoires_UQ_Plots.png'

  print("\nPlotting UQ uncorrected deterministic repertoires\n")
  start_time = time.time()
  plot_repertoires_solo_UQ(
    fig = fig_determ_unc,
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

  fig_determ_cor = plt.figure(figsize=(fig_size_width_UQ, fig_size_height_UQ))
  determ_cor_path = 'results_sim_arg/Containers_Comparison_UQ/Deterministic_Corrected_Repertoires_UQ_Plots.png'

  print("\nPlotting UQ corrected deterministic repertoires\n")
  plot_repertoires_solo_UQ(
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

  fig_uncert_unc = plt.figure(figsize=(fig_size_width_UQ, fig_size_height_UQ))
  uncert_unc_path = 'results_sim_arg/Containers_Comparison_UQ/Uncertain_Uncorrected_Repertoires_UQ_Plots.png'

  print("\nPlotting UQ uncorrected uncertain repertoires\n")
  plot_repertoires_solo_UQ(
    fig=fig_uncert_unc, 
    repertoires_fig=uncert_unc_rep, 
    repertoires_names=uncert_unc_names, 
    min_bd=min_bd, 
    max_bd=max_bd, 
    min_fit=None, # min_uncert_unc | None
    max_fit=None, # max_uncert_unc | None
    path=uncert_unc_path
  )

  fig_uncert_cor = plt.figure(figsize=(fig_size_width_UQ, fig_size_height_UQ))
  uncert_cor_path = 'results_sim_arg/Containers_Comparison_UQ/Uncertain_Corrected_Repertoires_UQ_Plots.png'

  print("\nPlotting UQ corrected uncertain repertoires\n")
  plot_repertoires_solo_UQ(
    fig=fig_uncert_cor, 
    repertoires_fig=uncert_cor_rep, 
    repertoires_names=uncert_cor_names, 
    min_bd=min_bd, 
    max_bd=max_bd,
    min_fit=min_uncert_cor,
    max_fit=max_uncert_cor, 
    path=uncert_cor_path
  )
