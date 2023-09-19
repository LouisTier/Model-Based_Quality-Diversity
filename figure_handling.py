"""
This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from annexed_methods import create_directory, retrieve_png, add_path_to_images


###############################
#         DIRECTORIES         #
###############################

method = "MEMB_Implicit" # MEMB_Implicit | MEMB_Explicit_Naive | MEMB_old

# # To create directories or ensure their existence
# current_path = os.path.dirname(os.path.realpath(__file__))
# result_path = os.path.join(current_path, "results_sim_arg")
# create_directory(result_path)
# figures_path = os.path.join(result_path, "Grouping_Figures")
# create_directory(figures_path)
# method_path = os.path.join(figures_path, method)
# create_directory(method_path)
# loss_fig_path = os.path.join(method_path, "Loss")
# create_directory(loss_fig_path)
# bd_fig_path = os.path.join(method_path, "BD")
# create_directory(bd_fig_path)
# fitness_fig_path = os.path.join(method_path, "Fitness")
# create_directory(fitness_fig_path)
# viobox_fig_path = os.path.join(method_path, "Viobox")
# create_directory(viobox_fig_path)
# arm_viobox_fig_path = os.path.join(viobox_fig_path, "arm")
# create_directory(arm_viobox_fig_path)
# noisy_arm_viobox_fig_path = os.path.join(viobox_fig_path, "noisy_arm")
# create_directory(noisy_arm_viobox_fig_path)

title = ""
# select_multi = True # True or False
# if select_multi == True:
#     # title = "8x8_"
#     title = f"{method}_"
# elif select_multi == False:
#     title = ""        PATH             #
###############################

# Path of the directories where we want to extract figures
path_loss_arm = f"results_sim_arg/{method}/Losses/arm"
path_loss_noisy_arm = f"results_sim_arg/{method}/Losses/noisy_arm"

path_BD_arm_IT_NO = f"results_sim_arg/{method}/BD_Comparison/arm/In_Training/No_Outliers"
path_BD_arm_IT_O = f"results_sim_arg/{method}/BD_Comparison/arm/In_Training/Outliers"
path_BD_arm_OT_NO = f"results_sim_arg/{method}/BD_Comparison/arm/Out_Training/No_Outliers"
path_BD_arm_OT_O = f"results_sim_arg/{method}/BD_Comparison/arm/Out_Training/Outliers"
path_fitness_arm_IT_NO = f"results_sim_arg/{method}/Fitness_Comparison/arm/In_Training/No_Outliers"
path_fitness_arm_IT_O = f"results_sim_arg/{method}/Fitness_Comparison/arm/In_Training/Outliers"
path_fitness_arm_OT_NO = f"results_sim_arg/{method}/Fitness_Comparison/arm/Out_Training/No_Outliers"
path_fitness_arm_OT_O = f"results_sim_arg/{method}/Fitness_Comparison/arm/Out_Training/Outliers"
path_viobox_arm_IT_NO = f"results_sim_arg/{method}/Viobox_Comparison/arm/In_Training/No_Outliers"
path_viobox_arm_IT_O = f"results_sim_arg/{method}/Viobox_Comparison/arm/In_Training/Outliers"
path_viobox_arm_OT_NO = f"results_sim_arg/{method}/Viobox_Comparison/arm/Out_Training/No_Outliers"
path_viobox_arm_OT_O = f"results_sim_arg/{method}/Viobox_Comparison/arm/Out_Training/Outliers"

path_BD_noisy_arm_IT_NO = f"results_sim_arg/{method}/BD_Comparison/noisy_arm/In_Training/No_Outliers"
path_BD_noisy_arm_IT_O = f"results_sim_arg/{method}/BD_Comparison/noisy_arm/In_Training/Outliers"
path_BD_noisy_arm_OT_NO = f"results_sim_arg/{method}/BD_Comparison/noisy_arm/Out_Training/No_Outliers"
path_BD_noisy_arm_OT_O = f"results_sim_arg/{method}/BD_Comparison/noisy_arm/Out_Training/Outliers"
path_fitness_noisy_arm_IT_NO = f"results_sim_arg/{method}/Fitness_Comparison/noisy_arm/In_Training/No_Outliers"
path_fitness_noisy_arm_IT_O = f"results_sim_arg/{method}/Fitness_Comparison/noisy_arm/In_Training/Outliers"
path_fitness_noisy_arm_OT_NO = f"results_sim_arg/{method}/Fitness_Comparison/noisy_arm/Out_Training/No_Outliers"
path_fitness_noisy_arm_OT_O = f"results_sim_arg/{method}/Fitness_Comparison/noisy_arm/Out_Training/Outliers"
path_viobox_noisy_arm_IT_NO = f"results_sim_arg/{method}/Viobox_Comparison/noisy_arm/In_Training/No_Outliers"
path_viobox_noisy_arm_IT_O = f"results_sim_arg/{method}/Viobox_Comparison/noisy_arm/In_Training/Outliers"
path_viobox_noisy_arm_OT_NO = f"results_sim_arg/{method}/Viobox_Comparison/noisy_arm/Out_Training/No_Outliers"
path_viobox_noisy_arm_OT_O = f"results_sim_arg/{method}/Viobox_Comparison/noisy_arm/Out_Training/Outliers"


###############################
#            IMAGE            #
###############################

# Extracting the name of .png files in the directory defined by the path above
images_loss_arm = retrieve_png(directory=path_loss_arm)
images_loss_noisy_arm = retrieve_png(directory=path_loss_noisy_arm)

images_BD_arm_IT_NO = retrieve_png(directory=path_BD_arm_IT_NO)
images_BD_arm_IT_O = retrieve_png(directory=path_BD_arm_IT_O)
images_BD_arm_OT_NO = retrieve_png(directory=path_BD_arm_OT_NO)
images_BD_arm_OT_O = retrieve_png(directory=path_BD_arm_OT_O)
images_fitness_arm_IT_NO = retrieve_png(directory=path_fitness_arm_IT_NO)
images_fitness_arm_IT_O = retrieve_png(directory=path_fitness_arm_IT_O)
images_fitness_arm_OT_NO = retrieve_png(directory=path_fitness_arm_OT_NO)
images_fitness_arm_OT_O = retrieve_png(directory=path_fitness_arm_OT_O)
images_viobox_arm_IT_NO = retrieve_png(directory=path_viobox_arm_IT_NO)
images_viobox_arm_IT_O = retrieve_png(directory=path_viobox_arm_IT_O)
images_viobox_arm_OT_NO = retrieve_png(directory=path_viobox_arm_OT_NO)
images_viobox_arm_OT_O =retrieve_png(directory=path_viobox_arm_OT_O)

images_BD_noisy_arm_IT_NO = retrieve_png(directory=path_BD_noisy_arm_IT_NO)
images_BD_noisy_arm_IT_O = retrieve_png(directory=path_BD_noisy_arm_IT_O)
images_BD_noisy_arm_OT_NO = retrieve_png(directory=path_BD_noisy_arm_OT_NO)
images_BD_noisy_arm_OT_O = retrieve_png(directory=path_BD_noisy_arm_OT_O)
images_fitness_noisy_arm_IT_NO = retrieve_png(directory=path_fitness_noisy_arm_IT_NO)
images_fitness_noisy_arm_IT_O = retrieve_png(directory=path_fitness_noisy_arm_IT_O)
images_fitness_noisy_arm_OT_NO = retrieve_png(directory=path_fitness_noisy_arm_OT_NO)
images_fitness_noisy_arm_OT_O = retrieve_png(directory=path_fitness_noisy_arm_OT_O)
images_viobox_noisy_arm_IT_NO = retrieve_png(directory=path_viobox_noisy_arm_IT_NO)
images_viobox_noisy_arm_IT_O = retrieve_png(directory=path_viobox_noisy_arm_IT_O)
images_viobox_noisy_arm_OT_NO = retrieve_png(directory=path_viobox_noisy_arm_OT_NO)
images_viobox_noisy_arm_OT_O = retrieve_png(directory=path_viobox_noisy_arm_OT_O)
# for img in images_BD_arm_IT_NO:
#     print(img)


###############################
#            FULL             #
###############################

# Merging the path and the name of the .png file
full_loss_arm = sorted(add_path_to_images(path_loss_arm, images_loss_arm))
full_loss_noisy_arm = sorted(add_path_to_images(path_loss_noisy_arm, images_loss_noisy_arm))
full_BD_arm_IT_NO = sorted(add_path_to_images(path_BD_arm_IT_NO, images_BD_arm_IT_NO))
full_BD_arm_IT_O = sorted(add_path_to_images(path_BD_arm_IT_O, images_BD_arm_IT_O))
full_BD_arm_OT_NO = sorted(add_path_to_images(path_BD_arm_OT_NO, images_BD_arm_OT_NO))
full_BD_arm_OT_O = sorted(add_path_to_images(path_BD_arm_OT_O, images_BD_arm_OT_O))
full_fitness_arm_IT_NO = sorted(add_path_to_images(path_fitness_arm_IT_NO, images_fitness_arm_IT_NO))
full_fitness_arm_IT_O = sorted(add_path_to_images(path_fitness_arm_IT_O, images_fitness_arm_IT_O))
full_fitness_arm_OT_NO = sorted(add_path_to_images(path_fitness_arm_OT_NO, images_fitness_arm_OT_NO))
full_fitness_arm_OT_O = sorted(add_path_to_images(path_fitness_arm_OT_O, images_fitness_arm_OT_O))
full_viobox_arm_IT_NO = sorted(add_path_to_images(path_viobox_arm_IT_NO, images_viobox_arm_IT_NO))
full_viobox_arm_IT_O = sorted(add_path_to_images(path_viobox_arm_IT_O, images_viobox_arm_IT_O))
full_viobox_arm_OT_NO = sorted(add_path_to_images(path_viobox_arm_OT_NO, images_viobox_arm_OT_NO))
full_viobox_arm_OT_O = sorted(add_path_to_images(path_viobox_arm_OT_O, images_viobox_arm_OT_O))
full_BD_noisy_arm_IT_NO = sorted(add_path_to_images(path_BD_noisy_arm_IT_NO, images_BD_noisy_arm_IT_NO))
full_BD_noisy_arm_IT_O = sorted(add_path_to_images(path_BD_noisy_arm_IT_O, images_BD_noisy_arm_IT_O))
full_BD_noisy_arm_OT_NO = sorted(add_path_to_images(path_BD_noisy_arm_OT_NO, images_BD_noisy_arm_OT_NO))
full_BD_noisy_arm_OT_O = sorted(add_path_to_images(path_BD_noisy_arm_OT_O, images_BD_noisy_arm_OT_O))
full_fitness_noisy_arm_IT_NO = sorted(add_path_to_images(path_fitness_noisy_arm_IT_NO, images_fitness_noisy_arm_IT_NO))
full_fitness_noisy_arm_IT_O = sorted(add_path_to_images(path_fitness_noisy_arm_IT_O, images_fitness_noisy_arm_IT_O))
full_fitness_noisy_arm_OT_NO = sorted(add_path_to_images(path_fitness_noisy_arm_OT_NO, images_fitness_noisy_arm_OT_NO))
full_fitness_noisy_arm_OT_O = sorted(add_path_to_images(path_fitness_noisy_arm_OT_O, images_fitness_noisy_arm_OT_O))
full_viobox_noisy_arm_IT_NO = sorted(add_path_to_images(path_viobox_noisy_arm_IT_NO, images_viobox_noisy_arm_IT_NO))
full_viobox_noisy_arm_IT_O = sorted(add_path_to_images(path_viobox_noisy_arm_IT_O, images_viobox_noisy_arm_IT_O))
full_viobox_noisy_arm_OT_NO = sorted(add_path_to_images(path_viobox_noisy_arm_OT_NO, images_viobox_noisy_arm_OT_NO))
full_viobox_noisy_arm_OT_O = sorted(add_path_to_images(path_viobox_noisy_arm_OT_O, images_viobox_noisy_arm_OT_O))
# for full_path in full_fitness_arm_IT_NO:
#     print(full_path)


###############################
#         IMG CONFIG          #
###############################

# Image resolution (the higher the better, but the higher the heavier)
dpi = 300 # 1200
# 1 row and 2 columns figures dimensions
fig_width_1_2 = 12
fig_height_1_2 = 6
# 2 rows and 2 columns figures dimensions
fig_width_2_2 = 20
fig_height_2_2 = 16


###############################
#         LOSS FIGURE         #
###############################

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_1_2, fig_height_1_2))
# for ax, image_path in zip(axes.flatten(), full_loss):
#     img = mpimg.imread(image_path)
#     ax.imshow(img)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
# plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Loss/{title}Losses_Figures.png', dpi=dpi)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for pos, (img_path_1, img_path_2) in enumerate(zip(full_loss_arm, full_loss_noisy_arm)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Position of the plot
    row = pos // 2
    col = pos % 2
    # Plotting
    axes[row, col].imshow(img_1)
    axes[row, col].axis('off')
    axes[row+1, col].imshow(img_2)
    axes[row+1, col].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Loss/{title}Losses_Figures.png', dpi=dpi)

###############################
#          BD FIGURE          #
###############################

# In training (IT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for pos, (img_path_1, img_path_2) in enumerate(zip(full_BD_arm_IT_NO, full_BD_noisy_arm_IT_NO)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Position of the plot
    row = pos // 2
    col = pos % 2
    # Plotting
    axes[row, col].imshow(img_1)
    axes[row, col].axis('off')
    axes[row+1, col].imshow(img_2)
    axes[row+1, col].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/BD/{title}BD_IT&NO_Figures.png', dpi=dpi)

# Outside training (OT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for pos, (img_path_1, img_path_2) in enumerate(zip(full_BD_arm_OT_NO, full_BD_noisy_arm_OT_NO)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Position of the plot
    row = pos // 2
    col = pos % 2
    # Plotting
    axes[row, col].imshow(img_1)
    axes[row, col].axis('off')
    axes[row+1, col].imshow(img_2)
    axes[row+1, col].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/BD/{title}BD_OT&NO_Figures.png', dpi=dpi)

# In training (IT) & Outliers (O)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for pos, (img_path_1, img_path_2) in enumerate(zip(full_BD_arm_IT_O, full_BD_noisy_arm_IT_O)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Position of the plot
    row = pos // 2
    col = pos % 2
    # Plotting
    axes[row, col].imshow(img_1)
    axes[row, col].axis('off')
    axes[row+1, col].imshow(img_2)
    axes[row+1, col].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/BD/{title}BD_IT&O_Figures.png', dpi=dpi)

# Outside training (OT) & Outliers (O)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for pos, (img_path_1, img_path_2) in enumerate(zip(full_BD_arm_OT_O, full_BD_noisy_arm_OT_O)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Position of the plot
    row = pos // 2
    col = pos % 2
    # Plotting
    axes[row, col].imshow(img_1)
    axes[row, col].axis('off')
    axes[row+1, col].imshow(img_2)
    axes[row+1, col].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/BD/{title}BD_OT&O_Figures.png', dpi=dpi)


# ###############################
# #       FITNESS FIGURE        #
# ###############################

# In training (IT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_1_2, fig_height_1_2))
for i, (img_path_1, img_path_2) in enumerate(zip(full_fitness_arm_IT_NO, full_fitness_noisy_arm_IT_NO)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Plotting images
    axes[i].imshow(img_1)
    axes[i].axis('off')
    axes[i+1].imshow(img_2)
    axes[i+1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Fitness/{title}Fitness_IT&NO_Figures.png', dpi=dpi)

# Outside training (OT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_1_2, fig_height_1_2))
for i, (img_path_1, img_path_2) in enumerate(zip(full_fitness_arm_OT_NO, full_fitness_noisy_arm_OT_NO)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Plotting images
    axes[i].imshow(img_1)
    axes[i].axis('off')
    axes[i+1].imshow(img_2)
    axes[i+1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Fitness/{title}Fitness_OT&NO_Figures.png', dpi=dpi)

# In training (IT) & Outliers (O)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_1_2, fig_height_1_2))
for i, (img_path_1, img_path_2) in enumerate(zip(full_fitness_arm_IT_O, full_fitness_noisy_arm_IT_O)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Plotting images
    axes[i].imshow(img_1)
    axes[i].axis('off')
    axes[i+1].imshow(img_2)
    axes[i+1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Fitness/{title}Fitness_IT&O_Figures.png', dpi=dpi)

# Outside training (OT) & Outliers (O)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_1_2, fig_height_1_2))
for i, (img_path_1, img_path_2) in enumerate(zip(full_fitness_arm_OT_O, full_fitness_noisy_arm_OT_O)):
    # Loading images
    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    # Plotting images
    axes[i].imshow(img_1)
    axes[i].axis('off')
    axes[i+1].imshow(img_2)
    axes[i+1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Fitness/{title}Fitness_OT&O_Figures.png', dpi=dpi)


# ###############################
# #         VIOLIN PLOTS        #
# ###############################

# In training (IT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_arm_IT_NO):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/arm/{title}Viobox_arm_IT&NO_Figures.png', dpi=dpi)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_noisy_arm_IT_NO):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/noisy_arm/{title}Viobox_noisy_arm_IT&NO_Figures.png', dpi=dpi)

# Outside training (OT) & No Outliers (NO)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_arm_OT_NO):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/arm/{title}Viobox_arm_OT&NO_Figures.png', dpi=dpi)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_noisy_arm_OT_NO):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/noisy_arm/{title}Viobox_noisy_arm_OT&NO_Figures.png', dpi=dpi)

# In training (IT) & Outliers (O)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_arm_IT_O):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/arm/{title}Viobox_arm_IT&O_Figures.png', dpi=dpi)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_noisy_arm_IT_O):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/noisy_arm/{title}Viobox_noisy_arm_IT&O_Figures.png', dpi=dpi)

# Outside training (OT) & Outliers (O)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_arm_OT_O):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/arm/{title}Viobox_arm_OT&O_Figures.png', dpi=dpi)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_2_2, fig_height_2_2))
for ax, image_path in zip(axes.flatten(), full_viobox_noisy_arm_OT_O):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig(f'results_sim_arg/Grouping_Figures/{method}/Viobox/noisy_arm/{title}Viobox_noisy_arm_OT&O_Figures.png', dpi=dpi)
