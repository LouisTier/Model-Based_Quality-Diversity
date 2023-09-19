import numpy as np
import matplotlib.pyplot as plt

# num_epochs = 125 * int(40000*0.7/64)
num_epochs = 125 * 6
lr_gene = 0.000001
gene_decay_factor = 0.9
lr_discrim = 0.000001
discrim_decay_factor = 0.99
num1 = 450
num2 = 100

plot_epochs = np.arange(num_epochs)

decay_function_gene_plot = lambda epoch: lr_gene * gene_decay_factor ** (epoch/num1)
learning_rates_gene = [decay_function_gene_plot(epoch) for epoch in plot_epochs]

decay_function_discrim_plot = lambda epoch: lr_discrim * discrim_decay_factor ** (epoch/num2)
learning_rates_discrim = [decay_function_discrim_plot(epoch) for epoch in plot_epochs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.plot(plot_epochs, learning_rates_gene)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Learning rate")
ax1.set_title("Evolution of the learning rate (Generator)\n")

ax2.plot(plot_epochs, learning_rates_discrim)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Learning rate")
ax2.set_title("Evolution of the learning rate (Discriminator)\n")

# fig.subplots_adjust(wspace=0.35)
plt.tight_layout
plt.show()
plt.savefig("test")