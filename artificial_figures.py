"""
This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)

# Model and noise definition
x = np.linspace(0, 10, 100)
ground_truth = np.cos(0.8*x)
observations1 = np.random.normal(loc=0, scale=0.1, size=20)
observations2 = np.random.normal(loc=0, scale=0.7, size=20)

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(1, 5):
    ax.axvline(x=i*2, color='gray', linestyle='--')

# Plot points and model
ax.plot(x, ground_truth, label='Ground Truth Model: cos(0.8x)')
ax.scatter(x[20:40], ground_truth[20:40] + observations2, color='red', label='Observation Data: Bad Quality')
ax.scatter(x[60:80], ground_truth[60:80] + observations1, color='blue', label='Observation Data: Good Quality')

# Add shaded regions
ax.axvspan(0, 2, alpha=0.1, color='gray')
ax.axvspan(2, 4, alpha=0.1, color='red')
ax.axvspan(4, 6, alpha=0.1, color='gray')
ax.axvspan(6, 8, alpha=0.1, color='blue')
ax.axvspan(8, 10, alpha=0.1, color='gray')

# Add text in the shaded regions
ax.text(1, -2, 'High Epistemic\nUncertainty', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(3, -2.5, 'High Aleatoric\nUncertainty', ha='center', va='center', fontsize=12, fontweight='bold', color = 'red')
ax.text(5, -2, 'High Epistemic\nUncertainty', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(7, -2.5, 'Low Aleatoric\nUncertainty', ha='center', va='center', fontsize=12, fontweight='bold', color = 'blue')
ax.text(9, -2, 'High Epistemic\nUncertainty', ha='center', va='center', fontsize=12, fontweight='bold')

# Define figure
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Comparison of the two types of uncertainty with fictitious data and model', fontsize=14)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 2.5)
ax.legend(loc='upper right')

# Save the figure
plt.savefig('Uncertainties.png')
