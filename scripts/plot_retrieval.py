import matplotlib.pyplot as plt

# Updated data for food172
k_values = [5, 10, 15, 20]
food101 = [91.17, 91.12, 90.60, 90.50]
food172 = [84.48, 83.25, 82.07, 81.50]  # Updated values

# Plotting the data
plt.figure(figsize=(6, 4))  # Reduced size for a more compact appearance

# Plot for food101
plt.plot(k_values, food101, marker='s', color='red', linestyle='-', label='food101')

# Plot for food172
plt.plot(k_values, food172, marker='o', color='green', linestyle='-', label='food172')

# Adding titles and labels
plt.title('')
plt.xlabel('k')
plt.ylabel('Top-1 accuracy of RAFR+LLaVA1.5-7B')
plt.grid(True)
plt.xticks([5, 10, 15, 20])  # Set x-axis ticks to show only 5, 10, 15, and 20
plt.legend()
plt.tight_layout()  # Make the layout more compact

# Save the updated plot as a high-resolution PNG file
output_path_updated = "/home/data_llm/madehua/LLaVA/scripts/llava_performance_updated_plot.png"
plt.savefig(output_path_updated, format='png', dpi=300)  # Increase dpi for high resolution
