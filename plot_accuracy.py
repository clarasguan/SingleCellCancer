import matplotlib.pyplot as plt

# Data for the boxplot
group_1 = [99.61, 99.65, 99.57, 99.56, 99.51]
group_2 = [63.1, 61.68, 96.79, 95.86, 89.79, 60.08]

# Create the plot
plt.figure(figsize=(4, 4))  # Smaller figure size
plt.boxplot(
    [group_1, group_2], 
    labels=['Mixed dataset', 'Cross dataset'], 
    patch_artist=True
)

# Adjust font sizes
plt.ylabel('accuracy (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set y-axis limits
plt.ylim(0, 120)

# Save the figure
plt.tight_layout()
plt.savefig('boxplot.png', dpi=300)  # Save as PNG with high resolution (300 dpi)

# Show the plot
plt.show()

