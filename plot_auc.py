import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set global font size
plt.rcParams.update({'font.size': 16})

# Replace with your actual file paths or names
file_names = [
    "auc_data_fold_0.csv",
    "auc_data_fold_1.csv",
    "auc_data_fold_2.csv",
    "auc_data_fold_3.csv",
    "auc_data_fold_4.csv",
]

plt.figure(figsize=(8, 7))

# Loop through each file to compute and plot the ROC curve
for file_name in file_names:
    # Load the data
    data = pd.read_csv(file_name)
    
    # Extract true labels and predicted probabilities
    true_labels = data["True Label"]
    predicted_probs = data["Predicted Probability"]
    
    # Compute the ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'{file_name} (AUC = {roc_auc:.2f})')

# Plot diagonal (random classifier line)
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")

# Customize the plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()

# Save the figure
output_path = "roc_curves_plot.png"
plt.savefig(output_path)

# Show the plot
plt.show()

print(f"Figure saved to: {output_path}")

