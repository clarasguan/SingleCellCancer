import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the data from the CSV file
data = pd.read_csv('ensemble_predictions_fold_0.csv')

# Extract the base file name (before the first underscore)
data['Base File Name'] = data['File Name'].str.split('_').str[0]

# Group data by base file name
file_groups = data.groupby('Base File Name')

# Initialize a dictionary to store confusion matrices for each base file name
conf_matrices = {}

# Iterate over each base file and calculate its confusion matrix
for base_file_name, group in file_groups:
    true_labels = group['True Label']
    binary_predictions = group['Binary Prediction']
    
    # Calculate the confusion matrix, ensuring labels are [0, 1]
    conf_matrix = confusion_matrix(true_labels, binary_predictions, labels=[0, 1])
    
    # Convert the confusion matrix to a DataFrame for better readability
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['True Negative (0)', 'True Positive (1)'],
        columns=['Predicted Negative (0)', 'Predicted Positive (1)']
    )
    
    # Store the confusion matrix in the dictionary
    conf_matrices[base_file_name] = conf_matrix_df

# Display the confusion matrices for each base file
for base_file_name, matrix in conf_matrices.items():
    print(f"Confusion Matrix for {base_file_name}:")
    print(matrix)
    print()

