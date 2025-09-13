import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# Define the neural network model (same as in the training script)
class GeneExpressionClassifier(nn.Module):
    def __init__(self, input_size):
        super(GeneExpressionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Load data from files
def load_batch(files):
    batch_data = []
    batch_labels = []
    batch_file_names = []

    for file_path, label in files:
        data = pd.read_csv(file_path, delimiter="\t", header=None)
        X = data.iloc[:, 1:].values  # Exclude the ID column
        y = np.full((len(data), 1), label)  # Assign the same label to all rows

        batch_data.append(X)
        batch_labels.append(y)
        batch_file_names.append(os.path.basename(file_path))

    # Concatenate all batch data and labels
    X_batch = np.vstack(batch_data)
    y_batch = np.vstack(batch_labels)

    return X_batch, y_batch, batch_file_names

# Predict using an ensemble of pre-trained models
def predict_with_ensemble(test_files, input_size, model_paths, predictions_save_path):
    # Load models
    models = []
    for model_path in model_paths:
        model = GeneExpressionClassifier(input_size=input_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)

    # Perform predictions
    all_predictions = []
    all_true_labels = []
    all_file_names = []

    with torch.no_grad():
        # Group files to ensure consistent predictions for files from the same source
        file_groups = defaultdict(list)
        for file, label in test_files:
            base_identifier = os.path.basename(file)
            file_groups[base_identifier].append((file, label))

        for base_identifier, group_files in file_groups.items():
            X_batch, y_batch, batch_file_names = load_batch(group_files)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_batch)
            y_tensor = torch.FloatTensor(y_batch)

            # Make predictions from each model and average them
            predictions = torch.zeros(X_tensor.size(0), 1)
            for model in models:
                predictions += model(X_tensor)

            predictions /= len(models)  # Average predictions
            binary_predictions = (predictions >= 0.5).float()  # Convert to binary predictions

            # Collect results
            all_predictions.append(predictions.numpy())
            all_true_labels.append(y_batch)
            all_file_names.extend(batch_file_names)

    # Prepare DataFrame for saving
    predictions_df = pd.DataFrame({
        "File Name": all_file_names,
        "True Label": np.vstack(all_true_labels).flatten(),
        "Predicted Probability": np.vstack(all_predictions).flatten(),
        "Binary Prediction": (np.vstack(all_predictions) >= 0.5).astype(int).flatten()
    })
    
    # Save predictions
    predictions_df.to_csv(predictions_save_path, index=False)
    print(f"Saved predictions to {predictions_save_path}")

    # Calculate and print accuracy
    correct = (predictions_df['Binary Prediction'] == predictions_df['True Label']).sum()
    total = len(predictions_df)
    accuracy = correct / total
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    return predictions_df

# Main execution
if __name__ == "__main__":
    # Directories and file paths
    healthy_dir = 'input/healthy'
    unhealthy_dir = 'input/unhealthy'

    # Get list of files
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".txt")]
    unhealthy_files = [os.path.join(unhealthy_dir, f) for f in os.listdir(unhealthy_dir) if f.endswith(".txt")]
    all_files = [(f, 0) for f in healthy_files] + [(f, 1) for f in unhealthy_files]

    # Load sample data to get input size
    sample_data = pd.read_csv(all_files[0][0], delimiter="\t", header=None)
    input_size = sample_data.shape[1] - 1

    # Paths to pre-trained models
    model_paths = [
        'model_fold_1.pth',
        'model_fold_2.pth',
        'model_fold_3.pth',
        'model_fold_4.pth',
        'model_fold_5.pth'
    ]

    # Prediction
    predictions_save_path = f"ensemble_predictions_fold_{sys.argv[1]}.csv"
    predictions_df = predict_with_ensemble(all_files, input_size, model_paths, predictions_save_path)
