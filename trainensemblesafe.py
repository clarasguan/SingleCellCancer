import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
from collections import defaultdict

# Get list of files from directories
def get_file_list(healthy_dir, unhealthy_dir):
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".txt")]
    unhealthy_files = [os.path.join(unhealthy_dir, f) for f in os.listdir(unhealthy_dir) if f.endswith(".txt")]
    all_files = [(f, 0) for f in healthy_files] + [(f, 1) for f in unhealthy_files]  # Label: 0 for healthy, 1 for unhealthy

    print(f"Found {len(healthy_files)} healthy files and {len(unhealthy_files)} unhealthy files.")
    return all_files

# Load a batch of data from files
def load_batch(files):
    batch_data = []
    batch_labels = []
    batch_file_names = []  # New list to track filenames row by row

    for file_path, label in files:
        data = pd.read_csv(file_path, delimiter="\t", header=None)
        X = data.iloc[:, 1:].values  # Exclude the ID column
        y = np.full((len(data), 1), label)  # Assign the same label to all rows

        # Append data and labels
        batch_data.append(X)
        batch_labels.append(y)

        # Append the filename for each row in this file
        file_name = os.path.basename(file_path)
        file_names_for_this_file = [file_name] * len(data)
        batch_file_names.extend(file_names_for_this_file)

    # Concatenate all batch data and labels
    X_batch = np.vstack(batch_data)
    y_batch = np.vstack(batch_labels)

    print(f"Loaded a batch with {X_batch.shape[0]} samples.")
    return X_batch, y_batch, batch_file_names


# Define the neural network model
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

# Train a single model
def train_single_model(model, optimizer, criterion, files, batch_size=1):
    model.train()
    total_loss = 0

    # Shuffle the files before each epoch
    random.shuffle(files)

    # Load data in batches
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        X_batch, y_batch, _ = load_batch(batch_files)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.FloatTensor(y_batch)

        # Forward pass
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item()

        # Print the loss for the current batch
        print(f"Batch {i // batch_size + 1}, Loss: {loss.item():.4f}")

    # Return the average loss for the epoch
    return total_loss / (len(files) / batch_size)

# Evaluate an ensemble of models by averaging their predictions
def evaluate_ensemble(models, files, batch_size=100, save_path=None, auc_data_path=None, auc_value_path=None):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    all_file_names = []  # This will now store file names for each row

    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            X_batch, y_batch, batch_file_names = load_batch(batch_files)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_batch)
            y_tensor = torch.FloatTensor(y_batch)

            # Make predictions from each model and average them
            predictions = torch.zeros(X_tensor.size(0), 1)
            for model in models:
                predictions += model(X_tensor)

            predictions /= len(models)  # Average predictions
            binary_predictions = (predictions >= 0.5).float()  # Convert to binary predictions

            correct += (binary_predictions.eq(y_tensor).sum().item())
            total += y_tensor.size(0)

            # Collect predictions and true labels
            all_predictions.append(predictions.numpy())
            all_true_labels.append(y_tensor.numpy())
            all_file_names.extend(batch_file_names)

    accuracy = correct / total
    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")

    # Save predictions if save_path is provided
    if save_path:
        predictions_array = np.vstack(all_predictions)
        true_labels_array = np.vstack(all_true_labels)

        predictions_df = pd.DataFrame({
            "File Name": all_file_names,
            "True Label": true_labels_array.flatten(),
            "Predicted Probability": predictions_array.flatten(),
            "Binary Prediction": (predictions_array >= 0.5).astype(int).flatten()
        })
        predictions_df.to_csv(save_path, index=False)
        print(f"Saved predictions to {save_path}")

    # Save data for AUC plotting if auc_data_path is provided
    if auc_data_path:
        auc_data = pd.DataFrame({
            "True Label": np.vstack(all_true_labels).flatten(),
            "Predicted Probability": np.vstack(all_predictions).flatten()
        })
        auc_data.to_csv(auc_data_path, index=False)
        print(f"Saved AUC data to {auc_data_path}")

    # Calculate and save AUC value if auc_value_path is provided
    if auc_value_path:
        true_labels_flat = np.vstack(all_true_labels).flatten()
        predicted_probs_flat = np.vstack(all_predictions).flatten()
        auc_value = roc_auc_score(true_labels_flat, predicted_probs_flat)

        with open(auc_value_path, "w") as f:
            f.write(f"AUC Value: {auc_value:.4f}\n")

        print(f"Saved AUC value: {auc_value:.4f} to {auc_value_path}")

    return accuracy

# Train and evaluate each model and get final ensemble predictions
def train_and_evaluate_ensemble(files, input_size, num_models=5):
    # Split groups into 80% training and 20% testing
    file_groups = defaultdict(list)
    for file, label in files:
        # Extract the unique identifier (everything before '_')
        base_identifier = file.split('_')[0]
        file_groups[base_identifier].append((file, label))
    
    grouped_files = list(file_groups.items())

    # Split groups into 80% training and 20% testing
    train_groups, test_groups = train_test_split(grouped_files, test_size=0.2, random_state=int(sys.argv[1]))
    
    train_files = [(file, label) for _, chunks in train_groups for file, label in chunks]
    test_files = [(file, label) for _, chunks in test_groups for file, label in chunks]
    
    with open('train_files.txt' + sys.argv[1], 'w') as f:
        for file, label in train_files:
            f.write(f"{file},{label}\n")
    with open('test_files.txt' + sys.argv[1], 'w') as f:
        for file, label in test_files:
            f.write(f"{file},{label}\n")

    models = []

    for i in range(num_models):
        print(f"\nTraining model {i + 1}/{num_models} with random seed {i}")
        torch.manual_seed(i)

        model = GeneExpressionClassifier(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        for epoch in range(1):
            print(f"Epoch {epoch + 1} for model {i + 1}")
            train_loss = train_single_model(model, optimizer, criterion, train_files)
            print(f"Model {i + 1}, Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        model_save_path = f"model_fold_{i+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model {i + 1} to {model_save_path}")

        models.append(model)

    auc_data_path = f"auc_data_fold_{sys.argv[1]}.csv"
    auc_value_path = f"auc_value_fold_{sys.argv[1]}.txt"
    predictions_save_path = f"ensemble_predictions_fold_{sys.argv[1]}.csv"
    accuracy = evaluate_ensemble(models, test_files, save_path=predictions_save_path, auc_data_path=auc_data_path, auc_value_path=auc_value_path)

    return accuracy

if __name__ == "__main__":
    healthy_dir = 'input/healthy'
    unhealthy_dir = 'input/unhealthy'

    files = get_file_list(healthy_dir, unhealthy_dir)

    sample_data = pd.read_csv(files[0][0], delimiter="\t", header=None)
    input_size = sample_data.shape[1] - 1

    print(f"Input size detected: {input_size}")

    accuracy = train_and_evaluate_ensemble(files, input_size)

    with open("ensemble_accuracy_results.txt." + sys.argv[1], "w") as f:
        f.write(f"Ensemble Test set accuracy: {accuracy * 100:.2f}%\n")

    print(f"\nEnsemble Test set accuracy: {accuracy * 100:.2f}%")

