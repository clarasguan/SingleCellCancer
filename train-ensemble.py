import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
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
def load_batch(files, return_ids=False):
    batch_data = []
    batch_labels = []
    batch_ids = []  # collect first-column IDs when requested

    for file_path, label in files:
        data = pd.read_csv(file_path, delimiter="\t", header=None)
        ids = data.iloc[:, 0].astype(str).values       # first column = cell id
        X = data.iloc[:, 1:].values                    # features (exclude ID col)
        y = np.full((len(data), 1), label, dtype=np.float32)

        batch_data.append(X)
        batch_labels.append(y)
        if return_ids:
            batch_ids.append(ids)

    # Concatenate all batch data and labels
    X_batch = np.vstack(batch_data)
    y_batch = np.vstack(batch_labels)

    if return_ids:
        ids_batch = np.concatenate(batch_ids)
        print(f"Loaded a batch with {X_batch.shape[0]} samples (with IDs).")
        return X_batch, y_batch, ids_batch

    print(f"Loaded a batch with {X_batch.shape[0]} samples.")
    return X_batch, y_batch

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
        X_batch, y_batch = load_batch(batch_files)

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
# and SAVE per-cell predictions with the first column as ID.
def evaluate_ensemble(models, files, batch_size=100, save_path=None):
    for model in models:
        model.eval()

    correct = 0
    total = 0

    all_ids = []
    all_true = []
    all_prob = []
    all_pred = []

    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            X_batch, y_batch, ids_batch = load_batch(batch_files, return_ids=True)

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

            # Collect per-cell outputs for saving
            all_ids.extend(ids_batch.tolist())
            all_true.extend(y_tensor.squeeze(1).cpu().numpy().tolist())
            all_prob.extend(predictions.squeeze(1).cpu().numpy().tolist())
            all_pred.extend(binary_predictions.squeeze(1).cpu().numpy().tolist())

    accuracy = correct / total
    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")

    # Save predictions if requested
    if save_path is not None:
        df = pd.DataFrame({
            "cell_id": all_ids,           # first column from txt
            "true_label": all_true,
            "pred_proba": all_prob,
            "pred_label": all_pred
        })
        df.to_csv(save_path, index=False)
        print(f"Saved predictions to: {save_path}")

    return accuracy

# Train and evaluate each model and get final ensemble predictions
def train_and_evaluate_ensemble(files, input_size, seed=0, num_models=5):
    # Group files by base identifier (prefix before first underscore of basename)
    file_groups = defaultdict(list)
    for file, label in files:
        base_identifier = os.path.basename(file).split('_')[0]
        file_groups[base_identifier].append((file, label))

    grouped_files = list(file_groups.items())

    # Split groups into 80% training and 20% testing
    train_groups, test_groups = train_test_split(grouped_files, test_size=0.2, random_state=seed)

    # Flatten grouped data into individual files for train and test sets
    train_files = [(file, label) for _, chunks in train_groups for file, label in chunks]
    test_files = [(file, label) for _, chunks in test_groups for file, label in chunks]

    # Save train and test file lists
    suffix = f".{seed}"
    with open('train_files.txt' + suffix, 'w') as f:
        for file, label in train_files:
            f.write(f"{file},{label}\n")
    with open('test_files.txt' + suffix, 'w') as f:
        for file, label in test_files:
            f.write(f"{file},{label}\n")

    # Train ensemble
    models = []
    for i in range(num_models):
        print(f"\nTraining model {i + 1}/{num_models} with random seed {i}")
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        model = GeneExpressionClassifier(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for a fixed number of epochs
        for epoch in range(1):
            print(f"Epoch {epoch + 1} for model {i + 1}")
            train_loss = train_single_model(model, optimizer, criterion, train_files)
            print(f"Model {i + 1}, Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        models.append(model)

    # Evaluate ensemble and save per-cell predictions with IDs
    pred_path = "ensemble_predictions.csv" + suffix
    accuracy = evaluate_ensemble(models, test_files, save_path=pred_path)

    return accuracy

# Main execution
if __name__ == "__main__":
    healthy_dir = 'input/healthy'
    unhealthy_dir = 'input/unhealthy'

    # Use sys.argv[1] as seed if provided; default 0 for reproducibility
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Using seed: {seed}")

    # Global seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get the list of files from directories
    files = get_file_list(healthy_dir, unhealthy_dir)
    if not files:
        raise RuntimeError("No .txt files found in the specified input directories.")

    # Calculate input size (exclude the ID column)
    sample_data = pd.read_csv(files[0][0], delimiter="\t", header=None)
    input_size = sample_data.shape[1] - 1  # Exclude only the ID column
    print(f"Input size detected: {input_size}")

    # Train and evaluate the ensemble of models
    accuracy = train_and_evaluate_ensemble(files, input_size, seed=seed)

    # Save the accuracy to a file
    acc_path = f"ensemble_accuracy_results.txt.{seed}"
    with open(acc_path, "w") as f:
        f.write(f"Ensemble Test set accuracy: {accuracy * 100:.2f}%\n")
    print(f"\nEnsemble Test set accuracy: {accuracy * 100:.2f}%")
    print(f"Saved accuracy to: {acc_path}")

