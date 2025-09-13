import os
import pandas as pd
import pickle

# Specify the directory containing the pkl files and the new directory for csv files
pkl_dir = '../processed/'
csv_dir = '../processed_csv'

# Create the new directory if it doesn't exist
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Loop through all the pkl files in the specified directory
for filename in os.listdir(pkl_dir):
    if filename.endswith('.pkl'):
        # Construct the full file path
        pkl_file_path = os.path.join(pkl_dir, filename)
        
        # Load the pkl file using pickle
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Assuming the data is a pandas DataFrame, convert it to csv
        if isinstance(data, pd.DataFrame):
            # Create the output csv file path
            csv_file_path = os.path.join(csv_dir, filename.replace('.pkl', '.csv'))
            
            # Save the DataFrame as a CSV file
            data.to_csv(csv_file_path, index=False)
            print(f"Converted {filename} to CSV and saved as {csv_file_path}")
        else:
            print(f"Warning: {filename} is not a pandas DataFrame, skipping...")

