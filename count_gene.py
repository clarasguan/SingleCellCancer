import os
import pickle
from collections import Counter

# Define the directory containing the .pkl files
gene_counter = Counter()
pkl_dir = '/local/disk5/clara/processed'

# Create a Counter to store gene counts

# Iterate over each file in the directory
for filename in os.listdir(pkl_dir):
    if filename.endswith(".pkl"):
        file_path = os.path.join(pkl_dir, filename)
        
        # Load the .pkl file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
            # Extract the first element (the gene list)
            gene_list = data[0]
            
            # Update the gene counts
            gene_counter.update(gene_list)

pkl_dir = '/local/disk5/clara/normal_processed'

# Create a Counter to store gene counts

# Iterate over each file in the directory
for filename in os.listdir(pkl_dir):
    if filename.endswith(".pkl"):
        file_path = os.path.join(pkl_dir, filename)
        
        # Load the .pkl file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
            # Extract the first element (the gene list)
            gene_list = data[0]
            gene_counter.update(gene_list)
# Write the gene counts to a .txt file
output_file = 'gene_counts.txt'
with open(output_file, 'w') as f:
    for gene, count in gene_counter.items():
        f.write(f"{gene}: {count}\n")

print(f"Gene counts have been written to {output_file}")

