import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix

# Load gene counts and select genes with 613 counts
def load_selected_genes(gene_count_file):
    selected_genes = []
    with open(gene_count_file, 'r') as f:
        for line in f:
            gene, count = line.strip().split(':')
            if int(count) == 613:
                selected_genes.append(gene.strip())
    return selected_genes

# Process .pkl files and handle missing genes
def process_pkl_files(pkl_dir, selected_genes, output_dir, chunk_size=100):
    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(pkl_dir, filename)

            # Load the .pkl file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

                gene_list = data[0]  # List of genes
                cell_ids = data[1]  # List of cell IDs
                matrix = data[2]  # Sparse matrix

                # Convert sparse matrix to dense
                full_matrix = matrix.toarray()

                # Create a matrix with all selected genes, filled with zeros initially
                num_cells = full_matrix.shape[0]
                selected_gene_matrix = np.zeros((num_cells, len(selected_genes)))

                # Map existing genes to their positions in the selected_gene_matrix
                for i, gene in enumerate(selected_genes):
                    if gene in gene_list:
                        gene_index = gene_list.index(gene)  # Get the index of the gene in the gene_list
                        selected_gene_matrix[:, i] = full_matrix[:, gene_index]

                # Process in chunks of 100 cells
                for start in range(0, num_cells, chunk_size):
                    end = min(start + chunk_size, num_cells)
                    cell_chunk = selected_gene_matrix[start:end]
                    cell_id_chunk = cell_ids[start:end]

                    # Combine cell IDs and filtered matrix for output
                    combined_output = np.column_stack((cell_id_chunk, cell_chunk))

                    # Save the chunk as a .txt file
                    chunk_file = os.path.join(output_dir, f"{filename}_chunk_{start}_{end}.txt")
                    np.savetxt(chunk_file, combined_output, delimiter='\t', fmt='%s')

                    print(f"Saved: {chunk_file}")

# Main Function to run
if __name__ == "__main__":
    # Paths
    gene_count_file = 'gene_counts.txt'  # Path to your gene count file
    pkl_dir = '/local/disk5/clara/normal_processed'  # Path to directory containing .pkl files
    output_dir = '/local/disk5/clara/normal_processed_chunk'  # Path to save output files

    # Load selected genes with 613 counts
    selected_genes = load_selected_genes(gene_count_file)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the pkl files and save the filtered chunks
    process_pkl_files(pkl_dir, selected_genes, output_dir)

