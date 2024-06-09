import pandas as pd
import os

# Load final_data.tsv
final_data = pd.read_csv("/home/ibab/SEM_4/project/data/gvcf/final/final_data.tsv", sep='\t', index_col=0)
directory = "/home/ibab/SEM_4/project/data/gvcf/final/filt_ann_gvcf"

# Write column headings only once
final_data.to_csv("final_data_updated.tsv", mode="w", sep='\t')

# Iterate over each file
for filename in os.listdir(directory):
    if filename.endswith(".tsv") and filename != "final_data.tsv":
        file_path = os.path.join(directory, filename)  # Full path to the data file
        file_name = os.path.splitext(filename)[0]
        # Load data file
        data = pd.read_csv(file_path, sep='\t')

        # Count occurrences of variant types and genes in data file
        variant_counts = data['VariantType'].value_counts()
        gene_counts = data['GeneName'].value_counts()
        
        for variant_type, count in variant_counts.items():
            if variant_type in final_data.columns:
                final_data.loc[file_name, variant_type] = count

        for gene_name, count in gene_counts.items():
            if gene_name in final_data.columns:
                final_data.loc[file_name, gene_name] = count
        print("DONE\n")
        # Append the counts for this file to the output file
final_data.to_csv("final_data_updated.tsv", mode="a", header=False, sep='\t')

