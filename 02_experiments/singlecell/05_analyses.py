import torch
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import textwrap

import sys
sys.path.append(".")
sys.path.append('src')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda:1'

###
# load results
###

# multiDGD GO terms
# look at the dataframe with GO term analysis
go_df_1 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_small_2_biological_process.csv')
go_df_1['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large_2_biological_process.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_small_2_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df4 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large_2_molecular_function.csv')
go_df4['go type'] = 'molecular function'
go_df_dgd = pd.concat([go_df_1, go_df2, go_df3, go_df4], axis=0, ignore_index=True)

activations_dgd = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')

# Geneformer GO terms
go_df1 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_small_biological_process.csv')
go_df1['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_large_biological_process.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_small_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df4 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_large_molecular_function.csv')
go_df4['go type'] = 'molecular function'
go_df_geneformer = pd.concat([go_df1, go_df2, go_df3, go_df4], axis=0, ignore_index=True)

activations_geneformer = torch.load('03_results/reports/sae_geneformer_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')

###
# get the GO-feature matrices
###

from goatools.obo_parser import GODag
obodag = GODag("01_data/go-basic.obo")

# how many GO terms are in the feature matrix on average per feature?
go_feature_matrix_dgd = torch.zeros((len(go_df_dgd['go_name'].unique()), activations_dgd.shape[1]))
for i, go_id in enumerate(go_df_dgd['go_id'].unique()):
    for feat in go_df_dgd[go_df_dgd['go_id'] == go_id]['feature']:
        go_feature_matrix_dgd[i,feat] = 1
mtrx_go_ids_dgd = go_df_dgd['go_id'].unique()
mtrx_go_names_dgd = [obodag[x].name for x in mtrx_go_ids_dgd]
mtrx_feature_ids_dgd = torch.where(go_feature_matrix_dgd.sum(dim=0) > 0)[0]
go_feature_matrix_dgd = go_feature_matrix_dgd[:,torch.where(go_feature_matrix_dgd.sum(dim=0) > 0)[0]]
avg_go_terms_per_feature_dgd = go_feature_matrix_dgd.sum(dim=0).mean().item()
# print the mean and SEM
print(f"Average number of GO terms per feature in DGD: {avg_go_terms_per_feature_dgd:.2f} ± {(go_feature_matrix_dgd.sum(dim=0).std().item()/go_feature_matrix_dgd.shape[0]**0.5):.2f} SEM")
# what are the min and max number of GO terms per feature?
min_go_terms_per_feature_dgd = go_feature_matrix_dgd.sum(dim=0).min().item()
max_go_terms_per_feature_dgd = go_feature_matrix_dgd.sum(dim=0).max().item()
print(f"Minimum number of GO terms per feature in DGD: {min_go_terms_per_feature_dgd}")
print(f"Maximum number of GO terms per feature in DGD: {max_go_terms_per_feature_dgd}")

go_feature_matrix_geneformer = torch.zeros((len(go_df_geneformer['go_name'].unique()), activations_geneformer.shape[1]))
for i, go_id in enumerate(go_df_geneformer['go_id'].unique()):
    for feat in go_df_geneformer[go_df_geneformer['go_id'] == go_id]['feature']:
        go_feature_matrix_geneformer[i,feat] = 1
mtrx_go_ids_geneformer = go_df_geneformer['go_id'].unique()
mtrx_go_names_geneformer = [obodag[x].name for x in mtrx_go_ids_geneformer]
mtrx_feature_ids_geneformer = torch.where(go_feature_matrix_geneformer.sum(dim=0) > 0)[0]
go_feature_matrix_geneformer = go_feature_matrix_geneformer[:,torch.where(go_feature_matrix_geneformer.sum(dim=0) > 0)[0]]
avg_go_terms_per_feature_geneformer = go_feature_matrix_geneformer.sum(dim=0).mean().item()
# print the mean and SEM
print(f"Average number of GO terms per feature in Geneformer: {avg_go_terms_per_feature_geneformer:.2f} ± {(go_feature_matrix_geneformer.sum(dim=0).std().item()/go_feature_matrix_geneformer.shape[0]**0.5):.2f} SEM")
# what are the min and max number of GO terms per feature?
min_go_terms_per_feature_geneformer = go_feature_matrix_geneformer.sum(dim=0).min().item()
max_go_terms_per_feature_geneformer = go_feature_matrix_geneformer.sum(dim=0).max().item()
print(f"Minimum number of GO terms per feature in Geneformer: {min_go_terms_per_feature_geneformer}")
print(f"Maximum number of GO terms per feature in Geneformer: {max_go_terms_per_feature_geneformer}")

###
# get the shared GO terms
###
# select the shared GO terms and sort the matrices by those
shared_go_terms = set(mtrx_go_names_dgd) & set(mtrx_go_names_geneformer)
shared_go_terms = sorted(shared_go_terms)
shared_go_ids = [x for x in mtrx_go_ids_dgd if obodag[x].name in shared_go_terms]
# get a translation of where the shared GO terms are in the matrices
shared_go_ids_dgd = [np.where(mtrx_go_ids_dgd == x)[0][0] for x in shared_go_ids]
shared_go_ids_geneformer = [np.where(mtrx_go_ids_geneformer == x)[0][0] for x in shared_go_ids]
go_feature_matrix_dgd_shared = go_feature_matrix_dgd[shared_go_ids_dgd,:].T
go_feature_matrix_geneformer_shared = go_feature_matrix_geneformer[shared_go_ids_geneformer,:].T
print(go_feature_matrix_dgd_shared.shape, go_feature_matrix_geneformer_shared.shape)

###
# Calculate feature similarity
###

# compare with optimal bipartite matching
# since we have binary matrices, we can use the Jaccard index

from scipy.spatial.distance import jaccard
from scipy.optimize import linear_sum_assignment

def calculate_matrix_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """
    Calculates the similarity between two binary matrices with shared columns
    but unsorted and unpaired rows.

    The method finds the optimal matching between rows of the two matrices
    using the Jaccard similarity and the Hungarian algorithm (linear_sum_assignment),
    and then computes the average similarity of these matched pairs.

    Args:
        matrix_a (np.ndarray): The first binary matrix with shape (n, k).
                               Rows are features/entities, columns are attributes.
        matrix_b (np.ndarray): The second binary matrix with shape (m, k).
                               Rows are features/entities, columns are attributes.
                               Must have the same number of columns (k) as matrix_a.

    Returns:
        float: A similarity score between 0.0 and 1.0.
               1.0 if both matrices are empty of rows.
               0.0 if one matrix is empty and the other is not.
    """
    n_rows_a, k_cols_a = matrix_a.shape
    n_rows_b, k_cols_b = matrix_b.shape

    if k_cols_a != k_cols_b:
        raise ValueError("Matrices must have the same number of columns (k).")

    # Handle edge cases with empty matrices
    if n_rows_a == 0 and n_rows_b == 0:
        return 1.0  # Two empty sets of rows can be considered perfectly similar
    if n_rows_a == 0 or n_rows_b == 0:
        return 0.0  # One empty and one non-empty matrix are not similar

    # 1. Calculate Jaccard Similarity for all pairs of rows
    # Similarity = 1 - Jaccard Distance
    # The similarity_matrix will have shape (n_rows_a, n_rows_b)
    similarity_matrix = np.zeros((n_rows_a, n_rows_b))

    for i in range(n_rows_a):
        for j in range(n_rows_b):
            row_a = matrix_a[i, :]
            row_b = matrix_b[j, :]
            # scipy.spatial.distance.jaccard computes the Jaccard *distance*
            # For boolean arrays, if both are all zeros, distance is 0 (similarity 1)
            # This is generally the desired behavior for identical all-zero patterns.
            # Ensure inputs are boolean for jaccard if they are int 0/1.
            # If they are already float 0.0/1.0, jaccard might treat them as continuous.
            # For binary (0/1) integer arrays, casting to bool is safest.
            try:
                j_distance = jaccard(row_a.astype(bool), row_b.astype(bool))
                similarity_matrix[i, j] = 1.0 - j_distance
            except ZeroDivisionError: 
                # This can happen if both rows are all zeros and the specific jaccard 
                # implementation doesn't gracefully handle 0/0.
                # Scipy's jaccard for boolean arrays should handle this (dist=0 if both all zero).
                # However, being explicit can be good.
                # If both rows sum to 0, they are identical in their emptiness.
                if np.sum(row_a) == 0 and np.sum(row_b) == 0:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = 0.0


    # 2. Construct the Cost Matrix for the assignment problem
    # The Hungarian algorithm (linear_sum_assignment) minimizes cost.
    # Cost = 1 - Similarity
    cost_matrix = 1.0 - similarity_matrix

    # 3. Solve the Assignment Problem
    # This finds the optimal pairing of rows from A to rows from B
    # (or vice-versa depending on which has fewer rows)
    # that minimizes the total cost.
    row_ind_a, col_ind_b = linear_sum_assignment(cost_matrix)

    # 4. Calculate the Overall Matrix Similarity
    # The number of matched pairs will be min(n_rows_a, n_rows_b)
    num_matched_pairs = len(row_ind_a)

    if num_matched_pairs == 0: # Should be covered by earlier checks, but as a safeguard
        return 0.0

    # Sum of similarities for the optimal matched pairs
    sum_optimal_similarities = similarity_matrix[row_ind_a, col_ind_b].sum()

    overall_similarity = sum_optimal_similarities / num_matched_pairs

    return overall_similarity

go_feature_similarity = calculate_matrix_similarity(go_feature_matrix_dgd_shared.numpy(), go_feature_matrix_geneformer_shared.numpy())

print(f"GO feature similarity (DGD to Geneformer): {go_feature_similarity:.4f}")