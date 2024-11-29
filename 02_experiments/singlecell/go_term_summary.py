import sys
sys.path.append(".")
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
import anndata as ad

# include command line arguments
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--metric', type=str, default='cosine', help='cosine, pearson, spearman')
parser.add_argument('--go-level', type=int, default=1, help='GO term level')

args = parser.parse_args()
metric = args.metric
go_level = args.go_level

##########################
# load data
##########################
# activations
activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
# metric
if metric == 'cosine':
    metric_matrix = torch.load('03_results/reports/sc_cosine_similarity_matrix.pt', weights_only=False)
    save_folder = 'cos_sim'
elif metric == 'pearson':
    metric_matrix = torch.load('03_results/reports/sc_pearson_correlation_matrix.pt', weights_only=False)
    # replace nan values with 0
    metric_matrix[torch.isnan(metric_matrix)] = 0
    save_folder = 'pearson'
elif metric == 'spearman':
    metric_matrix = torch.load('03_results/reports/sc_spearman_correlation_matrix.pt', weights_only=False)
    save_folder = 'spearman'
else:
    raise ValueError('Invalid metric')
adata_go = ad.read('01_data/go_gene_matrix.h5ad')
# make sure this go level exists
if go_level not in adata_go.obs['go_level'].unique():
    raise ValueError('Invalid go level')
else:
    # print how many go terms are in this level
    print('Number of GO terms in level {}: {}'.format(go_level, sum(adata_go.obs['go_level'] == go_level)))
    adata_go = adata_go[adata_go.obs['go_level'] == go_level]

##########################
# helper metrics and functions
##########################
# get the indices of the dead features
dead_features = torch.where(torch.sum(activations, dim=0) == 0)[0]
active_features = torch.where(torch.sum(activations, dim=0) != 0)[0]
# quantile thresholds
quantiles = (0.05, 0.95)
non_zero_values = metric_matrix[active_features,:].flatten()
non_zero_values = non_zero_values[non_zero_values > 0]
thresholds = [np.quantile(non_zero_values.numpy(), q) for q in quantiles]

def go_feature_analysis(go_idx, sim_mtrx, threshold):
    # get the genes involved in this go term
    genes = torch.tensor(adata_go.X[go_idx,:].todense()).flatten().bool()
    if torch.sum(genes) == 0:
        return None, None, None
    
    # get the similarity metrics for these genes
    go_sim_mtrx = sim_mtrx[:,genes].clone()
    
    # get the fraction of go genes with similarity above threshold per feature
    frac_per_feature = (go_sim_mtrx > threshold).sum(dim=1).float() / go_sim_mtrx.shape[1]

    # find the lowest value and compute quantile
    # do this relative, so normalized to just look at where the genes are
    sim_mtrx_norm = sim_mtrx.clone() / torch.max(sim_mtrx, dim=1).values.unsqueeze(1)
    # replace nan values with 0
    sim_mtrx_norm[torch.isnan(sim_mtrx_norm)] = 0
    lowest_values = torch.min(sim_mtrx_norm[:,genes], dim=1).values
    feature_quantiles = torch.tensor([torch.quantile(sim_mtrx_norm[i,:], lowest_values[i]) for i in range(sim_mtrx_norm.shape[0])])

    return sum(genes).item(), frac_per_feature, feature_quantiles

##########################
# summary metrics
##########################

n_genes = []
best_fractions = []
best_fractions_idx = []
quantiles_of_best_fractions = []
best_quantiles = []
best_quantiles_idx = []
fractions_of_best_quantiles = []

import tqdm
for go_idx in tqdm.tqdm(range(adata_go.shape[0])):
    go_term_n_genes, go_term_feature_fractions, go_term_feature_quantiles = go_feature_analysis(go_idx, metric_matrix, thresholds[1])
    if go_term_n_genes is None:
        n_genes.append(0)
        best_fractions.append(0)
        best_fractions_idx.append(0)
        quantiles_of_best_fractions.append(0)
        best_quantiles.append(0)
        best_quantiles_idx.append(0)
        fractions_of_best_quantiles.append(0)
    else:
        n_genes.append(go_term_n_genes)
        best_fractions.append(go_term_feature_fractions.max().item())
        best_fractions_idx.append(go_term_feature_fractions.argmax().item())
        quantiles_of_best_fractions.append(go_term_feature_quantiles[go_term_feature_fractions.argmax()].item())
        best_quantiles.append(go_term_feature_quantiles.max().item())
        best_quantiles_idx.append(go_term_feature_quantiles.argmax().item())
        fractions_of_best_quantiles.append(go_term_feature_fractions[go_term_feature_quantiles.argmax()].item())

df_go_metrics = pd.DataFrame({
    'go_id': adata_go.obs['go_id'],
    'go_name': adata_go.obs['go_name'],
    'go_level': adata_go.obs['go_level'],
    'n_genes': n_genes,
    'best_fraction': best_fractions,
    'best_fraction_idx': best_fractions_idx,
    'quantile_of_best_fraction': quantiles_of_best_fractions,
    'best_quantile': best_quantiles,
    'best_quantile_idx': best_quantiles_idx,
    'fraction_of_best_quantile': fractions_of_best_quantiles
    })
df_go_metrics.to_csv('03_results/reports/{}/go_term_summary_{}_golevel{}.csv'.format(save_folder, metric, go_level), index=False)