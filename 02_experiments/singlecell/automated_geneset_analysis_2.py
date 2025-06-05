import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
import math
import os
import gc
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
import multiprocessing as mp
import tqdm
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

import sys
sys.path.append(".")
sys.path.append('src')

dev_id = 1
device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
#go_category = 'biological_process'
go_category = 'molecular_function'

##########################
# load model and data
##########################

if torch.cuda.is_available():
    print(f"Using GPU: {device}")
    data_dir = '/home/vschuste/data/singlecell/'
else:
    data_dir = '/Users/vschuste/Documents/work/data/singlecell/'

data = ad.read_h5ad(data_dir+'human_bonemarrow.h5ad')

model = multiDGD.DGD.load(data=data, save_dir='./03_results/models/', model_name='human_bonemarrow_l20_h2-3_test50e').to(device)
data = data[data.obs["train_val_test"] == "train"]
library = data.obs['GEX_n_counts'].values
data_gene_names = (data.var[data.var['modality'] == 'GEX']).index
data_gene_ids = data.var[data.var['modality'] == 'GEX']['gene_id'].values
del data
gc.collect()
# get the model's dispersions for the DEG test
with torch.no_grad():
    dispersion_factors = (torch.exp(model.decoder.out_modules[0].distribution.log_r).detach().cpu().numpy() + 1).flatten()

reps = model.representation.z.detach()

# load the SAE model and activations

import torch.nn as nn

class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'

input_size = reps.shape[1]
hidden_size = 10**4
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model.load_state_dict(torch.load(sae_model_save_name+'.pt'))
sae_model.to(device)

activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
active_feature_ids = torch.where(activations.sum(dim=0) > 0)[0]

# also exclude features with less than 100 samples activated
n_zeros_per_feature = (activations == 0).sum(dim=0)
nonzero_feats = torch.where(n_zeros_per_feature < (activations.shape[0] - 1))[0]
active_feature_ids = torch.Tensor(list(set(active_feature_ids.numpy()).intersection(set(nonzero_feats.numpy())))).long()
print(f"Using {len(active_feature_ids)} features")

##########################
# functions
##########################

def run_nb_model(y_pos, y_neg, gene_idx, conditions, pairing):
    # Combine the gene expression data for the current gene across both conditions
    gene_expression = np.concatenate([y_pos[:, gene_idx], y_neg[:, gene_idx]])

    # Design matrix: Intercept (ones), pairing, and condition
    X = np.column_stack([np.ones_like(conditions), pairing, conditions])

    # Fit a negative binomial model for the current gene
    glm_model = sm.GLM(gene_expression, X, family=NegativeBinomial(alpha=dispersion_factors[gene_idx]))
    result = glm_model.fit()

    # Extract p-value for the condition (perturbation effect)
    p_value = result.pvalues[2]  # The p-value for the "condition" variable
    fold_change = np.exp(result.params[2])  # Fold change is exp(beta)

    return p_value, fold_change

def DEG_analysis_unpaired(y_pos, y_neg, gene_names):

    fold_changes = y_pos.mean(axis=0) / y_neg.mean(axis=0)

    # Perform unpaired t-tests for each gene
    pool = mp.Pool(mp.cpu_count())
    _, p_values = zip(*pool.starmap(stats.ttest_ind, [(y_pos[:, gene], y_neg[:, gene], 0, False) for gene in range(y_pos.shape[1])]))
    #p_values = []
    #for gene in range(y_pos.shape[1]):
    #    t_stat, p_val = stats.ttest_ind(y_pos[:,gene], y_neg[:,gene], equal_var=False)  # Unequal variance (Welch's t-test)
    #    p_values.append(p_val)

    # Convert the p-values into a numpy array for further processing
    p_values = np.array(p_values)

    # Convert p-values and fold changes into a DataFrame
    gene_p_values = pd.DataFrame({
        'gene': gene_names,  # Assuming you have gene names as your columns' index
        'p_value': p_values,
        'fold_change': fold_changes
    })

    # Adjust p-values for multiple testing using Benjamini-Hochberg correction
    gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]

    # Sort the results by p-value
    gene_p_values = gene_p_values.sort_values(by='p_value')

    return gene_p_values

def run_enrichment(ranks, in_set, n_r, n_h, n, epsilon, i):
    p_hit = (sum((ranks[:i])[in_set[:i]]) + epsilon) / n_r
    p_miss = (len((ranks[:i])[~in_set[:i]]) + epsilon) / (n - n_h)
    return abs(p_hit - p_miss)

def binomial_test(n_study, n_hit, n_c, n):
    p_c = n_c / n # this is the expected probability of a hit
    results = stats.binomtest(n_hit, n_study, p_c)
    over_under = '+' if n_hit > (n_study * p_c) else '-'
    fold_enrichment = n_hit / (n_study * p_c)
    fdr = (n_study - n_hit) / n_study
    expected = n_study * p_c
    return results.pvalue, expected, over_under, fold_enrichment, fdr

def mann_whitney_u_test(ranks, in_set):
    # s stands for set, t for total
    n_s = sum(in_set)
    # according to wikipedia
    n_t = len(ranks[~in_set])
    r_s = sum(ranks[in_set])
    r_t = sum(ranks[~in_set])
    u_s = n_s * n_t + ((n_s * (n_s + 1)) / 2) - r_s
    u_t = n_s * n_t + ((n_t * (n_t + 1)) / 2) - r_t
    u = min(u_s, u_t)
    z_score = (u - (n_s * n_t / 2)) / math.sqrt(n_s * n_t * (n_s + n_t + 1) / 12)
    p_value = stats.norm.cdf(z_score)
    effect_size = u_s / (n_s * n_t)
    return z_score, p_value, effect_size

def go_analysis(gene_df, go_id, ref_data, p_threshold=1e-5, fold_change_threshold=None):
    pos_ids = ref_data.X[go_id, :].indices
    gene_df['in_set'] = [gene in pos_ids for gene in gene_df['ref_idx']]

    gene_df_selected = gene_df[(gene_df['adj_p_value'] < p_threshold)]
    if fold_change_threshold is not None:
        gene_df_selected = gene_df_selected[(gene_df_selected['fold_change'] > fold_change_threshold) | (gene_df_selected['fold_change'] < 1/fold_change_threshold)]
    if len(gene_df_selected) > 0:
        ###
        # binomial test
        ###
        hit_positions = gene_df_selected['ref_idx'].values
        n_hits = sum(gene_df_selected['in_set'])
        try:
            binom_pval, binom_expected, binom_direction, binom_fold, binom_fdr = binomial_test(
                len(gene_df_selected),
                sum(gene_df_selected['in_set']),
                len(pos_ids),
                len(gene_df)
            )
        except:
            print('problem with binom test', n_hits, len(gene_df_selected))
            binom_pval = 1.0
            binom_expected = None
            binom_direction = None
            binom_fold = None
            binom_fdr = None
    else:
        hit_positions = []
        n_hits = 0
        binom_pval = 1.0
        binom_expected = None
        binom_direction = None
        binom_fold = None
        binom_fdr = None
    
    ###
    # Mann-Whitney U test
    ###
    z_score, mw_pval, effect_size = mann_whitney_u_test(
        gene_df['rank'].values, 
        gene_df['in_set'].values
    )
    
    return n_hits, binom_expected, binom_pval, binom_direction, binom_fold, binom_fdr, z_score, mw_pval, effect_size, hit_positions

##########################
# analysis
##########################

percentile = 99
min_genes = 10
max_genes = 500
p_value_threshold = 1e-5
fold_change_threshold = 2.0

###
# preparing GO term analysis
###
obodag = GODag("01_data/go-basic.obo")
ogaf = GafReader("01_data/goa_human.gaf")
ns2assc = ogaf.get_ns2assc()
prot2ensembl = pd.read_csv('01_data/protname2ensembl.tsv', sep='\t')
df_go_levels = pd.read_csv('01_data/go_term_levels.tsv', sep='\t')
adata_go = ad.read_h5ad('01_data/go_gene_matrix.h5ad')
adata_go.var['name'] = data_gene_names

# get the ids that are within a usable range of associated genes and are biological processes
go_bp = [go_id for go_id in obodag.keys() if obodag[go_id].namespace == go_category]
# get the ids that are within this range
go_ids_filtered = np.where((adata_go.X.sum(axis=1) >= min_genes) & (adata_go.X.sum(axis=1) <= max_genes))[0]
print(f"There are {len(go_ids_filtered)} go terms with at least {min_genes} gene associated")
adata_go_filtered = adata_go[go_ids_filtered, :]
# next further filter by the go_bp terms
go_ids_filtered_bp = np.where(adata_go_filtered.obs['go_id'].isin(go_bp))[0]
print(f"There are {len(go_ids_filtered_bp)} go terms that are {go_category}")
adata_go_filtered = adata_go_filtered[go_ids_filtered_bp, :]

###
# calc all sample predictions once here
###
chunk_size = 1000
n_chunks = math.ceil(reps.shape[0] / chunk_size)
y = torch.zeros(reps.shape[0], len(data_gene_names))
for i in range(n_chunks):
    y[i*chunk_size:(i+1)*chunk_size] = model.decoder(torch.cat((reps[i*chunk_size:(i+1)*chunk_size], model.correction_rep.z[i*chunk_size:(i+1)*chunk_size]), dim=1))[0].detach().cpu()
y = y * library.reshape(-1, 1)
y = y.numpy()
print(f"Predicted all samples of shape {y.shape}")

###
# prep output structures
###
gene_hits_per_feature = np.zeros((len(active_feature_ids), adata_go_filtered.X.shape[1]))

# run the analysis
# modify the tqdm message
pbar = tqdm.tqdm(range(len(active_feature_ids)))
for i in pbar:
    feat = active_feature_ids[i]

    ###
    # create feature-specific sample sets for DEG
    ###

    # get the top 1% of the feature
    top5pct = np.percentile(activations[feat].numpy(), percentile)
    top5pct_indices = torch.where(activations[feat] > top5pct)[0]
    # get the reps
    #reps_top5pct = reps[top5pct_indices, :].clone()
    #cov_reps_top5pct = model.correction_rep.z[top5pct_indices, :].clone()
    # get the positive predictions
    #y_pos = model.decoder(torch.cat((reps_top5pct, cov_reps_top5pct), dim=1))[0].detach().cpu().numpy()
    #y_pos = y_pos * library[top5pct_indices].reshape(-1, 1)
    y_pos = y[top5pct_indices, :]

    # get the bottom of the feature
    bottom_indices = torch.where(activations[feat] == 0.0)[0]
    #reps_bottom = reps[bottom_indices, :].clone()
    #cov_reps_bottom = model.correction_rep.z[bottom_indices, :].clone()
    # now get the model predictions
    #y_neg = model.decoder(torch.cat((reps_bottom, cov_reps_bottom), dim=1))[0].detach().cpu().numpy()
    #y_neg = y_neg * library[bottom_indices].reshape(-1, 1)
    y_neg = y[bottom_indices, :]


    ###
    # DEG analysis
    ###
    gene_p_values = DEG_analysis_unpaired(y_pos, y_neg, data_gene_names)
    gene_p_values_ranked = gene_p_values.sort_values(by='fold_change', ascending=True)
    gene_p_values_ranked['rank'] = range(1, len(gene_p_values_ranked)+1)
    # now sort them by the adata_go matrix
    gene_p_values_ranked['ref_idx'] = [list(adata_go.var['name']).index(gene) for gene in gene_p_values_ranked['gene']]
    max_fold_change = gene_p_values_ranked['fold_change'].max()
    min_p_value = gene_p_values_ranked['adj_p_value'].min()
    # adjust the thresholds for GO analysis to prevent empty lists for binomial analysis
    current_p_threshold = p_value_threshold
    if min_p_value > p_value_threshold:
        current_p_threshold = 0.05
    current_fold_change_threshold = fold_change_threshold
    if max_fold_change < fold_change_threshold:
        current_fold_change_threshold = None
    
    gene_df_temp = gene_p_values_ranked[(gene_p_values_ranked['adj_p_value'] < current_p_threshold)]
    if current_fold_change_threshold is not None:
        gene_df_temp = gene_df_temp[(gene_df_temp['fold_change'] > current_fold_change_threshold) | (gene_df_temp['fold_change'] < 1/current_fold_change_threshold)]
    n_top_genes = len(gene_df_temp)
    pbar.set_description(f"{i}: Feat {feat} w {n_top_genes} top genes")
    
    ###
    # GO analysis
    ###

    #for go_idx in tqdm.tqdm(range(adata_go_filtered.X.shape[0])):
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(go_analysis, [(gene_p_values_ranked, go_idx, adata_go_filtered, current_p_threshold, current_fold_change_threshold) for go_idx in range(adata_go_filtered.X.shape[0])])
    #for go_idx in range(adata_go_filtered.X.shape[0]):
    #    # print the go term and name and enrichment score
    #    results_temp = go_analysis(gene_p_values_ranked, go_idx, adata_go_filtered, current_p_threshold, current_fold_change_threshold)
    #    for j, key in enumerate(results.keys()):
    #        results[key].append(results_temp[j])
    feat_pos = np.where(active_feature_ids == feat)[0][0]
    gene_hits_per_feature[feat_pos,results[-1][-1]] = 1

    # make the results a dataframe
    result_columns = ['n_hits', 'expected', 'binom_pval', 'binom_direction', 'binom_fold_change', 'fdr', 'z_score', 'mw_pval', 'effect_size']
    results_df = pd.DataFrame(results)
    results_df = results_df.iloc[:, :-1]
    results_df.columns = result_columns
    results_df['go_id'] = adata_go_filtered.obs['go_id'].values
    results_df['go_name'] = adata_go_filtered.obs['go_name'].values
    results_df['go_level'] = adata_go_filtered.obs['go_level'].values
    results_df = results_df[(results_df['binom_pval'] < 0.05) & (results_df['mw_pval'] < 0.05)].sort_values(by='binom_fold_change', ascending=False)
    results_df['feature'] = feat.item()
    results_df['p_threshold'] = current_p_threshold
    results_df['fold_change_threshold'] = current_fold_change_threshold
    # save the data
    file_name = '03_results/reports/sc_dgd_sae_go_analysis_'+go_category+'.csv'
    if not os.path.exists(file_name):
        results_df.to_csv(file_name, index=False)
    else:
        results_df.to_csv(file_name, mode='a', header=False, index=False)

# save the gene hits per feature
np.save('03_results/reports/sc_dgd_sae_gene_hits_per_feature_'+go_category+'.npy', gene_hits_per_feature)

print('done')

