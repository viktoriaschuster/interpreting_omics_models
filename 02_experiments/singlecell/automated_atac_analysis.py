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
library = data.obs['ATAC_nCount_peaks'].values
data_peak_names = (data.var[data.var['modality'] == 'ATAC']).index
del data
gc.collect()
# get the model's dispersions for the DEG test
with torch.no_grad():
    dispersion_factors = (torch.exp(model.decoder.out_modules[1].distribution.log_r).detach().cpu().numpy() + 1).flatten()

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
nonzero_feats = torch.where(n_zeros_per_feature > activations.shape[0] - 100)[0]
active_feature_ids = torch.Tensor(list(set(active_feature_ids.numpy()).intersection(set(nonzero_feats.numpy())))).long()
print(f"Using {len(active_feature_ids)} features")

##########################
# functions
##########################

def binomial_test(n_study, n_hit, n_c, n):
    p_c = n_c / n # this is the expected probability of a hit
    expected = n_study * p_c
    over_under = '+' if n_hit > (expected) else '-'
    if n_hit == 0:
        return 1, expected, over_under, 0, 0
    fold_enrichment = n_hit / (n_study * p_c)
    fdr = (n_study - n_hit) / n_study
    results = stats.binomtest(n_hit, n_study, p_c)
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

##########################
# analysis
##########################

percentile = 99

###
# calc all sample predictions once here
###
chunk_size = 1000
n_chunks = math.ceil(reps.shape[0] / chunk_size)
y = torch.zeros(reps.shape[0], len(data_peak_names))
for i in range(n_chunks):
    y[i*chunk_size:(i+1)*chunk_size] = model.decoder(torch.cat((reps[i*chunk_size:(i+1)*chunk_size], model.correction_rep.z[i*chunk_size:(i+1)*chunk_size]), dim=1))[1].detach().cpu()
y = y * library.reshape(-1, 1)
y = y.numpy()
print(f"Predicted all samples of shape {y.shape}")

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
    # chromosome enrichment analysis
    ###
    pos_chromatin_avg, pos_chromatin_se = np.mean(y_pos, axis=0), np.std(y_pos, axis=0) / math.sqrt(y_pos.shape[0])
    neg_chromatin_avg, neg_chromatin_se = np.mean(y_neg, axis=0), np.std(y_neg, axis=0) / math.sqrt(y_neg.shape[0])
    mean_diffs = np.abs(pos_chromatin_avg - neg_chromatin_avg)
    total_ses = pos_chromatin_se + neg_chromatin_se
    conf_95 = np.where((mean_diffs - 1.96*total_ses) > 0)[0]

    peak_diff_df_0 = pd.DataFrame({
        'chromosome': [x.split('-')[0] for x in data_peak_names],
        'significant': (mean_diffs - 1.96*total_ses) > 0
    })
    peak_diff_df = peak_diff_df_0.copy()
    chr_counts = peak_diff_df['chromosome'].value_counts()
    peak_diff_df = peak_diff_df.groupby('chromosome').sum()
    peak_diff_df['count'] = chr_counts

    # binomial test
    try:
        peak_diff_df['binom_pval'], peak_diff_df['expected'], peak_diff_df['over_under'], peak_diff_df['fold_enrichment'], peak_diff_df['fdr'] = zip(*peak_diff_df.apply(lambda x: binomial_test(len(conf_95), x['significant'], x['count'], len(data_peak_names)), axis=1))
    except:
        print(peak_diff_df['significant'].values)
        exit()
    
    # now MWU test
    mw_pvals, effect_sizes = [], []
    for chrom in peak_diff_df.index:
        chrom_indices = np.where(peak_diff_df_0['chromosome'] == chrom)[0]

        df_temp = peak_diff_df_0.copy()
        df_temp['in_set'] = [1 if x == chrom else 0 for x in df_temp['chromosome']]
        # sort by mean_diffs
        df_temp = df_temp.sort_values(by='significant', ascending=True)
        df_temp['rank'] = range(1,len(df_temp)+1)
        z_score, mw_pval, effect_size = mann_whitney_u_test(
            df_temp['rank'].values, 
            df_temp['in_set'].values
        )
        mw_pvals.append(mw_pval)
        effect_sizes.append(effect_size)
        #peak_diff_df.loc[chrom, 'mw_pval'] = mw_pval
        #peak_diff_df.loc[chrom, 'effect_size'] = effect_size
    peak_diff_df['mw_pval'] = mw_pvals
    peak_diff_df['effect_size'] = effect_sizes
    
    peak_diff_df['feature'] = feat.item()

    peak_diff_df = peak_diff_df[(peak_diff_df['binom_pval'] < 0.05) | (peak_diff_df['mw_pval'] < 0.05)]

    if len(peak_diff_df) == 0:
        continue

    # save the data
    file_name = '03_results/reports/sc_dgd_sae_chromatin_analysis.csv'
    if not os.path.exists(file_name):
        peak_diff_df.to_csv(file_name, index=False)
    else:
        peak_diff_df.to_csv(file_name, mode='a', header=False, index=False)

print('done')

