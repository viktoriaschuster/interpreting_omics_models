import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import sys
sys.path.append(".")
sys.path.append('src')
from src.models.sparse_autoencoder import *
from src.visualization.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({'font.size': 14})

####################################################################################################
# load data and models
####################################################################################################

data = ad.read_h5ad("./01_data/human_bonemarrow.h5ad")

model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")
reps = model.representation.z.detach()
data = data[data.obs["train_val_test"] == "train"]
celltypes = np.unique(data.obs['cell_type'].values)

sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'

input_size = reps.shape[1]
hidden_size = 10**4
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model.load_state_dict(torch.load(sae_model_save_name+'.pt'))
sae_model.to(device)

batch_size = 128
train_data = reps.clone()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

threshold = 1e-10
unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)
avg_active_hidden_units = count_active_hidden_units(sae_model, train_loader, threshold=threshold)

reps_reconstructed, activations = sae_model(torch.tensor(reps, dtype=torch.float32).to(device))

import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.stats.multitest import multipletests

# get the model's dispersions for the test
with torch.no_grad():
    dispersion_factors = (torch.exp(model.decoder.out_modules[0].distribution.log_r).detach().cpu().numpy() + 1).flatten()

####################################################################################################
# plot
####################################################################################################

fig_dir = '03_results/figures/supplementaries/'
fig_name = 'sc_differential_expression.png'

figure_kwargs = {
    'fig_width': 8,
    'fig_height': 14,
    'font_size': 14,
    'point_size': 20,
    'line_width_background': 0.5,
    'wspace': 0.5,
    'hspace': 0.5,
}

# set up a figure with grid: 3 rows, 5 columns
fig = plt.figure(figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
gs = GridSpec(4, 1, figure=fig)
# add spacing
gs.update(wspace=figure_kwargs['wspace'], hspace=figure_kwargs['hspace'])
ax_list = []

feat = 2306

###
# subplot 1
###

ct1 = 'HSC'
ct2 = 'Normoblast'

ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device)

latents_hsc = reps[ct1_indices, :]
cov_latents_hsc = model.correction_rep.z.detach()[ct1_indices, :]
activs_hsc = activations[ct1_indices, :]
active_values_normoblast = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == 'Normoblast')[0],:].detach().cpu().numpy(), axis=0)).to(device)
activs_perturbed = activs_hsc.clone()
activs_perturbed[:, feat] = active_values_normoblast[feat]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device))
y_hsc = model.decoder(torch.cat((latents_hsc, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
y_perturb = model.decoder(torch.cat((latents_perturbed, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
library = data.obs['GEX_n_counts'].values[ct1_indices]
y_hsc = y_hsc * library.reshape(-1, 1)
y_perturb = y_perturb * library.reshape(-1, 1)
n_cells = y_hsc.shape[0]
n_genes = y_hsc.shape[1]
conditions = np.array([0] * n_cells + [1] * n_cells)
pairing = np.tile(np.arange(n_cells), 2)
p_values = []
fold_changes = []
for gene_idx in range(n_genes):
    # Combine the gene expression data for the current gene across both conditions
    gene_expression = np.concatenate([y_hsc[:, gene_idx], y_perturb[:, gene_idx]])
    # Design matrix: Intercept (ones), pairing, and condition
    X = np.column_stack([np.ones_like(conditions), pairing, conditions])
    # Fit a negative binomial model for the current gene
    glm_model = sm.GLM(gene_expression, X, family=NegativeBinomial(alpha=dispersion_factors[gene_idx]))
    result = glm_model.fit()
    # Extract p-value for the condition (perturbation effect)
    p_value = result.pvalues[2]  # The p-value for the "condition" variable
    fold_change = np.exp(result.params[2])  # Fold change is exp(beta)
    # Store results
    p_values.append(p_value)
    fold_changes.append(fold_change)
# Convert p-values and fold changes into a DataFrame
gene_p_values = pd.DataFrame({
    'gene': (data.var[data.var['modality'] == 'GEX']).index,  # Assuming you have gene names as your columns' index
    'p_value': p_values,
    'fold_change': fold_changes
})
# Adjust p-values for multiple testing using Benjamini-Hochberg correction
gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]
# Sort the results by p-value
gene_p_values = gene_p_values.sort_values(by='p_value')
# look at the genes above a certain threshold
fold_change_threshold = [0.5, 2.0]
p_value_threshold = 0.05
top_genes = gene_p_values[((gene_p_values['fold_change'] < fold_change_threshold[0]) | (gene_p_values['fold_change'] > fold_change_threshold[1])) & (gene_p_values['adj_p_value'] < p_value_threshold)]['gene']

ax_list.append(fig.add_subplot(gs[0, 0]))
sns.scatterplot(data=gene_p_values, x='fold_change', y='adj_p_value', ax=ax_list[-1], linewidth=0, alpha=0.5, label='all genes')
ax_list[-1].set_xlabel('Fold Change')
ax_list[-1].set_ylabel('Adjusted p-value')
ax_list[-1].set_title(ct1)
top_genes_indices = np.where(gene_p_values['gene'].isin(top_genes))[0]
sns.scatterplot(data=gene_p_values.iloc[top_genes_indices], x='fold_change', y='adj_p_value', color='red', ax=ax_list[-1], linewidth=0, alpha=0.5, label='significant genes')
ax_list[-1].legend(frameon=False, handletextpad=0.2)

###
# subplot 2
###

ct1 = 'Proerythroblast'
ct2 = 'Normoblast'

ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device) * 1.5

latents_hsc = reps[ct1_indices, :]
cov_latents_hsc = model.correction_rep.z.detach()[ct1_indices, :]
activs_hsc = activations[ct1_indices, :]
active_values_normoblast = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == 'Normoblast')[0],:].detach().cpu().numpy(), axis=0)).to(device)
activs_perturbed = activs_hsc.clone()
activs_perturbed[:, feat] = active_values_normoblast[feat]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device))
y_hsc = model.decoder(torch.cat((latents_hsc, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
y_perturb = model.decoder(torch.cat((latents_perturbed, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
library = data.obs['GEX_n_counts'].values[ct1_indices]
y_hsc = y_hsc * library.reshape(-1, 1)
y_perturb = y_perturb * library.reshape(-1, 1)
n_cells = y_hsc.shape[0]
n_genes = y_hsc.shape[1]
conditions = np.array([0] * n_cells + [1] * n_cells)
pairing = np.tile(np.arange(n_cells), 2)
p_values = []
fold_changes = []
for gene_idx in range(n_genes):
    # Combine the gene expression data for the current gene across both conditions
    gene_expression = np.concatenate([y_hsc[:, gene_idx], y_perturb[:, gene_idx]])
    # Design matrix: Intercept (ones), pairing, and condition
    X = np.column_stack([np.ones_like(conditions), pairing, conditions])
    # Fit a negative binomial model for the current gene
    glm_model = sm.GLM(gene_expression, X, family=NegativeBinomial(alpha=dispersion_factors[gene_idx]))
    result = glm_model.fit()
    # Extract p-value for the condition (perturbation effect)
    p_value = result.pvalues[2]  # The p-value for the "condition" variable
    fold_change = np.exp(result.params[2])  # Fold change is exp(beta)
    # Store results
    p_values.append(p_value)
    fold_changes.append(fold_change)
# Convert p-values and fold changes into a DataFrame
gene_p_values = pd.DataFrame({
    'gene': (data.var[data.var['modality'] == 'GEX']).index,  # Assuming you have gene names as your columns' index
    'p_value': p_values,
    'fold_change': fold_changes
})
# Adjust p-values for multiple testing using Benjamini-Hochberg correction
gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]
# Sort the results by p-value
gene_p_values = gene_p_values.sort_values(by='p_value')
# look at the genes above a certain threshold
fold_change_threshold = [0.5, 2.0]
p_value_threshold = 0.05
top_genes = gene_p_values[((gene_p_values['fold_change'] < fold_change_threshold[0]) | (gene_p_values['fold_change'] > fold_change_threshold[1])) & (gene_p_values['adj_p_value'] < p_value_threshold)]['gene']

ax_list.append(fig.add_subplot(gs[1, 0]))
sns.scatterplot(data=gene_p_values, x='fold_change', y='adj_p_value', ax=ax_list[-1])
ax_list[-1].set_xlabel('Fold Change')
ax_list[-1].set_ylabel('Adjusted p-value')
ax_list[-1].set_title(ct1)
top_genes_indices = np.where(gene_p_values['gene'].isin(top_genes))[0]
sns.scatterplot(data=gene_p_values.iloc[top_genes_indices], x='fold_change', y='adj_p_value', color='red', ax=ax_list[-1])
ax_list[-1].legend().remove()

###
# subplot 3
###

ct1 = 'NK'

ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device)

latents_hsc = reps[ct1_indices, :]
cov_latents_hsc = model.correction_rep.z.detach()[ct1_indices, :]
activs_hsc = activations[ct1_indices, :]
active_values_normoblast = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == 'Normoblast')[0],:].detach().cpu().numpy(), axis=0)).to(device)
activs_perturbed = activs_hsc.clone()
activs_perturbed[:, feat] = active_values_normoblast[feat]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device))
y_hsc = model.decoder(torch.cat((latents_hsc, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
y_perturb = model.decoder(torch.cat((latents_perturbed, cov_latents_hsc), dim=1))[0].detach().cpu().numpy()
library = data.obs['GEX_n_counts'].values[ct1_indices]
y_hsc = y_hsc * library.reshape(-1, 1)
y_perturb = y_perturb * library.reshape(-1, 1)
n_cells = y_hsc.shape[0]
n_genes = y_hsc.shape[1]
conditions = np.array([0] * n_cells + [1] * n_cells)
pairing = np.tile(np.arange(n_cells), 2)
p_values = []
fold_changes = []
for gene_idx in range(n_genes):
    # Combine the gene expression data for the current gene across both conditions
    gene_expression = np.concatenate([y_hsc[:, gene_idx], y_perturb[:, gene_idx]])
    # Design matrix: Intercept (ones), pairing, and condition
    X = np.column_stack([np.ones_like(conditions), pairing, conditions])
    # Fit a negative binomial model for the current gene
    glm_model = sm.GLM(gene_expression, X, family=NegativeBinomial(alpha=dispersion_factors[gene_idx]))
    result = glm_model.fit()
    # Extract p-value for the condition (perturbation effect)
    p_value = result.pvalues[2]  # The p-value for the "condition" variable
    fold_change = np.exp(result.params[2])  # Fold change is exp(beta)
    # Store results
    p_values.append(p_value)
    fold_changes.append(fold_change)
# Convert p-values and fold changes into a DataFrame
gene_p_values = pd.DataFrame({
    'gene': (data.var[data.var['modality'] == 'GEX']).index,  # Assuming you have gene names as your columns' index
    'p_value': p_values,
    'fold_change': fold_changes
})
# Adjust p-values for multiple testing using Benjamini-Hochberg correction
gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]
# Sort the results by p-value
gene_p_values = gene_p_values.sort_values(by='p_value')
# look at the genes above a certain threshold
fold_change_threshold = [0.5, 2.0]
p_value_threshold = 0.05
top_genes = gene_p_values[((gene_p_values['fold_change'] < fold_change_threshold[0]) | (gene_p_values['fold_change'] > fold_change_threshold[1])) & (gene_p_values['adj_p_value'] < p_value_threshold)]['gene']

ax_list.append(fig.add_subplot(gs[2, 0]))
sns.scatterplot(data=gene_p_values, x='fold_change', y='adj_p_value', ax=ax_list[-1])
ax_list[-1].set_xlabel('Fold Change')
ax_list[-1].set_ylabel('Adjusted p-value')
ax_list[-1].set_title(ct1)
top_genes_indices = np.where(gene_p_values['gene'].isin(top_genes))[0]
sns.scatterplot(data=gene_p_values.iloc[top_genes_indices], x='fold_change', y='adj_p_value', color='red', ax=ax_list[-1])
ax_list[-1].legend().remove()

###
# subplot 4
###

cd8_indices = np.where(data.obs['cell_type'] == 'CD8+ T')[0]

activs_feat = activations[cd8_indices, feat].detach().cpu().numpy()
# kick out the zeros (this is where the feature is not active, and this screws up the quantiles)
activs_feat = activs_feat[activs_feat > 0]

quantile_steps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
quantiles = np.quantile(activs_feat, quantile_steps)

i = 0
j = -1
thresholds = [quantiles[i], quantiles[j]]
print(f'Thresholds: {thresholds[0]} ({quantile_steps[i]} quantile) and {thresholds[1]} ({quantile_steps[j]} quantile)')

# get indices of CD8+T cells and indices of where the feature is active (> 0.01)
ct_indices = np.where(data.obs['cell_type'] == 'CD8+ T')[0]
feat_indices_pos = np.where(activations.detach().cpu()[:, feat] > thresholds[1])[0]
feat_indices_neg = np.where(activations.detach().cpu()[:, feat] < thresholds[0])[0]

ct_pos_indices = np.intersect1d(ct_indices, feat_indices_pos)
#ct_neg_indices = np.setdiff1d(ct_indices, ct_pos_indices)
ct_neg_indices = np.intersect1d(ct_indices, feat_indices_neg)

latents_pos = reps[ct_pos_indices, :].cpu().numpy()
latents_neg = reps[ct_neg_indices, :].cpu().numpy()
activs_pos = activations[ct_pos_indices, :]
activs_neg = activations[ct_neg_indices, :]

# get all necessary extras for predictions (model predctions are better than just raw data)
library_pos = data.obs['GEX_n_counts'].values[ct_pos_indices]
library_neg = data.obs['GEX_n_counts'].values[ct_neg_indices]
cov_latents_pos = model.correction_rep.z.detach()[ct_pos_indices, :]
cov_latents_neg = model.correction_rep.z.detach()[ct_neg_indices, :]

# make predictions
y_pos = model.decoder(torch.cat((torch.tensor(latents_pos).to(device), cov_latents_pos), dim=1))[0].detach().cpu().numpy()
y_neg = model.decoder(torch.cat((torch.tensor(latents_neg).to(device), cov_latents_neg), dim=1))[0].detach().cpu().numpy()
# rescale
y_pos = y_pos * library_pos.reshape(-1, 1)
y_neg = y_neg * library_neg.reshape(-1, 1)

from scipy import stats
from statsmodels.stats.multitest import multipletests

fold_changes = y_pos.mean(axis=0) / y_neg.mean(axis=0)

# Perform unpaired t-tests for each gene
p_values = []
for gene in range(y_pos.shape[1]):
    t_stat, p_val = stats.ttest_ind(y_pos[:,gene], y_neg[:,gene], equal_var=False)  # Unequal variance (Welch's t-test)
    p_values.append(p_val)

# Convert the p-values into a numpy array for further processing
p_values = np.array(p_values)

# Adjust p-values using Benjamini-Hochberg correction (FDR)
adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

# Convert p-values and fold changes into a DataFrame
gene_p_values = pd.DataFrame({
    'gene': (data.var[data.var['modality'] == 'GEX']).index,  # Assuming you have gene names as your columns' index
    'p_value': p_values,
    'fold_change': fold_changes,
    'adj_p_value': adjusted_p_values
})

# Sort the results by p-value
gene_p_values = gene_p_values.sort_values(by='p_value')

fold_change_threshold = [0.5, 2.0]
p_value_threshold = 0.05
top_genes = gene_p_values[((gene_p_values['fold_change'] < fold_change_threshold[0]) | (gene_p_values['fold_change'] > fold_change_threshold[1])) & (gene_p_values['adj_p_value'] < p_value_threshold)]['gene']

ax_list.append(fig.add_subplot(gs[3, 0]))
sns.scatterplot(data=gene_p_values, x='fold_change', y='adj_p_value', ax=ax_list[-1])
ax_list[-1].set_xlabel('Fold Change')
ax_list[-1].set_ylabel('Adjusted p-value')
ax_list[-1].set_title('CD8+T')
top_genes_indices = np.where(gene_p_values['gene'].isin(top_genes))[0]
sns.scatterplot(data=gene_p_values.iloc[top_genes_indices], x='fold_change', y='adj_p_value', color='red', ax=ax_list[-1])
ax_list[-1].legend().remove()

fig.savefig(fig_dir+fig_name, bbox_inches='tight', dpi=300)