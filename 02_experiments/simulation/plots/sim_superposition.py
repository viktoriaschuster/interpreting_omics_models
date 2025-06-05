import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

###
# general plotting stuff
###

fontsize = 14
# set the fontsize
plt.rcParams.update({'font.size': fontsize})
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
colors = ['#b14c8d','#0044ad','#5176e8','#82A0FD','#417a78','#c5a862']
#variables = ['Y', 'X₂', 'X₁', 'X₀', 'A', 'B']
variables = ['Y', "X''", "X'", 'X', 'A', 'B']

###
# get the superposition fits
###

file_dir = '03_results/models/'
# get all subdirectories starting with 'largesim_ae'
subdirs = [x for x in os.listdir(file_dir) if os.path.isdir(file_dir + x) and x.startswith('largesim_ae')]

r_squares = []
for subdir in subdirs:
    # get the r2 score
    try:
        r2 = pd.read_csv(file_dir + subdir + '/r_squares.csv')
        # give it column names
        r2.columns = ['variable', 'R^2']
        
        # get the latent dimension
        r2['latent'] = int(subdir.split('_latent-')[-1].split('_')[0])
        # get the depth
        r2['depth'] = int(subdir.split('_depth-')[-1].split('_')[0])
        # get the width
        r2['width'] = subdir.split('_width-')[-1].split('_')[0]
        # get the seed
        r2['seed'] = int(subdir.split('_seed-')[-1].split('_')[0])
        # create a name
        r2['name'] = subdir
        r2['name_short'] = subdir.split('_latent-')[-1].split('_')[0] + '-' + subdir.split('_depth-')[-1].split('_')[0] + '-' + ('n' if subdir.split('_width-')[-1].split('_')[0] == 'narrow' else 'w')

        # also load the history and add the validation loss
        history = pd.read_csv(file_dir + subdir + '/history.csv')
        r2['val_loss'] = history['val_loss'].iloc[-1]
        
        r_squares.append(r2)
    except:
        print('Error with ' + subdir)
        continue
# concatenate all r2 scores
r_squares = pd.concat(r_squares)
r_squares = r_squares.reset_index(drop=True)
# change the names of the variables to the last part of the string
r_squares['variable'] = r_squares['variable'].str.split('_').str[-1]
# sort the df first by latent dimension, then by depth, then by width
r_squares = r_squares.sort_values(by=['latent', 'depth', 'width', 'seed'])

r_squares.head()

###
# plot
###

# plot val loss against R2
r_squares['best_arch'] = ['2-w' if '2-w' in x else 'other' for x in r_squares['name_short']]
# change the variable names
r_squares['variable'] = r_squares['variable'].replace({'y': 'Y', 'x2': "X''", 'x1': "X'", 'x0': 'X', 'ct': 'A', 'co': 'B'})
# make them categorical with the order of the variables
r_squares['variable'] = pd.Categorical(r_squares['variable'], categories=variables, ordered=True)

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
#plt.axvspan(-1.0, 0, color='grey', alpha=0.3)
sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='variable', style='best_arch', palette=colors, ax=axs, s=100, alpha=0.7)
plt.xlim(-1, 1.05)
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yscale('log')
plt.xlabel('R²')
plt.ylabel('Validation Loss')
plt.title('Superposition fit')
# move the legend outside the plot  
plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left').remove()
#plt.tight_layout()

# save the figure
fig.savefig('03_results/figures/simL_val_loss_vs_r2.pdf', bbox_inches='tight')


##############################
# second plot for max correlation
##############################

# load the evaluation metrics

results_dir = '03_results/reports/files/sim2L_sae_metrics_pearson_latent-'
latents = [20, 100, 150]

metrics = []
for latent in latents:
    metrics.append(pd.read_csv(results_dir + str(latent) + '.csv'))
metrics = pd.concat(metrics)
metrics = metrics.reset_index(drop=True)

def transform_str_to_avg(x):
    items = [y for y in x.strip('[]').split(' ')]
    # remove all '' elements
    items = [x for x in items if x != '']
    # transform to float
    items = np.asarray([float(x) for x in items])
    return np.max(items)

metrics['highest_corr_x0'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x0'].values]
metrics['highest_corr_x1'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x1'].values]
metrics['highest_corr_x2'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x2'].values]
metrics['highest_corr_ct'] = [transform_str_to_avg(x) for x in metrics['highest_corr_ct'].values]
metrics['highest_corr_co'] = [transform_str_to_avg(x) for x in metrics['highest_corr_co'].values]

metrics_corr = metrics[['latent_dim', 'hidden_factor', 'highest_corr_x0', 'highest_corr_x1', 'highest_corr_x2', 'highest_corr_ct', 'highest_corr_co']]
# transform into long format to get highest correlation per variable
metrics_corr = pd.melt(metrics_corr, id_vars=['latent_dim', 'hidden_factor'], var_name='variable', value_name='highest_corr')
metrics_corr = metrics_corr.sort_values(by=['latent_dim', 'variable'])

# rename the variables (keep the last part of the string)
metrics_corr['variable'] = metrics_corr['variable'].str.split('_').str[-1]
#metrics_corr['variable'] = metrics_corr['variable'].replace({'y': 'Y', 'x2': 'X₂', 'x1': 'X₁', 'x0': 'X₀', 'ct': 'A', 'co': 'B'})
metrics_corr['variable'] = metrics_corr['variable'].replace({'y': 'Y', 'x2': "X''", 'x1': "X'", 'x0': 'X', 'ct': 'A', 'co': 'B'})
# make them categorical with the order of the variables
metrics_corr['variable'] = pd.Categorical(metrics_corr['variable'], categories=variables, ordered=True)
# set the hidden factor to a categorical string
metrics_corr = metrics_corr.sort_values(by=['hidden_factor'])
metrics_corr['hidden_factor'] = metrics_corr['hidden_factor'].astype(str)
# rename latent dim to Latent dimension
#metrics_corr = metrics_corr.rename(columns={'latent_dim': 'Latent\ndimension'})
metrics_corr['hidden_dim'] = metrics_corr['hidden_factor'] * metrics_corr['latent_dim']


# plot the variable against n_activation_features and color by latent dim
fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.lineplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='variable', palette=colors, ax=ax, linewidth=1.0, legend=False)
sns.scatterplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='latent_dim', palette=colors, ax=ax, s=20)
plt.ylim(-0.6, 1.05)
plt.xlabel('SAE hidden scaling factor')
plt.ylabel('Pearson correlation')
plt.title('max(corr(neurons, variable))')
#plt.title('Large simulation')
# set the ticks to the actual x values
#plt.xticks(metrics_corr['hidden_factor'].unique())
#plt.xscale('log')
#plt.tight_layout()
# keep only the style legend
plt.legend(title='Latent dim', loc='upper left', bbox_to_anchor=(1, 1)).remove()
handles, labels = ax.get_legend_handles_labels()
# remove unwanted elements and add an empty element to keep the order
import matplotlib
handles_n = handles[2:8] + handles[8:]
labels_n = labels[2:7] + [''] + labels[8:]
# create the legend and reduce the distance between markers and text
plt.legend(handles_n, labels_n, loc='upper left', bbox_to_anchor=(0.95, 1.05), markerscale=2, handletextpad=0.01, frameon=False)

fig.savefig('03_results/figures/simL_variable_corr.pdf', bbox_inches='tight')