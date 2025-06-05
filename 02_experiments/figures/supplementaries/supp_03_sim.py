import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import random

import sys
sys.path.append(".")
sys.path.append('src')
from src.models.sparse_autoencoder import *
from src.visualization.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

####################################################################################################
# load data and models
####################################################################################################

# load the data
rna_counts = torch.tensor(np.load("01_data/sim_rna_counts.npy"))
tf_scores = torch.tensor(np.load("01_data/sim_tf_scores.npy"))
activity_score = torch.tensor(np.load("01_data/sim_activity_scores.npy"))
accessibility_scores = torch.tensor(np.load("01_data/sim_accessibility_scores.npy"))

####################################################################################################
# set up figure
####################################################################################################

figure_kwargs = {
    'fig_width': 16,
    'fig_height': 10,
    'font_size': 14,
    'point_size': 20,
    'line_width_background': 0.5,
    'wspace': 0.5,
    'hspace': 0.5,
}
fig_dir = '03_results/figures/supplementaries/'
fig_name = 'ae_losses_and_superpositions.pdf'

plt.rcParams.update({'font.size': figure_kwargs['font_size']})

loss_colors = sns.color_palette(n_colors=2)
superposition_colors = sns.color_palette('colorblind', 3)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# set up a figure with grid: 3 rows, 5 columns
fig = plt.figure(figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
gs = GridSpec(3, 5, figure=fig)
# add spacing
gs.update(wspace=figure_kwargs['wspace'], hspace=figure_kwargs['hspace'])
ax_list = []

model_names = ['layer1_latent2_v1', 'layer1_latent4_v2', 'layer1_latent10_v1']
subplot_titles = ['2 latent units (compressed)', '4 latent units (# variables)', '10 latent units (overcomplete)']
figure_annotation = ['A', 'B', 'C']

for i, model_name in enumerate(model_names):
    # for each model, plot a row of learning curves and superpositions
    # load history and model
    history = pd.read_csv('03_results/models/sim1_'+model_name+'_history.csv')
    #if i == 1:
    #    model_name = 'layer1_latent4'
    encoder = torch.load('03_results/models/sim1_'+model_name+'_encoder.pth')
    # plot the learning curves
    ax_list.append(fig.add_subplot(gs[i, :2]))
    # get the A, B, C annotation
    ax_list[-1].text(-0.15, 1.25, figure_annotation[i], transform=ax_list[-1].transAxes, fontweight='bold', va='top', ha='right', fontsize=figure_kwargs['font_size']+2)
    ax_list[-1].plot(history['train_loss'], label='train loss', color=loss_colors[0])
    ax_list[-1].plot(history['val_loss'], label='validation loss', color=loss_colors[1])
    ax_list[-1].set_ylabel('MSE loss')
    if i < len(model_names)-1:
        # remove legend
        ax_list[-1].legend().remove()
        ax_list[-1].set_xticklabels([])
        ax_list[-1].set_xticks([])
        ax_list[-1].set_xlabel('')
    else:
        ax_list[-1].set_xlabel('Epoch')
        # set the legend under the last plot
        ax_list[-1].legend(bbox_to_anchor=(0.05, 1), loc='upper left', frameon=False, ncol=2)
    
    # now compute superpositions
    reps = encoder(rna_counts).detach()

    # plot the superpositions
    for j in range(tf_scores.shape[1]):
        reg = LinearRegression().fit(reps.detach().cpu().numpy(), tf_scores[:, j].detach().numpy())
        superposition = np.matmul(reps.detach().numpy(), reg.coef_.T)
        ax_list.append(fig.add_subplot(gs[i, j+2]))
        if j == 0:
            # add a title with spacing to the plot
            ax_list[-1].set_title(f'{subplot_titles[i]}', pad=20)
        # add a black line in the background for y=x
        ax_list[-1].plot(superposition, superposition, color='black', linewidth=figure_kwargs['line_width_background'])
        sns.scatterplot(x=tf_scores[:, j].detach().numpy(), y=superposition, ax=ax_list[-1], c=superposition_colors[j], s=figure_kwargs['point_size'])
        # plot the regression R value in the top right corner
        ax_list[-1].text(0.55, 0.05, f'R={reg.score(reps.detach().numpy(), tf_scores[:, j].detach().numpy()):.2f}', transform=ax_list[-1].transAxes)
        ax_list[-1].set_ylabel('Superposition')
        if i < len(model_names)-1:
            ax_list[-1].set_xticklabels([])
            ax_list[-1].set_xticks([])
            ax_list[-1].set_xlabel('')
        else:
            ax_list[-1].set_xlabel(f'X{j}'.translate(SUB).format(i))

#plt.tight_layout()
plt.savefig(fig_dir+fig_name, bbox_inches='tight')