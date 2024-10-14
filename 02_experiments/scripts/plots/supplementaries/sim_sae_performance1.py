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
# load data
####################################################################################################


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
fig_name = 'sim_sae_performance1.pdf'

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

###############


###############

#plt.tight_layout()
plt.savefig(fig_dir+fig_name, bbox_inches='tight')