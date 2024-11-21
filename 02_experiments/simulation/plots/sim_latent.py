import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(".")
sys.path.append('src')
from src.models.sparse_autoencoder import *
from src.visualization.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################################################################################
# load data and models
####################################################################################################

# load the data
rna_counts = torch.tensor(np.load("01_data/sim_rna_counts.npy"))
tf_scores = torch.tensor(np.load("01_data/sim_tf_scores.npy"))
activity_score = torch.tensor(np.load("01_data/sim_activity_scores.npy"))
accessibility_scores = torch.tensor(np.load("01_data/sim_accessibility_scores.npy"))

# load the model

model_name = 'layer1_latent4'
latent_size = 4

encoder = torch.load('03_results/models/sim1_'+model_name+'_encoder.pth')

reps = encoder(rna_counts).detach()

####################################################################################################
# compute coefficients and plot
####################################################################################################

# linear regression for the TF scores in the latent space and activations
reg_coeffs_reps = np.zeros((tf_scores.shape[1], reps.shape[1]))
reg_fits_reps = np.zeros(tf_scores.shape[1])

for i in range(tf_scores.shape[1]):
    reg = LinearRegression().fit(reps.cpu().detach().numpy(), tf_scores[:, i].cpu().detach().numpy())
    reg_fits_reps[i] = reg.score(reps.cpu().detach().numpy(), tf_scores[:, i].cpu().detach().numpy())
    reg_coeffs_reps[i,:] = reg.coef_

basis_vectors = np.array([
    [1, -1, -1, -1],
    [-1, 1, -1, -1],
    [-1, -1, 1, -1],
    [-1, -1, -1, 1]
])

# make a pca of the basis vectors

pca = PCA(n_components=2)
# fit the pca to both the basis vectors and the coefficients
pca.fit(np.vstack([basis_vectors, reg_coeffs_reps]))
basis_vectors_transformed = pca.transform(basis_vectors)
# transform the coefficients
coefficients_transformed = pca.transform(reg_coeffs_reps)

three_cols = sns.color_palette('colorblind', 3)
fontsize = 14
# set the fontsize
plt.rcParams.update({'font.size': fontsize})
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
plt.subplots_adjust(wspace=0.5)

mean = pca.transform(pca.mean_.reshape(1, -1))
axs[0].scatter(basis_vectors_transformed[:, 0], basis_vectors_transformed[:, 1], s=1)
mean = pca.transform(pca.mean_.reshape(1, -1))
for i in range(basis_vectors_transformed.shape[0]):
    axs[0].arrow(
        mean[0][0], mean[0][1], 
        basis_vectors_transformed[i, 0], basis_vectors_transformed[i, 1], 
        head_width=0.1, head_length=0.1, fc='black', ec='black'
        )
for i in range(coefficients_transformed.shape[0]):
    axs[0].arrow(
        mean[0][0], mean[0][1], 
        coefficients_transformed[i, 0], coefficients_transformed[i, 1], 
        head_width=0.2, head_length=0.2, fc=three_cols[i], ec=three_cols[i]
        )
    # label
    if coefficients_transformed[i, 0] < 0:
        x_annot = coefficients_transformed[i, 0] - 0.4
    else:
        x_annot = coefficients_transformed[i, 0] + 0.1
    if coefficients_transformed[i, 1] < 0:
        y_annot = coefficients_transformed[i, 1] - 0.4
    else:
        y_annot = coefficients_transformed[i, 1] + 0.3
    axs[0].annotate(('X {}'.format(i)).translate(SUB), (x_annot, y_annot))
axs[0].set_xlabel('PC 1')
axs[0].set_ylabel('PC 2')
# set xlim
axs[0].set_xlim(-1.2, 2)
axs[0].set_title('Superpositions of X')
# remove the ticks
axs[0].set_xticks([])
axs[0].set_yticks([])
# next to that plot the histograms of X values
df_x = pd.DataFrame(tf_scores.cpu().detach().numpy(), columns=['X {}'.format(i) for i in range(tf_scores.shape[1])])
# long format
df_x_long = df_x.melt()
df_x_long['Variable'] = df_x_long['variable'].apply(lambda x: x.translate(SUB))
sns.histplot(data=df_x_long, x='value', hue='Variable', ax=axs[1], bins=20, element='step', multiple='dodge', palette=three_cols, linewidth=0)
axs[1].set_xlabel('Count')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distributions of X')
#axs[1].legend(['X0', 'X1', 'X2'], three_cols, title='Variable', frameon=False)
# remove frame
#axs[1].get_legend().set_title('Variable')
# make sure the x ticks stay integers
# define MaxNLocator object
from matplotlib.ticker import MaxNLocator
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

# save the figure as pdf
fig.savefig('03_results/figures/sim1_latent4_tf_score_coefficients_and_distributions.pdf', bbox_inches='tight')