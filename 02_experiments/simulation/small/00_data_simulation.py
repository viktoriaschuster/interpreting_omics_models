import numpy as np
import torch
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# setup:
# 3 genes, transcribed into mRNA and translated into proteins
# the randomness comes from 2 places:
# 1. sampling from a poisson distribution, indicating the level of transcription activity
# 2. sampling from bernoulli distributions for each gene, indicating whether the gene is even accessible or not

# set up experiment parameters
poisson_lambda = 2.0
p_bernoulli = torch.tensor([0.5, 0.1, 0.9])
gene_regulation = torch.tensor([
    [0., 1., 1., 0., 0.],
    [1., 1., 0., 1., 0.],
    [0., 0., 0., 1., 1.]
])

n_samples_validation = 2000
n_samples = 10000 + n_samples_validation
n_epochs = 20000

# load or generate the data
data_path = "01_data/"

if os.path.exists(data_path + "sim_rna_counts.npy"):
    rna_counts = torch.tensor(np.load(data_path + "sim_rna_counts.npy"))
    tf_scores = torch.tensor(np.load(data_path + "sim_tf_scores.npy"))
    activity_score = torch.tensor(np.load(data_path + "sim_activity_scores.npy"))
    accessibility_scores = torch.tensor(np.load(data_path + "sim_accessibility_scores.npy"))

    print("Loaded data from files.")
else:
    # init distributions
    activity_distribution = torch.distributions.poisson.Poisson(torch.tensor([poisson_lambda]))
    accessibility_distribution = torch.distributions.bernoulli.Bernoulli(p_bernoulli)

    # sample from distributions to create mRNA counts
    # get the activity score
    activity_score = activity_distribution.sample((n_samples,))
    # get the accessibility scores
    accessibility_scores = accessibility_distribution.sample((n_samples,))
    # get the mRNA counts
    tf_scores = activity_score * accessibility_scores

    # correct mRNA counts for gene regulation
    # first, get the regulation scores
    def get_regulation_scores(tf_scores, gene_regulation):
        # get the regulation scores
        return torch.matmul(tf_scores, gene_regulation)

    rna_counts = get_regulation_scores(tf_scores, gene_regulation)

    np.save("01_data/sim_rna_counts.npy", rna_counts.numpy())
    np.save("01_data/sim_tf_scores.npy", tf_scores.numpy())
    np.save("01_data/sim_activity_scores.npy", activity_score.numpy())
    np.save("01_data/sim_accessibility_scores.npy", accessibility_scores.numpy())

print(f'RNA shape: {rna_counts.shape}')
print(f'activity score shape: {activity_score.shape}') # one per sample
print(f'accessibility score shape: {accessibility_scores.shape}') # one per originating gene