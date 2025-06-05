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

data_dir = '/home/vschuste/data/simulation/'

def data_simulation(seed, y_dim, x_dim, n_ct, n_cov, n_cells, printing=False):
    # set the seed (here always 0 for the base distributions)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # adjacency matrices (binary)
    p_xy = 0.1
    xy = torch.tensor(np.random.binomial(1, p_xy, (x_dim, y_dim))).float()
    if n_cov == 1:
        p_ctx = 0.7
    else:
        p_ctx = 0.3
    cx = torch.tensor(np.random.binomial(1, p_ctx, (n_ct, x_dim))).float()
    if n_cov == 1:
        cxo = torch.ones(n_cov, x_dim)
    else:
        p_cox = 0.9
        cxo = torch.tensor(np.random.binomial(1, p_cox, (n_cov, x_dim))).float()

    # distributions for the choice of categorical variables
    p_ct = torch.distributions.categorical.Categorical(1/n_ct*torch.ones(n_ct))
    p_co = torch.distributions.categorical.Categorical(1/n_cov*torch.ones(n_cov))

    # sample activity is a poisson
    poisson_lambda = 1.0
    # get x many slightly different poisson lambdas
    poisson_lambdas = torch.tensor([poisson_lambda + 0.1 * i for i in range(x_dim)])
    activity_distribution = torch.distributions.poisson.Poisson(poisson_lambdas)

    # distribution for sample noise
    # random values between 0 and 1 for p_noise in shape n_cov
    #p_noise = torch.rand((n_cov, x_dim))
    #noise_distribution = torch.distributions.bernoulli.Bernoulli(p_noise)
    # n_cov many gaussians with different means
    noise_means = torch.tensor([1.0 * i for i in range(n_cov)])
    noise_distribution = torch.distributions.normal.Normal(noise_means, 0.1)

    ###
    # sampling
    ###
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # sample activities
    x_0 = activity_distribution.sample((n_cells,))#.unsqueeze(1)

    if printing:
        print('Generating covariate-specific activities')
    co = p_co.sample((n_cells,))
    if n_cov == 1:
        #noise = noise_distribution.sample((n_cells, x_dim)) * 1e-6
        #co_noise = noise.unsqueeze(1) * cxo.unsqueeze(0)
        #x_1 = x_0.unsqueeze(1) + co_noise
        x_1 = x_0
    else:
        noise = noise_distribution.sample((n_cells,))# * 1e-3
        x_1 = x_0.unsqueeze(1) + noise.unsqueeze(-1)
        x_1 = x_1[torch.arange(n_cells), co]
    if printing:
        print(x_0.unsqueeze(1).shape, cxo.unsqueeze(0).shape, x_1.shape)
    
    if printing:
        print('Generating ct specific activities of biological programs')
    # sample cell type identities and create ct-specific activities
    ct = p_ct.sample((n_cells,)) # indices of which ct to use
    x_2 = x_1.unsqueeze(1) * cx.unsqueeze(0)
    x_2 = x_2[torch.arange(n_cells), ct]
    if printing:
        print(x_1.shape, cx.unsqueeze(0).shape, x_2.shape)

    # create observable data
    if printing:
        print('Generating observable data Y')
    y = xy.unsqueeze(0) * x_2.unsqueeze(2)
    y = y.sum(1)

    if printing:
        print(xy.unsqueeze(0).shape, x_2.unsqueeze(2).shape, y.shape)

    return y, x_0, x_1, x_2, ct, co, xy

complexity = 'high'
n_ct = 40
n_cov = 3
for i in range(0,10):

    y, x_0, x_1, x_2, ct, co, yx_mtrx = data_simulation(
        seed=i,
        y_dim=20000,
        x_dim=100,
        n_ct=n_ct,
        n_cov=n_cov,
        n_cells=10000
    )

    torch.save(y, data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed))
    torch.save(x_0, data_dir+'large_{}-complexity_rs{}_x0.pt'.format(complexity, seed))
    torch.save(x_1, data_dir+'large_{}-complexity_rs{}_x1.pt'.format(complexity, seed))
    torch.save(x_2, data_dir+'large_{}-complexity_rs{}_x2.pt'.format(complexity, seed))
    torch.save(ct, data_dir+'large_{}-complexity_rs{}_ct.pt'.format(complexity, seed))
    torch.save(co, data_dir+'large_{}-complexity_rs{}_co.pt'.format(complexity, seed))

    if i == 0:
        torch.save(yx_mtrx, data_dir+'large_{}-complexity_yx_mtrx.pt'.format(complexity))