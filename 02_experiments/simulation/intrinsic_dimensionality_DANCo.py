import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import math

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

complexity = 'high'
n_samples = 100000
data_dir = '/home/vschuste/data/simulation/'

for seed in range(10):
    temp_y = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    if seed == 0:
        rna_counts = temp_y
    else:
        rna_counts = torch.cat((rna_counts, temp_y), dim=0)
# limit to the training data
n_samples_train = int(n_samples*0.9)
rna_counts = rna_counts[:n_samples_train]
rna_counts = rna_counts.cpu().numpy()

#!pip install scikit-dimension
import skdim
import time

start_time = time.time()

pca_dim = skdim.id.DANCo().fit(rna_counts).dimension_

end_time = time.time()
print("Execution time: {} minutes".format((end_time - start_time)/60))
print("Estimated intrinsic dimension: {}".format(pca_dim))