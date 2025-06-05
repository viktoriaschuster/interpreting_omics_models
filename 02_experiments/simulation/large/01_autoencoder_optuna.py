import numpy as np
import torch
import random
import gc
import wandb
import optuna

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

from src.models.sparse_autoencoder import *
from src.models.autoencoder import *
from src.functions.ae_training import *
from src.visualization.plotting import *

###
# argparse
###

import argparse

parser = argparse.ArgumentParser(description='Train an autoencoder on the simulated data.')
#parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to train on.')
parser.add_argument('--latent_dim', type=int, default=150, help='Dimension of the latent space.')
parser.add_argument('--model_depth', type=int, default=2, help='Number of layers in the encoder and decoder.')
parser.add_argument('--width', type=str, default='narrow', help='AE can be wide or narrow')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
args = parser.parse_args()

#n_samples = args.n_samples
latent_dim = args.latent_dim
model_depth = args.model_depth
model_width = args.width
complexity = 'high'
n_samples = 20000

################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    # specify the gpu id
    torch.cuda.set_device(args.gpu)
    print('Using GPU', torch.cuda.current_device())

# set a random seed
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

################################

# load the data

data_dir = '/home/vschuste/data/simulation/'

for seed in range(10):
    if seed == 0:
        rna_counts = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    else:
        temp = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
        rna_counts = torch.cat((rna_counts, temp), dim=0)
# now subsample the data if necessary
if n_samples < rna_counts.shape[0]:
    rna_counts = rna_counts[:n_samples]
rna_counts = rna_counts.to(device)
n_samples_validation = int(n_samples*0.1)
print('Data loaded.')
print('Data shape:', rna_counts.shape)

################################

n_epochs = 100
n_trials = 50
study_name = 'largesim_ae_optuna_latent-{}_depth-{}_width-{}'.format(latent_dim, model_depth, model_width)

################################

def objective(trial):

    #lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3) # deprecated
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    wd = trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-5, 1e-3, 1e-1])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1])
    bs = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])


    encoder = Encoder2(20000, latent_dim, model_depth, dropout=dropout, width=model_width).to(device)
    decoder = Decoder2(latent_dim, 20000, model_depth, hidden_dims=encoder.hidden_dims, dropout=dropout, width=model_width).to(device)

    encoder, decoder, history = train_and_eval_model(
        encoder, decoder, rna_counts, n_samples, n_samples_validation,
        n_epochs=n_epochs, batch_size=bs, learning_rate=lr, weight_decay=wd,
        early_stopping=10,
        loss_type="MSE"
        #loss_type="PoissonNLL"
    )

    del encoder, decoder

    torch.cuda.empty_cache()
    gc.collect()

    if math.isnan(history["val_loss"].values[-1]):
        print("Loss is NaN")
        return 1e9
    else:
        return np.mean(history["val_loss"].values[-3:])

# set up wandb
from optuna_integration import WeightsAndBiasesCallback
wandb_kwargs = {"project": "sc_simulation"}
wandbc = WeightsAndBiasesCallback(metric_name=["Loss"], wandb_kwargs=wandb_kwargs)

study = optuna.create_study(study_name=study_name, directions=["minimize"])
wandb.run.name = study_name
study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])

print(" ")
print("##########################")
print("Best study parameters:")
print("##########################")
print(" ")

trial_with_lowest_mse = min(study.best_trials, key=lambda t: t.values[0])
print(f"trial with best loss {trial_with_lowest_mse.number}: {trial_with_lowest_mse.values[0]}")