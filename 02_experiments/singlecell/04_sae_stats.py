import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

def load_data(model_type):
    if model_type == 'DGD':
        # load data and model
        data = ad.read_h5ad('./01_data/human_bonemarrow.h5ad')
        model = multiDGD.DGD.load(data=data, save_dir='./03_results/models/', model_name='human_bonemarrow_l20_h2-3_test50e')
        reps = model.representation.z.detach()
        del model
    elif model_type == 'Enformer':
        data = ad.read_h5ad('./01_data/human_bonemarrow.h5ad')
        train_indices = np.where(data.obs['train_val_test'] == 'train')[0]
        enformer_embeddings = pd.read_csv('./03_results/embeddings/human_bonemarrow_geneformer.csv', index_col=0)
        enformer_embeddings = enformer_embeddings.iloc[train_indices, :]
        reps = torch.FloatTensor(enformer_embeddings.iloc[:,:-1].values) # excluding cell_type
        del enformer_embeddings
    else:
        raise ValueError("Invalid model type. Choose 'DGD' or 'Enformer'.")
    data = data[data.obs['train_val_test'] == 'train']

    return data, reps

def load_model(model_name, input_size, hidden_size):
    # Sparse Autoencoder Model with Mechanistic Interpretability
    class SparseAutoencoder(nn.Module):
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
    
    sae_model = SparseAutoencoder(input_size, hidden_size)
    sae_model.load_state_dict(torch.load(model_name+'.pt'))
    return sae_model

def calculate_global_and_local_features(data, activations):
    # go through all cell types and compute each features mean activation and std for ct positive and negative
    cell_types = [x for x in data.obs['cell_type'].unique()]
    means_positive = torch.zeros((len(cell_types), activations.shape[1]))
    se_positive = torch.zeros((len(cell_types), activations.shape[1]))
    means_negative = torch.zeros((len(cell_types), activations.shape[1]))
    se_negative = torch.zeros((len(cell_types), activations.shape[1]))
    for i, ct in enumerate(cell_types):
        data_indices = data.obs['cell_type'] == ct
        activations_pos = activations[data_indices,:]
        activations_neg = activations[~data_indices,:]
        means_positive[i,:] = torch.mean(activations_pos, dim=0).detach().cpu()
        # compute the standard error of the mean
        se_positive[i,:] = torch.std(activations_pos, dim=0).detach().cpu() / math.sqrt(activations_pos.shape[0])
        means_negative[i,:] = torch.mean(activations_neg, dim=0).detach().cpu()
        se_negative[i,:] = torch.std(activations_neg, dim=0).detach().cpu() / math.sqrt(activations_neg.shape[0])
    # that gives the "significant features" per cell type
    significant_features = torch.BoolTensor((means_positive - means_negative) > 1.96*(se_positive + se_negative))
    sum_significant_features = torch.sum(significant_features, dim=0)

    ###
    # unique (local) vs shared (global) features, and there is a thing in the middle which I call regional
    ###
    # all active features
    active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()
    # local features are the ct specific ones
    local_features = (torch.where(sum_significant_features == 1)[0]).tolist()
    regional_features = (torch.where(sum_significant_features > 1)[0]).tolist()
    # all remaining active features are global
    global_features = list(set(active_features).difference(set(local_features)))
    return active_features, local_features, global_features

def get_locality_stats(device_id=2, seed=9307, model_type='DGD', model_name=''):
    # Set the device for PyTorch
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # Set a random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ###
    # model loading and activation extraction
    ###
    data, reps = load_data(model_type)

    input_size = reps.shape[1]
    if model_type == 'DGD':
        hidden_size = 10**4
    elif model_type == 'Enformer':
        hidden_size = 10**4
    sae_model = load_model(model_name, input_size, hidden_size)
    sae_model.to(device)

    # get activations in chunks
    batch_size = 128
    activations = []
    for i in range(0, reps.shape[0], batch_size):
        batch_reps = reps[i:i+batch_size, :].to(device)
        with torch.no_grad():
            _, activations_batch = sae_model(batch_reps)
        activations.append(activations_batch.cpu())
    activations = torch.cat(activations, dim=0)

    ###
    # general stats
    ###
    # all active features
    active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()
    active_features, local_features, global_features = calculate_global_and_local_features(data, activations)

    # get the mean and max activations, and the number of samples per features
    mean_activations = torch.mean(activations, dim=0).detach().cpu().numpy()
    max_activations = torch.max(activations, dim=0)[0].detach().cpu().numpy()
    nonzero_activations = torch.sum(activations != 0, dim=0).detach().cpu().numpy()
    # get the feature types
    feature_types = []
    for i in range(activations.shape[1]):
        if i in local_features:
            feature_types.append('local')
        elif i in global_features:
            feature_types.append('global')
        else:
            feature_types.append('dead')

    activ_df = pd.DataFrame({'mean': mean_activations, 'max': max_activations, 'feature type': feature_types, 'nonzero': nonzero_activations})
    activ_df['model'] = model_type
    activ_df['seed'] = seed

    return activ_df

def main(device_id=2, model_type='DGD'):
    activ_dfs = []
    if model_type == 'DGD':
        # DGD models
        for seed in [0, 42, 9307]:
            if seed == 0:
                model_name = f'03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'
            else:
                model_name = f'03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_seed{seed}'
            activ_df = get_locality_stats(model_type=model_type, model_name=model_name, seed=seed)
            activ_dfs.append(activ_df)
    elif model_type == 'Enformer':
        # Enformer models
        seed = 0
        activ_df = get_locality_stats(model_type='Enformer', model_name='03_results/models/sae_enformer_10000_l1-1e-3_lr-1e-4_500epochs', seed=seed, device_id=device_id)
        activ_dfs.append(activ_df)
    else:
        raise ValueError("Invalid model type. Choose 'DGD' or 'Enformer'.")
    activ_df = pd.concat(activ_dfs)
    activ_df.to_csv(f'03_results/reports/singlecell_sae_activations_stats_{model_type}.csv')

    # print out a summary statistic (number of local, global, dead features per model (model_type + seed))
    means = []
    for model_type in activ_df['model'].unique():
        print(f"Model: {model_type}")
        for seed in activ_df[activ_df['model'] == model_type]['seed'].unique():
            print(f"  Seed: {seed}")
            temp_means = []
            for feature_type in ['local', 'global', 'dead']:
                n_features = activ_df[(activ_df['feature type'] == feature_type) & (activ_df['seed'] == seed)].shape[0]
                print(f"    Number of {feature_type} features: {n_features}")
                temp_means.append(n_features)
            means.append(temp_means)
        print("\n")
    # print the mean and standard error of the number of local, global, dead features across all models
    means = np.array(means)
    mean_means = np.mean(means, axis=0)
    se_means = np.std(means, axis=0) / math.sqrt(means.shape[0])
    print("Mean number of features across all models:")
    print(f"  Local: {mean_means[0]} ± {se_means[0]}")
    print(f"  Global: {mean_means[1]} ± {se_means[1]}")
    print(f"  Dead: {mean_means[2]} ± {se_means[2]}")

if __name__ == "__main__":
    # take in arguments whether to run on DGD or Enformer (cannot run both at the same time because they require different environments)
    import argparse
    parser = argparse.ArgumentParser(description='Run locality stats on DGD or Enformer models.')
    parser.add_argument('--model_type', type=str, choices=['DGD', 'Enformer'], required=True, help='Model type to run on (DGD or Enformer)')
    parser.add_argument('--device_id', type=int, default=2, help='Device ID to use for PyTorch (default: 2)')
    args = parser.parse_args()
    device_id = args.device_id
    model_type = args.model_type
    main(device_id=device_id, model_type=model_type)