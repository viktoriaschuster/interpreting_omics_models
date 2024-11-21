import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def plot_obs_activations(data, obs, activations, unique_ids):
    obs_vals = np.unique(data.obs[obs].values)

    means = np.zeros((len(obs_vals), len(unique_ids)))

    for i, ct in enumerate(obs_vals):
        ids_ct = np.where(data.obs[obs] == ct)[0]
        activations_ct = activations[ids_ct,:].detach().cpu().numpy()
        avg_activations = np.mean(activations_ct, axis=0)
        means[i, :] = avg_activations[unique_ids]
        del activations_ct
        del avg_activations

    df_means = pd.DataFrame(means, index=obs_vals, columns=unique_ids)

    sns.heatmap(df_means, cmap='rocket_r')
    plt.title(f'Mean activations: {obs}')
    plt.xlabel('Hidden unit')
    plt.ylabel(obs)
    plt.show()

def plot_activation_range(data, activations, range_start_end=None, indices=None, obs=None, plot_size=(10,10), obs_order=None):

    if range_start_end is not None:
        activs = activations[:, range_start_end[0]:range_start_end[1]].detach().cpu().numpy()
    elif indices is not None:
        activs = activations[:, indices].detach().cpu().numpy()
    else:
        activs = activations.detach().cpu().numpy()

    if obs is not None:
        # sort by obs
        if range_start_end is not None:
            df_temp = pd.DataFrame(activs, columns=range(range_start_end[0], range_start_end[1]))
        elif indices is not None:
            df_temp = pd.DataFrame(activs, columns=indices)
        else:
            df_temp = pd.DataFrame(activs, columns=range(activs.shape[1]))
        df_temp['obs'] = data.obs[obs].values
        df_temp.index = data.obs[obs].values
        if obs_order is not None:
            df_temp['obs'] = pd.Categorical(df_temp['obs'], categories=obs_order, ordered=True)
        df_temp = df_temp.sort_values(by='obs')

        fig, ax = plt.subplots(figsize=plot_size)
        sns.heatmap(df_temp.drop(columns='obs'), cmap='rocket_r')
        if range_start_end is not None:
            plt.title(f'Activations for range {range_start_end[0]}-{range_start_end[1]}')
        else:
            plt.title(f'Activations')
        plt.xlabel('Hidden unit')
        plt.ylabel(obs)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=plot_size)
        sns.heatmap(activs, cmap='rocket_r')
        if range_start_end is not None:
            plt.title(f'Activations for range {range_start_end[0]}-{range_start_end[1]}')
        else:
            plt.title(f'Activations')
        plt.xlabel('Hidden unit')
        plt.show()

import torch

# Function to get the unique indices of active hidden units across all samples
def get_unique_active_unit_indices(model, data_loader, threshold=1e-5):
    unique_active_indices = set()  # Use a set to store unique active neuron indices

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data in data_loader:
            inputs = data
            
            # Forward pass through the encoder to get the hidden layer activations
            _, encoded = model(inputs)
            
            # Find the indices of the active neurons (where activation > threshold)
            active_indices_batch = (encoded > threshold).nonzero(as_tuple=False)  # Get nonzero indices

            # This returns indices in the form [batch_idx, neuron_idx]
            # We are only interested in neuron_idx for overall unique active units
            unique_active_indices.update(active_indices_batch[:, 1].tolist())

    # Convert to sorted list for convenience
    unique_active_indices = sorted(unique_active_indices)

    return unique_active_indices

# Assuming 'model' is your trained SparseAutoencoder and 'data_loader' is your dataset loader
def count_active_hidden_units(model, data_loader, threshold=1e-5, avg=True):
    n_active_units = []
    total_samples = 0

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data in data_loader:
            inputs = data
            
            # Forward pass through the encoder to get the hidden layer activations
            _, encoded = model(inputs)
            
            # Count how many activations are above the threshold (active neurons)
            active_units_per_sample = (encoded > threshold).sum(dim=1)
            
            # Sum over all samples
            n_active_units.append(active_units_per_sample.sum().item())
            total_samples += inputs.size(0)
    
    if avg:
        avg_active_units = sum(n_active_units) / total_samples
        return avg_active_units
    else:
        return n_active_units

from scipy.signal import find_peaks

def detect_valleys(activations, individual_activation):
    # Compute histogram (adjust bins as necessary)
    hist_values, bin_edges = np.histogram(activations[:, individual_activation].detach().cpu().numpy(), bins=100)
    
    # Get bin centers (midpoints of bin edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Invert the histogram to find valleys (local minima)
    inverted_hist = -hist_values

    # Find peaks in the inverted histogram (which correspond to valleys in the original histogram)
    valley_indices, _ = find_peaks(inverted_hist, distance=5)  # Adjust distance based on expected group separations

    # Get the bin center values corresponding to the valleys
    valley_positions = bin_centers[valley_indices]

    return valley_positions