import torch
import numpy as np
from scipy.stats import spearmanr

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

def get_n_features(sae_model, train_loader, activations, threshold=1e-10):
    n_activation_features = activations.shape[1]
    unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)
    avg_active_hidden_units = count_active_hidden_units(sae_model, train_loader, threshold=threshold, avg=True)
    return n_activation_features, unique_active_unit_indices, avg_active_hidden_units

def get_number_of_redundant_features(activations, unique_activs, threshold=0.95):
    # compute correlations between all active features
    redundant_set = set()
    for i in range(len(unique_activs)):
        for j in range(i+1, len(unique_activs)):
            corr = np.corrcoef(activations[:, unique_activs[i]].cpu().detach().numpy(), activations[:, unique_activs[j]].cpu().detach().numpy())[0, 1]
            if corr > threshold:
                # add the feature to the redundant set
                redundant_set.add(unique_activs[j])
    n_redundant = len(redundant_set)
    return n_redundant

def get_correlations_with_data(activations, unique_activs, comparison_data):
    correlations_p = np.zeros((len(unique_activs), comparison_data.shape[1]))
    correlations_s = np.zeros((len(unique_activs), comparison_data.shape[1]))
    for i, feat in enumerate(unique_activs):
        # get the activations
        feat_activation = activations[:, feat]
        for j in range(comparison_data.shape[1]):
            corr = np.corrcoef(feat_activation.cpu().detach().numpy(), comparison_data[:, j])[0, 1]
            correlations_p[i, j] = corr
            # now spearman
            corr, _ = spearmanr(feat_activation.cpu().detach().numpy(), comparison_data[:, j])
            correlations_s[i, j] = corr
    return correlations_p, correlations_s

def get_n_features_per_attribute(correlations, threshold=0.95):
    n_per_attribute = np.zeros(correlations.shape[1])
    for i in range(correlations.shape[1]):
        n_per_attribute[i] = np.sum(np.abs(correlations[:, i]) > threshold)
    return n_per_attribute

def get_highest_corr_per_attribute(correlations):
    best_corrs = np.zeros(correlations.shape[1])
    for i in range(correlations.shape[1]):
        best_id = np.argmax(np.abs(correlations[:, i]))
        best_corrs[i] = correlations[best_id, i]
    return best_corrs

def run_sae_analysis(sae_model, train_loader, activations, comparison_data, corrtype='pearson'):
    n_activation_features, unique_active_unit_indices, avg_active_hidden_units = get_n_features(sae_model, train_loader, activations)
    n_unique = len(unique_active_unit_indices)
    n_redundant = get_number_of_redundant_features(activations, unique_active_unit_indices)
    if len(unique_active_unit_indices) > 0:
        correlations_p, correlations_s = get_correlations_with_data(activations, unique_active_unit_indices, comparison_data)
        corrs = correlations_p if corrtype == 'pearson' else correlations_s
        n_per_attribute = get_n_features_per_attribute(corrs)
        highest_corrs = get_highest_corr_per_attribute(corrs)
    else:
        n_per_attribute = np.zeros(comparison_data.shape[1])
        highest_corrs = np.zeros(comparison_data.shape[1])
    return n_activation_features, avg_active_hidden_units, n_unique, n_redundant, n_per_attribute, highest_corrs