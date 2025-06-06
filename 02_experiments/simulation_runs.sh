###
# small simulation for SAE sweep
###

# first training the autoencoders on the data
python 02_experiments/simulation/small/00_data_simulation.py
python 02_experiments/simulation/small/01_ae_training.py

# next the SAE sweeps
python 02_experiments/simulation/small/01_simulation_bricken_sweep.py
python 02_experiments/simulation/small/01_simulation_topk_sweep.py
python 02_experiments/simulation/small/01_simulation_vanilla_sweep.py

###
# large simulation
###

# first generating the data
python 02_experiments/simulation/large/00_data_simulation.py

# next find good hyperparameters for the autoencoders
python 02_experiments/simulation/large/01_autoencoder_optuna.py

# train the autoencoders with the best hyperparameters (we ran with 3 different random seeds: use argument --seed)
python 02_experiments/simulation/large/02_autoencoder_training.py --latent_dim 20 # underdetermined
python 02_experiments/simulation/large/02_autoencoder_training.py --latent_dim 150 # roughly determined
python 02_experiments/simulation/large/02_autoencoder_training.py --latent_dim 1000 # over determined

# another SAE sweep (also returning analysis files)
python 02_experiments/simulation/03_simulation2L_bricken_sweep.py --latent_dim 20
python 02_experiments/simulation/03_simulation2L_bricken_sweep.py --latent_dim 150
python 02_experiments/simulation/03_simulation2L_bricken_sweep.py --latent_dim 1000
# topk
python 02_experiments/simulation/03_simulation2L_topk_sweep.py --latent_dim 20
python 02_experiments/simulation/03_simulation2L_topk_sweep.py --latent_dim 150
python 02_experiments/simulation/03_simulation2L_topk_sweep.py --latent_dim 1000
# vanilla
python 02_experiments/simulation/03_simulation2L_vanilla_sweep.py --latent_dim 20
python 02_experiments/simulation/03_simulation2L_vanilla_sweep.py --latent_dim 150
python 02_experiments/simulation/03_simulation2L_vanilla_sweep.py --latent_dim 1000

# SAE training of the SAE used for section 4.3 analysis
python 02_experiments/simulation/large/03_sae_training.py --latent_dim 150

# running baselines for comparison of variable recovery in 4.3
python 02_experiments/simulation/large/05_rebuttal_ica.py
python 02_experiments/simulation/large/05_rebuttal_pca.py
python 02_experiments/simulation/large/05_rebuttal_svd.py