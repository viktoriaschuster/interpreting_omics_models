# export the code from the notebook to data-generation script and execute here

# first training the autoencoders on the data
# the first one is the test run (maybe I will keep the analysis on low-sample models, could be valuable)
python 02_experiments/simulation/larger_simulation_AE_training.py --n_samples 1000 --latent_dim 150
# now the full data models: underdetermined, roughly determined, and over determined
python 02_experiments/simulation/larger_simulation_AE_training.py --latent_dim 20
python 02_experiments/simulation/larger_simulation_AE_training.py --latent_dim 150
python 02_experiments/simulation/larger_simulation_AE_training.py --latent_dim 1000

# now the SAE sweeps and analyses
python 02_experiments/simulation/simulation2L_bricken_sweep.py --n_samples 1000 --latent_dim 150
python 02_experiments/simulation/simulation2L_bricken_sweep.py --latent_dim 20
python 02_experiments/simulation/simulation2L_bricken_sweep.py --latent_dim 150
python 02_experiments/simulation/simulation2L_bricken_sweep.py --latent_dim 1000
# topk
python 02_experiments/simulation/simulation2L_topk_sweep.py --n_samples 1000 --latent_dim 150
python 02_experiments/simulation/simulation2L_topk_sweep.py --latent_dim 20
python 02_experiments/simulation/simulation2L_topk_sweep.py --latent_dim 150
python 02_experiments/simulation/simulation2L_topk_sweep.py --latent_dim 1000
# vanilla
python 02_experiments/simulation/simulation2L_vanilla_sweep.py --latent_dim 20
python 02_experiments/simulation/simulation2L_vanilla_sweep.py --latent_dim 150
python 02_experiments/simulation/simulation2L_vanilla_sweep.py --latent_dim 1000