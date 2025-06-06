# get the data used here
python 02_experiments/singlecell/00_data_download.py

###
# multiDGD
###

# running another small set of SAE sweeps on different model instances and datasets
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_bricken.py
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_vanilla.py
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_brain_bricken.py
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_brain_vanilla.py
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_mousegast_bricken.py
python 02_experiments/singlecell/01_sae_training_rebuttal_sweep_mousegast_vanilla.py

# run the SAE training for the single-cell data used for analysis with 3 random seeds
python 02_experiments/singlecell/01_sae_training_randomseed.py --seed 0
python 02_experiments/singlecell/01_sae_training_randomseed.py --seed 42
python 02_experiments/singlecell/01_sae_training_randomseed.py --seed 9307

# automated analysis
python 02_experiments/singlecell/02_automated_geneset_analysis_faster.py

###
# Geneformer
###

# clone the geneformer repo and install it
bash 02_experiments/singlecell/03_geneformer_utils.sh # clone the geneformer repo and install it (needs manual download of some files though)

# get the latent embeddings
python 02_experiments/singlecell/03_geneformer_latent_extraction.py

# run the SAE training for the geneformer embeddings and save the activations
python 02_experiments/singlecell/03_sae_training_bonemarrow.py
python 02_experiments/singlecell/04_geneformer_sae_save_activation.py

# run the automated GO analysis
python 02_experiments/singlecell/04_geneformer_sae_go_analysis.py

# compare SAEs in terms of feature numbers and feature matrices for similarity
python 02_experiments/singlecell/04_sae_stats.py
python 02_experiments/singlecell/05_analyses.py

###
# Notebooks for manual feature analysis
###
# 02_experiments/singlecell/sae_hematopoiesis_dev.ipynb # SAE analysis of hematopoiesis development (global feature)
# 02_experiments/singlecell/analysis_local.ipynb # finding local features