# Paper Experiments

## Simulation Experiments (Section 4)

Autoencoder hyperparameter searches, training, SAE sweeps, and SAE training can be executed with commands in `simulation_runs.sh` as well as additional analyses. Plots for this section can be generated afterwards with `simulation_plots.sh`.

## Case Study: Extracting and annotating meaningful features from single-cell models (Section 5)

Embedding extraction, SAE training, and automated annotation analyses can be executed with commands in `singlecell_runs.sh`. Plots for this section can be generated afterwards with `singlecell_plots.sh`. Manual feature discovery was based on explorations in notebooks mentioned at the end of `singlecell_runs.sh`.

**Important note**: In order to be able to run scripts with Geneformer, follow [their instructions](https://huggingface.co/ctheodoris/Geneformer) to set up the environment and download the model. The run script includes the setup, but I experienced that several files had to be downloaded manually from the Hugging Face repository, so I recommend running the scripts one by one and not executing the whole exec script at once. Their purpose is to provide a general overview of the analysis pipeline, but they are not meant to be executed in one go. I also had to make some changes to the source code to be able to run the model with a specified GPU.