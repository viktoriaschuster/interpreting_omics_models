# Can sparse autoencoders make sense of gene expression latent variable models?

This is the repository accompanying my paper "Can sparse autoencoders make sense of gene expression latent variable models?". It is has been cleaned up for the submission, which is available in branch [paper_cleanup](https://github.com/viktoriaschuster/interpreting_omics_models/tree/paper_cleanup). That branch presents a clean and minimal setup to reproduce the results of the paper. Go into the [02_experiments](https://github.com/viktoriaschuster/interpreting_omics_models/tree/paper_cleanup/02_experiments) folder to see the guide on how to run the experiments. As the paper, the code is structured around simulation and single-cell experiments. The simulation experiments are designed to test the performance of sparse autoencoders on latent variable models, while the single-cell experiments apply the methods to real-world data.

## Environments

It would be beneficial to either use a fitting environment or set up a project-specific one. I made the following for this project:

### Conda

```bash
conda create -n sc_mechinterp python=3.9
conda activate sc_mechinterp
```

### Get the necessary sc packages

For using multiDGD, the package has to be installed before the remaining requirements.

```bash
pip install multiDGD@git+https://github.com/Center-for-Health-Data-Science/multiDGD
```

### THEN install the remaining requirements

```bash
pip install -r requirements.txt
```

## scFeatureLens

The tool that I developed for this paper is available in the [scFeatureLens](https://github.com/viktoriaschuster/sc_mechinterp/tree/main/tools/scFeatureLens) repository. It is in its baby stage, so I am very happy to get feedback on any issues or desired features.