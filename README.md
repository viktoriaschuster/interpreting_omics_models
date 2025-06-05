# SAEs for Biology
This is the start of a project using sparse autoencoders (SAEs) for interpretability in biological models. **It is being cleaned up now for a current paper submission.**

## Environments

It would be beneficial to either use a fitting environment or set up a project-specific one. I made the following for this project:

### Conda

```bash
conda create -n sc_mechinterp python=3.9
conda activate sc_mechinterp
```

For working on a server, the environment can be reproduced as follows:

```bash
conda env export -f env_no_version_numbers.yml --no-builds
conda env create -f env_no_version_numbers.yml # worked except for multidgd cause not in pip
```

### Get the necessary sc packages

```bash
pip install multiDGD@git+https://github.com/Center-for-Health-Data-Science/multiDGD
```

### THEN install the remaining requirements

```bash
pip install -r requirements.txt
```

## Single-cell Experiments

### Download data and models

```bash
python 02_experiments/singlecell/data_download.py
```
