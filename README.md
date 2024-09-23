# project_template
This repository helps me set up deep learning projects quickly and consistently. It contains a pre-defined structure and guides on how to set up as a package.

## Environments

It would be beneficial to either use a fitting environment or set up a project-specific one.

### Conda

```bash
conda create -n sc_mechinterp python=3.9
conda activate sc_mechinterp

#pip install -r requirements.txt
```

### Get the necessary sc packages

```bash
pip install multiDGD@git+https://github.com/Center-for-Health-Data-Science/multiDGD
```