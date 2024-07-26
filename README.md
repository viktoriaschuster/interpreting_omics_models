# project_template
This repository helps me set up deep learning projects quickly and consistently. It contains a pre-defined structure and guides on how to set up as a package.

## Environments

It would be beneficial to either use a fitting environment or set up a project-specific one.

### Conda

```bash
conda create -n project_name python=3.11
conda activate project_name
pip install -r requirements.txt
```

### Install the package (if applicable)

To install the package, run the following command in the root directory of the project:

```bash
python -m pip install .
```