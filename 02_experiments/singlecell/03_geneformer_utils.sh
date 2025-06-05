conda create -n sc_mechinterp_geneformer python=3.11
conda activate sc_mechinterp_geneformer

git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .