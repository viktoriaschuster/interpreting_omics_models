import os
import scvi
import anndata as ad
import mudata as md
import pandas as pd
import numpy as np
import torch

raise AttributeError("This does not work because of issues with MultiVI (matrix problems in model)")

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

dev_id = 3
device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")

model_version = 0 # for the one I used in multiDGD paper
#model_version = 1 # for the one I update with my findings here

##########################
# load model and data
##########################

if torch.cuda.is_available():
    print(f"Using GPU: {device}")
    data_dir = '/home/vschuste/data/singlecell/'
else:
    data_dir = '/Users/vschuste/Documents/work/data/singlecell/'

data = scvi.data.read_h5ad(data_dir+'human_bonemarrow.h5ad')
#data = ad.read_h5ad(data_dir+'human_bonemarrow.h5ad')
#print(data)
data = data[data.obs["train_val_test"] != "test"]
#print(data)
#data = data[::100, :]
data.var_names_make_unique()
data.obs['modality'] = 'paired'
#data.obs['modality'] = data.obs['feature_types']
#data.var.modality = ["Gene Expression" if x == "GEX" else "Peaks" for x in data.var['modality'].values]
data.X = data.layers['counts'] # they want unnormalized data in X
#data.train_indices = np.where(data.obs['train_val_test'] == 'train')[0]
#data.validation_indices = np.where(data.obs['train_val_test'] == 'validation')[0]

# just make it a mudata file because updates to scvi have crashed the anndata setup apparently
rna_data = data[:, data.var['feature_types'] == 'GEX'].copy()
atac_data = data[:, data.var['feature_types'] == 'ATAC'].copy()
mdata = md.MuData({"rna": rna_data, "atac": atac_data})
mdata.obs['Site'] = data.obs['Site']
del rna_data, atac_data

if model_version == 0:
    model_name = 'multivi_l20_e2_d2'
    scvi.model.MULTIVI.setup_mudata(mdata, batch_key='Site', modalities={"rna_layer": "rna", "protein_layer": "atac"})
    #scvi.model.MULTIVI.setup_anndata(data, batch_key='Site')
    mvi = scvi.model.MULTIVI(
        mdata, 
        n_genes=len(data.var[data.var['feature_types'] == 'GEX']),
        n_regions=(len(data.var)-len(data.var[data.var['feature_types'] == 'GEX'])),
        #n_genes = data["GEX"].shape[1],
        #n_regions = data["ATAC"].shape[1],
        #n_hidden=100,
        n_latent=20,
        n_layers_encoder=2,
        n_layers_decoder=2
    )
    max_epochs = 500
elif model_version == 1:
    model_name = 'multivi_l100_e2_d2_h1000'
    #scvi.model.MULTIVI.setup_mudata(mdata, batch_key='Site')
    mvi = scvi.model.MULTIVI(
        data, 
        n_genes=len(data.var[data.var['feature_types'] == 'GEX']),
        n_regions=(len(data.var)-len(data.var[data.var['feature_types'] == 'GEX'])),
        n_hidden=1000,
        n_latent=100,
        n_layers_encoder=2,
        n_layers_decoder=2
    )
    max_epochs = 1000

#print(mvi.module.z_encoder_expression)
#print(mvi.module.z_encoder_accessibility)
#print(mvi.module.z_encoder_protein) # this should not exist
del data

###
# now train
###

max_epochs = 10
mvi.train(accelerator='gpu', devices=3, max_epochs=max_epochs, early_stopping=True)
mvi.save('03_results/models/multivi/'+model_name)

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')