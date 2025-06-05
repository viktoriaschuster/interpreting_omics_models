import os
import scvi
import anndata as ad
import mudata as md
import pandas as pd
import numpy as np
import torch

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
data = data[:, data.var['feature_types'] == 'GEX']
#print(data)
data = data[data.obs["train_val_test"] != "test"]
#print(data)
#data = data[::100, :]
data.var_names_make_unique()
#data.obs['modality'] = 'paired'
#data.obs['modality'] = data.obs['feature_types']
#data.var.modality = ["Gene Expression" if x == "GEX" else "Peaks" for x in data.var['modality'].values]
data.X = data.layers['counts'] # they want unnormalized data in X
#data.train_indices = np.where(data.obs['train_val_test'] == 'train')[0]
#data.validation_indices = np.where(data.obs['train_val_test'] == 'validation')[0]

# just make it a mudata file because updates to scvi have crashed the anndata setup apparently
#rna_data = data[:, data.var['feature_types'] == 'GEX'].copy()
#atac_data = data[:, data.var['feature_types'] == 'ATAC'].copy()
#mdata = md.MuData({"rna": rna_data, "atac": atac_data})
#mdata.obs['Site'] = data.obs['Site']
#del rna_data, atac_data

if model_version == 0:
    model_name = 'scvi_l20_l2'
    scvi.model.SCVI.setup_anndata(data, batch_key='Site')
    #scvi.model.MULTIVI.setup_anndata(data, batch_key='Site')
    scvi = scvi.model.SCVI(
        data, 
        #n_genes=len(data.var[data.var['feature_types'] == 'GEX']),
        #n_genes = data["GEX"].shape[1],
        #n_regions = data["ATAC"].shape[1],
        #n_hidden=100,
        n_latent=20,
        n_layers=2
    )
    max_epochs = 500
elif model_version == 1:
    model_name = 'scvi_l100_l2_h1000'
    exit()

#print(mvi.module.z_encoder_expression)
#print(mvi.module.z_encoder_accessibility)
#print(mvi.module.z_encoder_protein) # this should not exist
del data

###
# now train
###

max_epochs = 10
scvi.train(accelerator='gpu', devices=3, max_epochs=max_epochs, early_stopping=True)
scvi.save('03_results/models/scvi/'+model_name)

elbo = scvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')