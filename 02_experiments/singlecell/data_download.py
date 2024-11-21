import requests, zipfile

###
# data
###

# Download
file_name = 'human_bonemarrow.h5ad.zip'
file_url = 'https://api.figshare.com/v2/articles/23796198/files/41740251'

file_response = requests.get(file_url).json()
file_download_url = file_response['download_url']
response = requests.get(file_download_url, stream=True)
with open(file_name, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Unzip
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall('./01_data')
# now delete the zip file
import os
os.remove(file_name)

###
# model
###

model_name = 'human_bonemarrow_l20_h2-3_test50e'

try:
    from multiDGD.utils.model_loading import *

    # download the model file
    get_figshare_file('/articles/23796198/files/41735907', model_name+'.pt')
    # download the hyperparameters file
    get_figshare_file('/articles/23796198/files/41735904', model_name+'_hyperparameters.json')
except:
    print('Could not download the model and hyperparameters file. Are you sure you installed multiDGD?')