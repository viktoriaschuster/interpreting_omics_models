import requests, zipfile
import os
from multiDGD.utils.model_loading import get_figshare_file

###
# data
###

# Download the human bone marrow data
file_name = 'human_bonemarrow.h5ad'
if not os.path.exists('./01_data/'+file_name):
    print('Downloading human bone marrow data')
    file_url = 'https://api.figshare.com/v2/articles/23796198/files/41740251'

    file_response = requests.get(file_url).json()
    file_download_url = file_response['download_url']
    response = requests.get(file_download_url, stream=True)
    with open(file_name+'.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # Unzip
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('./01_data')
    # now delete the zip file
    os.remove(file_name)
model_name = 'human_bonemarrow_l20_h2-3_test50e'
if not os.path.exists('03_resuts/models/'+model_name+'.pt'):
    try:
        # download the model file
        get_figshare_file('/articles/23796198/files/41735907', '03_results/models/'+model_name+'.pt')
        # download the hyperparameters file
        get_figshare_file('/articles/23796198/files/41735904', '03_results/models/'+model_name+'_hyperparameters.json')
    except:
        print('Could not download the model and hyperparameters file. Are you sure you installed multiDGD?')

# next mouse gastrulation
file_name = 'mouse_gastrulation.h5mu'
if not os.path.exists('./01_data/'+file_name):
    print('Downloading mouse gastrulation data')
    file_url = 'https://api.figshare.com/v2/articles/23796198/files/41740323'

    file_response = requests.get(file_url).json()
    file_download_url = file_response['download_url']
    response = requests.get(file_download_url, stream=True)
    with open(file_name+'.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # Unzip
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('./01_data')
    # now delete the zip file
    os.remove(file_name)
model_name = 'mouse_gastrulation_l20_h2-3'
if not os.path.exists('03_resuts/models/'+model_name+'.pt'):
    print('Downloading mouse gastrulation model')
    #try:
    get_figshare_file('/articles/23796198/files/41735919', '03_results/models/'+model_name+'.pt')
    get_figshare_file('/articles/23796198/files/41735916', '03_results/models/'+model_name+'_hyperparameters.json')

# next human brain
file_name = 'human_brain.h5mu'
if not os.path.exists('./01_data/'+file_name):
    print('Downloading human brain data')
    file_url = 'https://api.figshare.com/v2/articles/23796198/files/41736012'

    file_response = requests.get(file_url).json()
    file_download_url = file_response['download_url']
    response = requests.get(file_download_url, stream=True)
    with open(file_name+'.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # Unzip
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('./01_data')
    # now delete the zip file
    os.remove(file_name)
model_name = 'human_brain_l20_h2-3'
if not os.path.exists('03_resuts/models/'+model_name+'.pt'):
    print('Downloading human brain model')
    try:
        get_figshare_file('/articles/23796198/files/41735913', '03_results/models/'+model_name+'.pt')
        get_figshare_file('/articles/23796198/files/41735910', '03_results/models/'+model_name+'_hyperparameters.json')
    except:
        print('Could not download the model and hyperparameters file. Are you sure you installed multiDGD?')
if not os.path.exists('./01_data/human_brain.h5mu'):
    try:
        # unzip the data
        print('Unzipping data')
        with zipfile.ZipFile('./01_data/human_brain.h5mu.zip', 'r') as zip_ref:
            zip_ref.extractall('./01_data/')
    except:
        raise Exception('Please download the data first')