import numpy as np
import torch
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.linear_model import LinearRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 5
MOD_NAME = 'l4_s'+str(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

###
# data
###
# load or generate the data
data_path = "01_data/"
rna_counts = torch.tensor(np.load(data_path + "sim_rna_counts.npy"))
tf_scores = torch.tensor(np.load(data_path + "sim_tf_scores.npy"))
activity_score = torch.tensor(np.load(data_path + "sim_activity_scores.npy"))
accessibility_scores = torch.tensor(np.load(data_path + "sim_accessibility_scores.npy"))

# plot setups
loss_colors = sns.color_palette(n_colors=2)
superposition_colors = sns.color_palette('colorblind', 3)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

###
# function
###

def train_and_eval_model(encoder, decoder, rna_counts, n_samples, n_samples_validation, learning_rate=1e-4):
    # loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # train
    train_loss = []
    val_loss = []

    for e in range(n_epochs):
        # train
        # forward pass
        encoded = encoder(rna_counts[:(n_samples-n_samples_validation)])
        decoded = decoder(encoded)

        # compute loss
        loss = loss_fn(decoded, rna_counts[:(n_samples-n_samples_validation)])
        train_loss.append(loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # validation
        # forward pass
        encoded = encoder(rna_counts[(n_samples-n_samples_validation):])
        decoded = decoder(encoded)

        # compute loss
        loss = loss_fn(decoded, rna_counts[(n_samples-n_samples_validation):])
        val_loss.append(loss.item())
    
    # plot loss curves
    import matplotlib.pyplot as plt
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.legend()
    # save in temp_plots
    plt.savefig('03_results/reports/temp_plots/sim1_'+MOD_NAME+'_ae_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # make scatter plots for each gene with output vs input
    # get the output
    encoded = encoder(rna_counts)
    decoded = decoder(encoded)
    # plot subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(5):
        axs[i].scatter(rna_counts[:, i].detach().numpy(), decoded[:, i].detach().numpy(), s=1)
        axs[i].set_xlabel('input')
        axs[i].set_ylabel('output')
        axs[i].set_title('gene {}'.format(i))
    # save in temp_plots
    plt.savefig('03_results/reports/temp_plots/sim1_'+MOD_NAME+'_ae_train_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # rank underlying features by their importance
    # first, get the weights of the encoder
    encoder_weights = encoder[0].weight.detach().t().numpy()
    # then, get the weights of the decoder
    decoder_weights = decoder[-2].weight.detach().t().numpy()
    # multiply them together
    weights = np.matmul(encoder_weights, decoder_weights)
    # get the absolute values
    weights = np.abs(weights)
    # sum the weights for each gene
    weights = np.sum(weights, axis=1)
    # sort the weights
    #weights = np.argsort(weights)

    # plot the weights
    plt.bar(np.arange(5), weights)
    plt.xlabel('gene')
    plt.ylabel('weight')
    plt.title('gene importance')
    plt.savefig('03_results/reports/temp_plots/sim1_'+MOD_NAME+'_ae_train_gene_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    history = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': np.arange(n_epochs)})

    return encoder, decoder, history

###
# set up model
###
encoder = torch.nn.Sequential(
    torch.nn.Linear(5, 4)
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(4, 5),
    torch.nn.ReLU()
)

# training
n_samples_validation = 2000
n_samples = 10000 + n_samples_validation
n_epochs = 20000
encoder, decoder, history = train_and_eval_model(encoder, decoder, rna_counts, n_samples, n_samples_validation)

# also compute the superpositions
reps = encoder(rna_counts).detach()
# plot the superpositions
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1,3, figure=fig)
ax_list = []
for j in range(tf_scores.shape[1]):
    reg = LinearRegression().fit(reps.detach().cpu().numpy(), tf_scores[:, j].detach().numpy())
    superposition = np.matmul(reps.detach().numpy(), reg.coef_.T)
    ax_list.append(fig.add_subplot(gs[0, j]))
    # add a black line in the background for y=x
    ax_list[-1].plot(superposition, superposition, color='black')
    sns.scatterplot(x=tf_scores[:, j].detach().numpy(), y=superposition, ax=ax_list[-1])
    # plot the regression R value in the top right corner
    ax_list[-1].text(0.55, 0.05, f'R={reg.score(reps.detach().numpy(), tf_scores[:, j].detach().numpy()):.2f}', transform=ax_list[-1].transAxes)
    ax_list[-1].set_ylabel('Superposition')
# save in temp_plots
plt.savefig('03_results/reports/temp_plots/sim1_'+MOD_NAME+'_ae_train_superpositions.png', dpi=300, bbox_inches='tight')
plt.close()

# save this model as the best one
model_name = 'layer1_latent4_seed'+str(seed)

torch.save(encoder, '03_results/models/sim1_'+model_name+'_encoder.pth')
torch.save(decoder, '03_results/models/sim1_'+model_name+'_decoder.pth')

# also save the history of the training
history.to_csv('03_results/models/sim1_'+model_name+'_history.csv', index=False)