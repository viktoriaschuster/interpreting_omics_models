import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import wandb

def train_and_eval_model(
        encoder,
        decoder,
        rna_counts,
        n_samples,
        n_samples_validation,
        learning_rate=1e-4,
        n_epochs=20000,
        plotting=False,
        batch_size=128,
        weight_decay=0,
        early_stopping=None,
        loss_type="MSE",
        log_wandb=False
):
    # loss function
    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif loss_type == "PoissonNLL":
        loss_fn = torch.nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean', eps=1e-3)
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))

    # dataloader
    train_loader = torch.utils.data.DataLoader(rna_counts[:n_samples_validation,:], batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(rna_counts[n_samples_validation:,:], batch_size=batch_size, shuffle=True)

    # optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=weight_decay)

    # train
    train_loss = []
    val_loss = []

    for e in tqdm.tqdm(range(n_epochs)):
        train_loss_temp = 0

        for data in train_loader:
            # forward pass
            encoded = encoder(data)
            decoded = decoder(encoded)

            # compute loss
            loss = loss_fn(decoded, data)
            train_loss_temp += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
        
            # update weights
            optimizer.step()

        train_loss.append(train_loss_temp/len(train_loader))

        # validation
        val_loss_temp = 0
        for data in val_loader:
            # forward pass
            encoded = encoder(data)
            decoded = decoder(encoded)

            # compute loss
            loss = loss_fn(decoded, data)
            val_loss_temp += loss.item()

        val_loss.append(val_loss_temp/len(val_loader))

        # early stopping
        if early_stopping is not None:
            if e > early_stopping:
                # check if we have achieved a new minimum within the last early_stopping epochs
                if min(val_loss[-early_stopping:]) > min(val_loss):
                    print("Early stopping at epoch ", e)
                    break
        
        if log_wandb:
            wandb.log({'train_loss': train_loss[-1], 'val_loss': val_loss[-1]})
    
    if plotting:
        # plot loss curves
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='validation')
        plt.legend()
        plt.show()

        # make scatter plots for each gene with output vs input
        # get the output
        encoded = encoder(rna_counts)
        decoded = decoder(encoded)
        # plot subplots
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        for i in range(5):
            axs[i].scatter(rna_counts[:, i].detach().cpu().numpy(), decoded[:, i].detach().cpu().numpy(), s=1)
            axs[i].set_xlabel('input')
            axs[i].set_ylabel('output')
            axs[i].set_title('gene {}'.format(i))
        plt.show()

        """
        # rank underlying features by their importance
        # first, get the weights of the encoder
        encoder_weights = encoder[0].weight.detach().cpu().t().numpy()
        # then, get the weights of the decoder
        decoder_weights = decoder[-2].weight.detach().cpu().t().numpy()
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
        plt.show()
        """

    history = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': np.arange(len(train_loss))})

    return encoder, decoder, history

def train_and_eval_model2(encoder, decoder, d_decoder, rna_counts, n_samples, n_samples_validation, learning_rate=1e-4, n_epochs=20000, plotting=False, warmup_steps=1000, pretrain=0, beta_max=1, adversarial_scheme='basic'):
    # loss function
    #loss_fn = torch.nn.MSELoss(reduction='mean')
    loss_fn = get_loss(scheme=adversarial_scheme)
    adv_loss_fn = get_adversarial_loss(scheme=adversarial_scheme) # defining how to calculate the loss

    # dataloader
    train_loader = torch.utils.data.DataLoader(rna_counts[:n_samples_validation,:], batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(rna_counts[n_samples_validation:,:], batch_size=128, shuffle=True)

    # optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    d_optimizer = torch.optim.Adam(list(d_decoder.parameters()), lr=learning_rate)

    # train
    train_loss = []
    gen_loss = []
    dis_loss = []
    val_loss = []

    # create a warmup phase for the discriminator
    beta = 0
    beta_increment = beta_max/warmup_steps

    for e in tqdm.tqdm(range(n_epochs)):
        train_loss_temp = 0
        gen_loss_temp = 0
        dis_loss_temp = 0

        for data in train_loader:
            # forward pass
            encoded = encoder(data)
            d_decoded = d_decoder(encoded)

            ## first the superposition discriminator
            # compute loss
            #d_loss = loss_fn(d_decoded, data)
            d_loss = adv_loss_fn(d_decoded, data)

            # backward pass
            optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
        
            # update weights
            d_optimizer.step()

            encoded = encoder(data)
            decoded = decoder(encoded)
            d_decoded = d_decoder(encoded)

            # now the model
            loss = loss_fn(decoded, data)
            #d_loss = loss_fn(d_decoded, data)
            d_loss = adv_loss_fn(d_decoded, data)

            gen_loss_temp += loss.item()
            dis_loss_temp += d_loss.item()
            train_loss_temp += loss.item()
            train_loss_temp -= (beta * d_loss.item())

            d_optimizer.zero_grad()
            optimizer.zero_grad()

            g_loss = loss - (beta * d_loss)
            g_loss.backward()

            optimizer.step()

        train_loss.append(train_loss_temp/len(train_loader))
        gen_loss.append(gen_loss_temp/len(train_loader))
        dis_loss.append(dis_loss_temp/len(train_loader))

        # validation
        val_loss_temp = 0
        for data in val_loader:
            # forward pass
            encoded = encoder(data)
            decoded = decoder(encoded)

            # compute loss
            loss = loss_fn(decoded, data)
            val_loss_temp += loss.item()

        val_loss.append(val_loss_temp/len(val_loader))

        # update beta
        if e >= pretrain:
            beta = min(beta_max, beta + beta_increment)
    
    if plotting:
        # plot loss curves
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='validation')
        plt.legend()
        plt.show()

        # make scatter plots for each gene with output vs input
        # get the output
        encoded = encoder(rna_counts)
        decoded = decoder(encoded)
        # plot subplots
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        for i in range(5):
            axs[i].scatter(rna_counts[:, i].detach().cpu().numpy(), decoded[:, i].detach().cpu().numpy(), s=1)
            axs[i].set_xlabel('input')
            axs[i].set_ylabel('output')
            axs[i].set_title('gene {}'.format(i))
        plt.show()

    history = pd.DataFrame({
        'train_loss': train_loss,
        'g_loss': gen_loss,
        'd_loss': dis_loss,
        'val_loss': val_loss,
        'epoch': np.arange(n_epochs)
    })

    return encoder, decoder, d_decoder, history

class r_square_loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(r_square_loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, overflow=1e-6):
        # input shapes are (N, M)
        means = torch.mean(y_true, dim=0)
        ss_tot = torch.sum((y_true - means)**2, dim=0)
        ss_res = torch.sum((y_true - y_pred)**2, dim=0)
        #r2 = 1 - ss_res/ss_tot
        # we want the loss to be better if smaller
        r2 = ss_res/(ss_tot + overflow) # so it is not really r2
        if self.reduction == 'mean':
            return torch.mean(r2)
        elif self.reduction == 'sum':
            return torch.sum(r2)
        else:
            return r2

class MSELossNorm(torch.nn.Module):
    '''
    This is a custom loss function that returns the mean squared error loss
    scaled to 0-1
    '''
    def __init__(self, reduction='mean'):
        super(MSELossNorm, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # input shapes are (N, M)
        loss = torch.mean((y_pred - y_true)**2, dim=1)
        norm_loss = (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss))
        if self.reduction == 'mean':
            return torch.mean(norm_loss)
        elif self.reduction == 'sum':
            return torch.sum(norm_loss)
        else:
            return norm_loss

def get_adversarial_loss(scheme='basic'):
    if scheme == 'basic':
        return torch.nn.MSELoss(reduction='mean')
    elif scheme == 'r2':
        return r_square_loss(reduction='mean')
    else:
        raise ValueError('Unknown adversarial loss scheme: {}'.format(scheme))

def get_loss(scheme='basic'):
    if scheme == 'basic':
        return torch.nn.MSELoss(reduction='mean')
    elif scheme == 'r2':
        return MSELossNorm(reduction='mean')
    else:
        raise ValueError('Unknown loss scheme: {}'.format(scheme))