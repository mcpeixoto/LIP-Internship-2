import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torch.utils.data.dataset import TensorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch.optim import Adam
from pytorch_lightning import Trainer
import os
import numpy as np
import pandas as pd
from os.path import join
from typing import Optional
from config import processed_data_path
from sklearn.utils import shuffle

# Defining the dataset
class _dataset(Dataset): #
    def __init__(self, key, type, random_seed=42):
        # TODO: Improve efficiency/handle names
        
        # Check if key is valid
        assert key in {'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'}, "Invalid Key!"

        # With specified key, get data
        file = join(processed_data_path, key+".csv")
        data = pd.read_csv(file, index_col=0)

        # Shuffle the dataframe
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # This will equally devide the dataset into 
        # train, validation and test
        train, validation, test = np.split(data.sample(frac=1), [int(len(data)*(1/3)), int(len(data)*(2/3))])
        
        if type == "train":
            data = train
        elif type == "validation":
            data = validation
        elif type == "test":
            data = test
        
        # This data we want on a seperate variable
        self.weights = torch.from_numpy(data["weights"].to_numpy(dtype=np.float32))

        data.drop(["name", "weights"], axis=1, inplace=True)

        self.data = torch.from_numpy(data.to_numpy(dtype=np.float32))
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.weights[index]

    def __len__(self):
        return self.n_samples



class VAE(pl.LightningModule):
    def __init__(self, dataset, batch_size, hidden_size, alpha, lr):
        """
        Args:
        - > key e {'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'}; it's the type of data
        - > hidden_size : Latent Hidden Size
        - > alpha : Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        - > lr : Learning Rate, will not be used if auto_lr_find is used.
        - > dataset : Dataset to used
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.alpha = alpha

        # Architecture
        self.encoder = nn.Sequential(
            nn.Linear(69, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(), 
            nn.Linear(128, hidden_size),
            nn.LeakyReLU()
        )

        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 69), 
            nn.LeakyReLU(),
        )

    def encode(self, x):
        # Pass through encoder
        out = self.encoder(x)
        mu = self.hidden2mu(out)
        log_var = self.hidden2log_var(out)
        return mu, log_var

    def decode(self, x):
        # Pass through encoder
        return self.decoder(x)

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick
        # It outputs a sample of the dist.
        # mu -> average | log_var -> std
        
        #e = torch.randn(size=(mu.size(0), mu.size(1))) 
        #e = e.type_as(mu) # To the same device

        log_var = torch.exp(0.5*log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1))) # Epsilon, normal distribution
        z = z.type_as(mu)
        return mu + log_var*z

    def forward(self, x):
        # Pass through encoder
        mu, log_var = self.encode(x)
        # Reparametrization Trick
        hidden = self.reparametrize(mu, log_var)
        # Pass through decoder
        output = self.decoder(hidden)

        return mu, log_var, output

    def training_step(self, batch, batch_idx):
        x, weights = batch
        mu, epsilon, x_out = self.forward(x)

        # kl loss é a loss da distribuição - disentangle auto encoders?
        # kl aparece para a distribuição nao ser mt diferente da normal

        # a loss é a loss de reconstrução mais a kl loss!
        #kl_loss = (-0.5*(1+torch.log(epsilon**2)-epsilon**2 - mu**2).sum(dim=1)).mean(dim=0) 
        kl_loss = (-0.5*(1+epsilon - mu**2 -
                         torch.exp(epsilon)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        # Weights on final loss
        loss = (weights * loss) / weights.sum()
        loss = torch.mean(loss, dtype=torch.float32)

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, weights = batch

        mu, epsilon, x_out = self.forward(x)

        # K-L Loss
        #kl_loss = (-0.5*(1+torch.log(epsilon**2)-epsilon**2 - mu**2).sum(dim=1)).mean(dim=0) 
        kl_loss = (-0.5*(1+epsilon - mu**2 -
                         torch.exp(epsilon)).sum(dim=1)).mean(dim=0)
        # Weights on KL Loss
        kl_loss = (weights * kl_loss) / weights.sum()
        kl_loss = torch.mean(kl_loss, dtype=torch.float32)

        # Reconstruction loss
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # Weights on recon loss
        recon_loss = (weights * recon_loss) / weights.sum()
        recon_loss = torch.mean(recon_loss, dtype=torch.float32)


        loss = recon_loss*self.alpha + kl_loss

        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return x_out, loss

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    # Functions for dataloading
    def train_dataloader(self):
        train_set = _dataset(self.dataset, type="train")
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=6)

    def val_dataloader(self):
        val_set = _dataset(self.dataset, type="validation")
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=6)



from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
if __name__ == "__main__":
    trainer = Trainer(
        fast_dev_run = True,
        gpus=1,
        auto_lr_find=True,
        max_epochs=500,
        #max_time=
        callbacks=[EarlyStopping(monitor="val_loss", patience=50), ModelCheckpoint(dirpath="models", monitor="val_loss", mode="min")]

        )
    model = VAE(
    dataset = "bkg",
    hidden_size=5,
    batch_size = 1024,
    alpha = 1, 
    lr = 0.001,
    )
    #trainer.tune(model) 
    trainer.fit(model)
